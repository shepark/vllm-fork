###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import habana_frameworks.torch as htorch
from typing import List, Optional, Tuple

def silu_and_mul(output, input):
    htorch.core.mark_step()
    d = input.shape[-1] // 2
    silu = torch.nn.SiLU().to(input.device)
    x, y = torch.split(input, d, dim=-1)
    output.copy_(silu(x) * y)
    htorch.core.mark_step()

def gelu_new(output, input):
    raise NotImplementedError

def gelu_fast(output, input):
    raise NotImplementedError

def _paged_attention_masked_fill(
    attn_weights_blocks,
    value_blocks,
    context_lens,
    attn_weights_blocks_filler,
    value_blocks_filler,
    block_size,
    max_num_blocks_per_seq,
    num_seqs,
    sanitize_values,
    device,
):
    # NOTE (kzawora): this code performs unconditinal out-of-bound cleanup on attention weights.
    # It was pretty insane to write and is probably hard to read, but it allows us to avoid
    # recompilations and D2H-H2D copies on Gaudi2, making it very efficient.

    # First, we're filling full out-of bound blocks. We want to create 2D mask [num_seqs, max_num_blocks_per_seq]
    # indicating which blocks need to be cleaned

    # Create [num_seqs, max_num_blocks_per_seq] tensor of block indices per each sequence,
    # which we'll then transform into a boolean tensor with mask
    block_indices = torch.arange(max_num_blocks_per_seq, dtype=torch.int64, device=device).view(1, -1)
    block_indices = block_indices.expand(num_seqs, block_indices.size(1))

    # Create mask with 1s for all blocks that are fully out of bound, and 0s for the rest.
    # In order to broadcast the mask across all dimensions, we need to transpose it and
    # view it as 5D tensor with ones in broadcasted dimensions (max_num_blocks_per_seq, num_seqs, 1, 1, 1)
    attn_weights_blocks_mask = (block_indices >= (torch.ceil(context_lens / block_size)).unsqueeze(-1)).T.view(
        max_num_blocks_per_seq, num_seqs, 1, 1, 1
    )

    # Apply block mask to attenton weights
    attn_weights_blocks.masked_fill_(attn_weights_blocks_mask, attn_weights_blocks_filler)

    # We're done with filling full OoB blocks. Now, we need to fill out-of-bound values within last blocks
    # The problem here is that now, we'll need to fetch all last blocks of each sequence, and fill
    # the out-of-bound activation in the last dimension (block_size). This is pretty hard to do without
    # loops and conditons.

    # Collect last block indices. This will include blocks that are both partially, and fully filled.
    # We expect this index to be in bounds (< max_blocks_per_seq).
    last_block_indices = (torch.ceil((context_lens / block_size)) - 1).to(torch.int64)

    # Gather indices of last blocks. We will collect plenty of superfluous blocks,
    # as we'll fetch all (num_seq) indices per each sequence. This will result in
    # (num_seq, num_seq, num_query_heads, 1, block_size) tensor.
    last_blocks = attn_weights_blocks.index_select(0, last_block_indices)

    # Extract only relevant blocks. Since dim0 and dim1 are the same, and we passed last_block_indices in order,
    # we can reduce these dimensions by extracting the diagonal value. torch.diagonal returns the extracted value
    # as the last dimension, so we'll need to permute the tensor to get it back to the first one.
    # We expect to transform the source (num_seq, num_seq, num_query_heads, 1, block_size) tensor into
    # (num_seq, num_query_heads, 1, block_size) tensor, with the first dimension containing each sequence's last block.
    last_blocks_diag = torch.diagonal(last_blocks, dim1=0, dim2=1, offset=0).permute((3, 0, 1, 2))

    # Similarly to block mask, we'll create s 2D tensor of token indices per each block,
    # which we'll then transform into a boolean tensor with mask
    seq_indices = torch.arange(block_size, dtype=torch.int64, device=device).view(1, -1)
    seq_indices = seq_indices.expand(num_seqs, seq_indices.size(1))

    # Create mask with 1s for all tokens that are fully out of bound, and 0s for the rest.
    # We apply a bias of block_size for sequences that have context length divisible by block_size,
    # as we don't want to clear anything within their last block - it is fully filled
    last_block_offsets = (context_lens % block_size + block_size * (context_lens % block_size == 0)).view(-1, 1)
    seq_mask = seq_indices >= last_block_offsets

    # Apply block mask to weights to diagonal (num_seq, num_query_heads, 1, block_size) tensor.
    last_blocks_diag.masked_fill_(seq_mask.view(num_seqs, 1, 1, block_size), attn_weights_blocks_filler)

    # Scatter the (num_seq, num_query_heads, 1, block_size) tensor back into attn_weights_blocks using
    # the same indices as we did in gathering. Each "row" will be stored at (last_block_index[i],i),
    # where i is sequence index.
    seq_idx = torch.arange(num_seqs, dtype=torch.int64, device=device)
    edge_indices = (last_block_indices, seq_idx)
    attn_weights_blocks.index_put_(edge_indices, last_blocks_diag)

    # NOTE(kzawora): For whatever reason, we also need to cleanup values, otherwise vLLM inference will return garbage.
    # I'll need to figure out why. For now, we'll follow exactly the same steps as we did with weight cleanup.
    if sanitize_values:
        # For filling fully OoB value blocks, we can use the same mask as we did previously.
        value_blocks.masked_fill_(attn_weights_blocks_mask, value_blocks_filler)

        # Same thing for filling partial blocks - obtain all last blocks for all sequences
        last_values = value_blocks.index_select(0, last_block_indices)  # [num_seqs, num_seqs, num_kv_heads, block_size,  head_size]

        # Reduce the tensor to have only a single last block per each sequence
        last_values_diag = torch.diagonal(last_values, dim1=0, dim2=1, offset=0).permute((3, 0, 1, 2))

        # Mask blocks
        last_values_diag.masked_fill_(seq_mask.view(num_seqs, 1, block_size, 1), value_blocks_filler)

        # Put masked blocks back in their place
        value_blocks.index_put_(edge_indices, last_values_diag)

    # Phew, that was a lot.

def paged_attention_v1(
    query_in,
    key_cache_in,
    value_cache_in,
    head_mapping,
    scale,
    block_tables,
    context_lens,
    block_size,
    max_context_len,
    alibi_slopes,
    attn_masks=None,
) -> None:
    sanitize_values = True
    device = query_in.device
    query = query_in
    key_cache = key_cache_in
    value_cache = value_cache_in
    num_kv_heads = value_cache[0].shape[0]
    head_size = value_cache[0].shape[1]
    block_size = value_cache[0].shape[2]
    num_seqs = query.shape[0]
    num_query_heads = query.shape[1]
    max_num_blocks_per_seq = block_tables.shape[1]

    if alibi_slopes:
        raise NotImplementedError

    num_queries_per_kv = num_query_heads // num_kv_heads
    attn_weights_blocks = torch.full(
        (max_num_blocks_per_seq, num_seqs, num_query_heads, 1, block_size), torch.finfo(query.dtype).min, dtype=query.dtype, device=device
    )
    value_blocks = torch.zeros((max_num_blocks_per_seq, num_seqs, num_kv_heads, block_size, head_size), dtype=query.dtype, device=device)
    seq_index = torch.tensor([0], dtype=torch.int64, device=device)
    block_index = torch.tensor([0], dtype=torch.int64, device=device)
    seq_block_table_idx = torch.tensor([0], dtype=torch.int64, device=device)
    seq_context_len = torch.tensor([0], dtype=torch.int64, device=device)
    for i in range(0, max_num_blocks_per_seq):  # can run in parallel
        with torch.profiler.record_function(f"block_loop"):
            with torch.profiler.record_function("seq_index_fill"):
                seq_index.fill_(0)

            ## hard override for filler. These blocks would contribute nothing to the output due to zero attention_probs and will clog up compute resources
            # with torch.profiler.record_function('block_seq_len_check'):
            #    if (block_index - 2) * block_size > torch.max(context_lens):
            #        break

            with torch.profiler.record_function("slice_weights_values"):
                # single block attn weight of shape [B, Hq, Mq(=1), block_size], equivalent to attn_weights_blocks[i]
                attn_weights = attn_weights_blocks.index_select(0, block_index).squeeze(0)

            with torch.profiler.record_function("fetch_block_table"):
                block_table = block_tables.index_select(1, block_index).squeeze(1)

            with torch.profiler.record_function("fetch_block_keys"):
                keys = torch.index_select(key_cache, 0, block_table)

            with torch.profiler.record_function("fetch_block_values"):
                values = torch.index_select(value_cache, 0, block_table)

            with torch.profiler.record_function("store_block_values"):
                value_blocks.index_copy_(0, block_index, values.permute((0, 1, 3, 2)).unsqueeze(0))

            with torch.profiler.record_function("repeat_interleave_seq_key_check"):
                if num_queries_per_kv > 1:
                    with torch.profiler.record_function("repeat_interleave_seq_key_check_true"):
                        # Handle MQA and GQA
                        keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            with torch.profiler.record_function("gemm_attn_weight"):
                q_bmm = query.reshape(1, num_seqs * num_query_heads, head_size).transpose(0, 1)
                k_bmm = keys.reshape(num_seqs * num_query_heads, head_size, block_size)
                attn_weights_2 = scale * torch.matmul(q_bmm, k_bmm).reshape(num_seqs, num_query_heads, 1, block_size)
            # FIXME: need to restore attention masks in a more sensible way
            if attn_masks is not None:
                for j in range(num_seqs):  # can run in parallel
                    with torch.profiler.record_function(f"seq_loop"):
                        with torch.profiler.record_function("seq_context_len_fill"):
                            seq_context_len[0] = context_lens.index_select(0, seq_index)[0]
                        with torch.profiler.record_function("context_len_check"):
                            if (block_index.mul(block_size)).lt(context_lens.index_select(0, seq_index)):
                                with torch.profiler.record_function("context_len_check_true"):

                                    attn_weight = torch.index_select(attn_weights_2, 0, seq_index).squeeze(0)
                                    with torch.profiler.record_function("create_attn_mask_check"):
                                        if attn_masks is not None:
                                            with torch.profiler.record_function("create_attn_mask_check_true"):
                                                attn_mask = torch.index_select(attn_masks.index_select(0, block_index), 0, seq_index)
                                                attn_weight = torch.masked_fill(
                                                    attn_weight, ~(attn_mask.unsqueeze(0).to(torch.bool)), torch.finfo(attn_weight.dtype).min
                                                )

                                    with torch.profiler.record_function("store_seq_weights"):
                                        attn_weights.index_copy_(0, seq_index, attn_weight.unsqueeze(0))

                        with torch.profiler.record_function("seq_index_inc"):
                            seq_index.add_(1)
            else:
                attn_weights = attn_weights_2

            with torch.profiler.record_function("store_block_weight"):
                attn_weights_blocks.index_copy_(0, block_index, attn_weights.unsqueeze(0))

        with torch.profiler.record_function("block_index_inc"):
            block_index.add_(1)
        if device == "hpu":
            htorch.core.mark_step()

    # Cleanup out-of-bound weights and values
    attn_weights_blocks_filler = torch.finfo(query.dtype).min
    value_blocks_filler = 0.0
    _paged_attention_masked_fill(
        attn_weights_blocks,
        value_blocks,
        context_lens,
        attn_weights_blocks_filler,
        value_blocks_filler,
        block_size,
        max_num_blocks_per_seq,
        num_seqs,
        sanitize_values,
        device,
    )

    exp_sum = attn_weights_blocks.exp().sum(dim=-1, keepdim=True).sum(0)

    output = torch.zeros_like(query)
    for i in range(max_num_blocks_per_seq):
        with torch.profiler.record_function(f"output_block_loop_{i}"):
            with torch.profiler.record_function("compute_block_softmax"):
                attention_probs = torch.exp(attn_weights_blocks[i]) / exp_sum
            value = value_blocks[i]
            if num_queries_per_kv > 1:
                with torch.profiler.record_function("repeat_interleave_block_value"):
                    # Handle MQA and GQA
                    value_4d_view = value.reshape(num_seqs, num_kv_heads, block_size, head_size)
                    value = torch.repeat_interleave(value_4d_view, num_queries_per_kv, dim=1)
            with torch.profiler.record_function("reshape_for_gemm"):
                attention_probs = attention_probs.to(value.dtype).reshape(num_seqs * num_query_heads, 1, block_size)
                value = value.reshape(num_seqs * num_query_heads, block_size, head_size)
            with torch.profiler.record_function("gemm_out"):
                out = torch.matmul(attention_probs, value).reshape(num_seqs, num_query_heads, head_size)
            with torch.profiler.record_function("gemm_accumulate"):
                output.add_(out)
    if device == "hpu":
        htorch.core.mark_step()
    return output.to(dtype=query_in.dtype)

def rms_norm(out, hidden_states, weight, eps):
    htorch.core.mark_step()
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    out.copy_(weight * hidden_states.to(input_dtype))
    htorch.core.mark_step()

def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotate_fn = rotate_neox if is_neox_style else rotate_gptj
    q_embed = (q * cos) + (rotate_fn(q) * sin)
    k_embed = (k * cos) + (rotate_fn(k) * sin)
    return q_embed, k_embed


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style):
    # FIXME: the below code is unused legacy code not meant to be used. Use FusedRoPE
    #  on HPU and delete this once coverage is verified
    raise NotImplementedError

def awq_gemm(*args):
    raise NotImplementedError
