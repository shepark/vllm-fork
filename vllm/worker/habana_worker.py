###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

import gc
import os
from typing import List, Optional, Set, Tuple

import habana_frameworks.torch as htorch  # noqa:F401
import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, MultiModalConfig, ParallelConfig,
                         PromptAdapterConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import ExecuteModelRequest
from vllm.utils import HabanaMemoryProfiler, format_bytes
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.habana_model_runner import HabanaModelRunner
from vllm.worker.worker_base import LocalOrDistributedWorkerBase, WorkerInput

logger = init_logger(__name__)


class HabanaWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes (a partition of) the model on a HPU.

    Each worker is associated with a single HPU. The worker is responsible for
    maintaining the KV cache and executing the model on the HPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        lora_config: Optional[LoRAConfig] = None,
        multimodal_config: Optional[MultiModalConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prompt_adapter_config: Optional[PromptAdapterConfig] = None,
        is_driver_worker: bool = False,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.lora_config = lora_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker
        if self.is_driver_worker:
            assert self.rank == 0, "The driver worker must have rank 0."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()
        self.multimodal_config = multimodal_config

        self.model_runner: HabanaModelRunner = HabanaModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config=cache_config,
            load_config=load_config,
            lora_config=self.lora_config,
            kv_cache_dtype=self.cache_config.cache_dtype,
            multimodal_config=self.multimodal_config,
            is_driver_worker=is_driver_worker)
        # Uninitialized cache engine. Will be initialized by
        # initialize_cache.
        self.cache_engine: List[CacheEngine]
        # Initialize gpu_cache as embedding models don't initialize kv_caches
        self.hpu_cache: Optional[List[List[torch.tensor]]] = None

    def _set_env_vars(self):
        local_rank = self.local_rank
        if self.parallel_config.world_size == 1:
            local_rank = -1
        import os
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["ID"] = str(local_rank)
        os.environ["WORLD_SIZE"] = str(self.parallel_config.world_size)
        os.environ["RANK"] = str(self.rank)

    def init_device(self) -> None:
        if self.device_config.device.type == "hpu":
            self.device = torch.device("hpu")
            torch.hpu.set_device(self.device)
        else:
            raise RuntimeError(
                f"Not support device type: {self.device_config.device}")
        # Initialize the distributed environment.
        if self.model_config.quantization == 'inc':
            self._set_env_vars()
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank)
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Profiles the peak memory usage of the model to determine how many
        KV blocks may be allocated without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculate the maximum possible number of GPU and CPU blocks
        that can be allocated with the remaining free memory.

        .. tip::
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with HabanaMemoryProfiler() as m:
            self.model_runner.profile_run()
            torch.hpu.synchronize()
        msg = ("Model profiling run "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        # At this point we should've allocated the maximum workspace for all
        # recipes we will use the extra memory for graphs/blocks
        free_hpu_memory = torch.hpu.mem_get_info()[0]

        cache_block_size = self.get_cache_block_size_bytes()
        graph_reserved_mem = (float(
            os.environ.get('VLLM_GRAPH_RESERVED_MEM', '0.4'))
                              if not self.model_config.enforce_eager else 0)
        graph_headroom = 1 - graph_reserved_mem
        available_hpu_memory = free_hpu_memory * \
            self.cache_config.gpu_memory_utilization
        hpu_memory_margin = free_hpu_memory * (
            1 - self.cache_config.gpu_memory_utilization)
        self.model_runner.mem_margin = hpu_memory_margin
        cache_size_bytes = available_hpu_memory * graph_headroom
        graph_headroom_bytes = available_hpu_memory * (1 - graph_headroom)
        msg = (
            f"Free device memory: {format_bytes(free_hpu_memory)}, "
            f"{format_bytes(available_hpu_memory)} usable "
            f"(gpu_memory_utilization={self.cache_config.gpu_memory_utilization}),"
            f" {format_bytes(graph_headroom_bytes)} reserved for HPUGraphs "
            f"(VLLM_GRAPH_RESERVED_MEM={graph_reserved_mem}), "
            f"{format_bytes(cache_size_bytes)} reserved for KV cache")
        logger.info(msg)
        num_hpu_blocks = int(cache_size_bytes // cache_block_size)
        num_cpu_blocks = int(self.cache_config.swap_space_bytes //
                             cache_block_size)
        num_hpu_blocks = max(num_hpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)

        if self.model_runner.lora_manager:
            self.model_runner.remove_all_loras()

        gc.collect()
        return num_hpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Allocate GPU and CPU KV cache with the specified number of blocks.

        This also warms up the model, which may record CUDA graphs.
        """
        raise_if_cache_size_invalid(num_gpu_blocks,
                                    self.cache_config.block_size,
                                    self.model_config.max_model_len)

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        with HabanaMemoryProfiler() as m:
            self._init_cache_engine()
            torch.hpu.synchronize()
        msg = ("Initializing cache engine "
               f"took {m.get_summary_string()}")
        logger.info(msg)
        self._warm_up_model()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        self.cache_engine = [
            CacheEngine(self.cache_config, self.model_config,
                        self.parallel_config, self.device_config)
            for _ in range(self.parallel_config.pipeline_parallel_size)
        ]
        self.hpu_cache = [
            self.cache_engine[ve].gpu_cache
            for ve in range(self.parallel_config.pipeline_parallel_size)
        ]

    def _warm_up_model(self) -> None:
        # NOTE(kzawora): We should use virtual engine index here
        # for pipeline parallelism. Using 0 for now.
        assert self.hpu_cache is not None
        self.model_runner.warmup_model(self.hpu_cache[0])
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)

    def finish_measurements(self):
        self.model_runner.finish_measurements()

    @property
    def do_metadata_broadcast(self) -> bool:
        return self.parallel_config.tensor_parallel_size > 1

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.hpu_cache

    @torch.inference_mode()
    def prepare_worker_input(
            self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        virtual_engine = execute_model_req.virtual_engine
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
        # they contain parameters to launch cudamemcpyasync.
        blocks_to_swap_in = torch.tensor(execute_model_req.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64).view(-1, 2)
        blocks_to_swap_out = torch.tensor(execute_model_req.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64).view(-1, 2)
        # `blocks_to_copy` is a gpu tensor. The src and tgt of
        # blocks to copy are in the same device, and `blocks_to_copy`
        # can be used directly within cuda kernels.
        blocks_to_copy = torch.tensor(execute_model_req.blocks_to_copy,
                                      device=self.device,
                                      dtype=torch.int64).view(-1, 2)

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
        )

    @torch.inference_mode()
    def execute_worker(self, worker_input: WorkerInput) -> None:
        virtual_engine = worker_input.virtual_engine
        # Issue cache operations.
        if (worker_input.blocks_to_swap_in is not None
                and worker_input.blocks_to_swap_in.numel() > 0):
            self.cache_engine[virtual_engine].swap_in(
                worker_input.blocks_to_swap_in)
        if (worker_input.blocks_to_swap_out is not None
                and worker_input.blocks_to_swap_out.numel() > 0):
            self.cache_engine[virtual_engine].swap_out(
                worker_input.blocks_to_swap_out)
        if (worker_input.blocks_to_copy is not None
                and worker_input.blocks_to_copy.numel() > 0):
            self.cache_engine[virtual_engine].copy(worker_input.blocks_to_copy)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def list_loras(self) -> Set[int]:
        return self.model_runner.list_loras()

    def add_prompt_adapter(
            self, prompt_adapter_request: PromptAdapterRequest) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def remove_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def pin_prompt_adapter(self, prompt_adapter_id: int) -> bool:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def list_prompt_adapters(self) -> Set[int]:
        raise NotImplementedError(
            "Prompt Adapter is not implemented for HPU backend.")

    def shutdown_inc(self):
        self.model_runner.shutdown_inc()

    def __del__(self):
        self.shutdown_inc()

    @property
    def max_model_len(self) -> int:
        return self.model_config.max_model_len

    @property
    def vocab_size(self) -> int:
        return self.model_runner.vocab_size

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of the KV cache block size in bytes.
        """
        return CacheEngine.get_cache_block_size(self.cache_config,
                                                self.model_config,
                                                self.parallel_config)


def init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    init_distributed_environment(parallel_config.world_size,
                                 rank,
                                 distributed_init_method,
                                 local_rank,
                                 backend='hccl')

    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)

    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="hccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup & checking conformance.
    dummy_tensor_hpu = torch.ones(1).to('hpu')
    torch.distributed.all_reduce(dummy_tensor_hpu)
    assert dummy_tensor_hpu.item() == parallel_config.world_size
    ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
                                      parallel_config.pipeline_parallel_size)


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
