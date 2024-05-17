.. _installation:

Installation with Intel® Gaudi® 2
=================================

vLLM-fork supports Intel® Gaudi® 2 accelerators with SynapseAI 1.16.0.

Requirements
------------

* OS: Ubuntu 22.04 LTS
* Python: 3.10
* Intel® Gaudi® 2 accelerator with SynapseAI 1.16.0

Build from source
-----------------

You can build and install vLLM-fork from source:

.. code-block:: console

    $ git clone https://github.com/HabanaAI/vllm-fork.git
    $ cd vllm-fork
    # git checkout 0.4.2-SynapseAI-1.16.0
    $ pip install -e .  # This may take 5-10 minutes.

.. tip::
    It is highly recommended to use the latest Docker image from Intel® Gaudi® Vault. Please follow the guide from latest `SynapseAI documentation <https://docs.habana.ai/en/latest/shared/Pull_Prebuilt_Containers.html>`_.
    For SynapseAI 1.16.0, you can use the following commands to run a Docker image:

    .. code-block:: console

        $ docker pull vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
        $ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest

    `Here <https://docs.habana.ai/en/latest/Installation_Guide/SW_Verification.html#platform-upgrade>`_ is a sanity check to verify that the SynapseAI 1.16.0 is correctly installed:

    .. code-block:: console

        $ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
        $ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core and habanalabs-thunk are installed
        $ pip list | habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml, habana-media-loader and habana_quantization_toolkit are installed

Supported features
------------------
* `Offline batched inference <quickstart.html#offline-batched-inference>`_
* Online inference via `OpenAI-Compatible Server <quickstart.html#openai-compatible-server>`_\
* HPU autodetection - no need to manually select device within vLLM
* Paged KV cache with algorithms enabled for Intel® Gaudi® 2 accelerators
* Custom Gaudi implementations of Paged Attention, KV cache ops, prefill attention, Root Mean Square Layer Normalization, Rotary Positional Encoding 
* Tensor parallelism support for multi-card inference
* Inference with `HPU Graphs <https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html>`_ for accelerating low-batch latency and throughput  

.. note::
   In this release (1.16.0), we are only targeting functionality and accuracy. Performance will be improved in next releases.

Supported configurations
------------------------
* `meta-llama/Llama-2-7b <https://huggingface.co/meta-llama/Llama-2-7b>`_ on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
* `meta-llama/Llama-2-7b-chat-hf <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`_ on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling
* `meta-llama/Llama-2-70b <https://huggingface.co/meta-llama/Llama-2-70b>`_ with tensor parallelism on 2x and 8x HPU on BF16 datatype with greedy decode
* `meta-llama/Llama-2-70b-chat-hf <https://huggingface.co/meta-llama/Llama-2-70b-chat-hf>`_ with tensor parallelism on 2x and 8x HPU, BF16 datatype with random or greedy sampling

.. note::
   The configurations were checked functionally and for accuracy. Performance will be improved in next releases.
.. note::
   Other configurations are not validated and may or may not work. 

Unsupported features
--------------------
* Beam search
* LoRA adapters
* Attention with Linear Biases (ALiBi)
* Quantization (AWQ, FP8 E5M2, FP8 E4M3)
* Prefill chunking (mixed-batch inferencing)

Performance tips
----------------
* We recommend running inference on Gaudi 2 with `block_size` of 128 for BF16 data type. Using default values (16, 32) might lead to sub-optimal performance due to Matrix Multiplication Engine under-utilization (see `Gaudi Architecture <https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html>`_).
* For max throughput on Llama 7B, we recommend running with batch size of 128 or 256 and max context length of 2048 with HPUGraphs enabled. If you encounter out-of-memory issues, see troubleshooting section


Troubleshooting: Tweaking HPU Graphs
------------------------------------
If you:

* experience device out-of-memory issues
* want to attempt inference at higher batch sizes

You might want to tweak `gpu_memory_utilization` knob. It will decrease the allocation of KV cache, leaving some headroom for capturing graphs with larger batch size. By default, it is set 
to 0.9 (attempts to allocate ~90% of HBM left for KV cache after short profiling run). Note that decreasing it will reduce the number of KV cache blocks you have available, and will therefore reduce the effective maximum number of tokens you can handle at a given time.

If that is not enough, you can disable HPUGraph usage completely. With HPU Graphs disabled, you're trading latency and throughput at lower batches for potentially higher throughput on higher batches. You can do that by providing `--enforce-eager` flag to server (for online inference), or by passing `enforce_eager=True` argument to LLM constructor (for offline inference)

