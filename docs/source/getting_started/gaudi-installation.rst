.. _installation:

Installation
============

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

Limitations
-----------------

WIP