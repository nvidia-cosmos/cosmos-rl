\[Experimental\] TensorRT-LLM
=============================

Cosmos-RL supports `TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ as the backend of rollout generation.

.. note::

    To use TensorRT-LLM as the rollout backend, you have to build docker image with the file ``docker/Dockerfile.trtllm`` in the root directory of ``cosmos-rl`` project.



Enable TensorRT-LLM
-------------------

To enable TensorRT-LLM, you need to set the fields of ``rollout`` section in the config file:

.. code-block:: toml

    [rollout]
    backend = "trtllm"


For now, tested models are:

- Qwen3-moe
- Qwen2-5
- Qwen2-5 VL

.. note::
    We just support rollout replica within a single node now.





