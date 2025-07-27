\[Experimental\] TensorRT-LLM
=============================

Cosmos-RL supports `TensorRT-LLM <https://github.com/NVIDIA/TensorRT-LLM>`_ as the backend of rollout generation.


Enable TensorRT-LLM
-------------------

To enable TensorRT-LLM, you need to set the fields of ``rollout`` section in the config file:

.. code-block:: toml

    [rollout]
    backend = "trtllm"


For now, tested models are:

- Qwen3-moe






