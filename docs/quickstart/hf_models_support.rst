Hugging Face Model Support
==========================

Overview
--------

Cosmos-RL provides comprehensive support for Hugging Face models through its generic ``HFModel`` wrapper. This design enables seamless integration of any LLMs or VLMs compatible with HF Transformers into the training and inference pipeline.

Architecture Design
-------------------

The ``HFModel`` class, implemented in ``cosmos_rl/policy/model/hf_models/__init__.py``, serves as a universal adapter for Hugging Face models. Key design principles include:

**Universal Compatibility**
    The wrapper supports any LLM or VLM architecture available through Hugging Face's model ecosystem.

**Automatic Fallback Mechanism**
    When a model lacks custom implementation in the ``cosmos_rl/policy/model`` directory, the system automatically defaults to the HF Model wrapper, ensuring broad compatibility without requiring custom code.

**Seamless Integration**
    - Model configuration loading via ``AutoConfig.from_pretrained``
    - Weight loading through ``model_class.from_pretrained`` (``model_class`` is defined in "architectures" of config.json)

**Distributed Training Ready**
    The wrapper is optimized for distributed training strategies, particularly FSDP (Fully Sharded Data Parallel), with built-in support for:
    - Weight synchronization
    - Rotary embedding handling
    - Parallelism compatibility checks

**Extensibility**
    Users can rapidly experiment with new or custom Hugging Face models without waiting for explicit cosmos-rl support, making the framework future-proof and adaptable to the evolving AI model landscape.

Distributed Training Support
----------------------------

Currently, the Policy module exclusively supports **Fully Sharded Data Parallel (FSDP)** for distributed training and inference operations.

Training Examples
-----------------

SFT
~~~

Note:

1. For LLMs like Mistral that require special handling to avoid role alternation errors, use the provided ``dummy_sft.py`` script.
2. For VLMs, use the provided ``hf_vlm_sft.py`` script.

.. code-block:: bash

    # LLM SFT
    cosmos-rl --config configs/mistral/mistral-7b-fsdp4-sft.toml ./tools/dataset/dummy_sft.py

    # VLM SFT
    cosmos-rl --config configs/gemma/gemma3-12b-vlm-fsdp4-sft.toml ./tools/dataset/hf_vlm_sft.py

GRPO
~~~~

.. code-block:: bash

    # LLM GRPO
    cosmos-rl --config configs/phi/phi4-14b-p-fsdp4-r-tp2-grpo.toml ./tools/dataset/gsm8k_grpo.py

Supported Models
----------------

The following models have been thoroughly tested and validated:

LLMs
~~~~

- **Mistral**
- **Gemma**
- **Phi**
- **GPT-OSS**

VLMs
~~~~

- **Gemma VLM**
- **Llama VLM**
- **LLava**

- Currently under active development

.. note::
   While models above have been extensively tested, other Hugging Face models should also be compatible. If you encounter issues with untested models, please report them by opening an issue in our repository.

.. tip::
   The generic HF Model wrapper enables support for most Hugging Face models without additional configuration. For optimal performance, consider using models that have been specifically tested and optimized.

