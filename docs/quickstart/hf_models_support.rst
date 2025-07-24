Hugging Face Model Type Support
===============================

Design of HF Model Type
------------------------------------

The HF Model Type (``HFLLMModel``) in cosmos-rl is designed to provide a generic and flexible interface for integrating HF models into the training and inference pipeline. The core implementation is located in ``cosmos_rl/policy/model/hf_llm/__init__.py``, where the ``HFLLMModel`` class acts as a universal wrapper for any model supported by Hugging Face's ``AutoModelForCausalLM``.
Key design points:

- **Generic Wrapper:** ``HFLLMModel`` can load any model architecture supported by Hugging Face, as long as it is compatible with ``AutoModelForCausalLM``.
- **Automatic Fallback:** If a model is not custom-defined in the ``cosmos_rl/policy/model`` directory, the system will automatically use the HF Models Type, ensuring broad compatibility and reducing the need for custom code.
- **Config and Weight Loading:** The model configuration is loaded using ``AutoConfig.from_pretrained``, and model weights are loaded via ``AutoModelForCausalLM.from_pretrained``. This allows seamless integration with models from the Hugging Face Hub or local checkpoints.
- **Distributed Training Support:** The wrapper is designed to work with distributed training strategies such as FSDP, and includes logic for weight synchronization, rotary embedding handling, and compatibility checks for parallelism.
- **Extensibility:** By using the HF Models Type, users can quickly experiment with new or custom Hugging Face models without waiting for explicit support in cosmos-rl.

This design ensures that cosmos-rl remains extensible and up-to-date with the rapidly evolving Hugging Face ecosystem, while still allowing for custom optimizations and model-specific logic when needed.


Policy Parallelism Support
--------------------------

Currently, the Policy module only supports Fully Sharded Data Parallel (FSDP) for distributed training and inference.

Special Case: Mistral SFT with Custom Dataset Loader
----------------------------------------------------

For certain models like Mistral, SFT requires a custom dataset loader to avoid role alternation errors. Use the provided `raw_text_sft.py` script for this purpose.

Example command:

.. code-block:: bash

    cosmos-rl --config configs/mistral/mistral-7b-fsdp4-sft.toml cosmos_rl.tools.dataset.dummy_sft.py

Another example for different configuration:

.. code-block:: bash

    cosmos-rl --config configs/mistral/mistral-7b-p-fsdp1-r-tp1-grpo.toml

Supported HF Models
-------------------

- Mistral
- Gemma
- Phi

.. note::
   Mistral, Gemma, Phi have been tested. Other HF models may also be supported. If you encounter any issues when training other models, please feel free to open an issue and let us know.

