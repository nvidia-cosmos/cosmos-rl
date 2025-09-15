Multi-Turn Rollout Example (For GSM8K)
==================

This example shows how to perform multi-turn RL training on cosmos-rl.

Usage
::::::

Step 1: Configure the config file
::::::::::::::::::::::::::::::::::

cosmos-rl provides a multi-turn rollout config file, which can be referred to ``configs/qwen2-5/qwen2-5-multiturn-3b-p-fsdp1-tp1-r-tp1-pp1-grpo.toml``

The ``[rollout.multi_turn_config]`` section configures the parameters for multi-turn rollout, as follows:

- ``enable``: Whether to enable multi-turn rollout
- ``enable_tools``: Whether to enable tools, if enabled, the related tool call prompt will be added to the generated prompt. The tool addition method can be referred to another document
- ``enable_thinking``: Whether to enable thinking
- ``custom_chat_template_path``: The path to the custom chat template, which is a jinja2 template, and the writing method can be referred to huggingface chat_template writing: `<https://huggingface.co/docs/transformers/en/chat_templating_writing>`_
- ``max_assistant_turns``: Maximum assistant turns
- ``add_generation_prompt``: Whether to add generation prompt, default to true
- ``continue_final_message``: Whether to continue the final message

Step 2: Run the GRPO training
::::::::::::::::::::::::::::::

.. code-block:: bash

    cosmos-rl --config configs/qwen2-5/qwen2-5-multiturn-3b-p-fsdp1-tp1-r-tp1-pp1-grpo.toml tools/dataset/gsm8k_grpo.py

Notes
::::::

- To enable model to call tools and reference tool response to generate final answer, it is recommended to set ``max_assistant_turns`` to at least 2, i.e., the first assistant calls the tool, and the second assistant references the tool response to generate the final answer
- The effect of tool_call is closely related to chat_template, if the tool call is not ideal, please print the conversation, so as to determine whether the chat_template is appropriate
- To execute the GRPO algorithm, it is necessary to set ``rollout.n_generation`` to be greater than 1, so as to calculate the correct reward