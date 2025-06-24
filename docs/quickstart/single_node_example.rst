Single node example
==============================

One-click solution
::::::::::::::::::::::

This example demonstrates how to run `Qwen3-8B` on a single node with 8 GPUs.

>>> cosmos-rl \
    --config configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml \
    --policy 1 \
    --rollout 2

Explanation of the command:

- ``--config``: the path to the training config file.
- ``--policy``: the number of policy replicas.
- ``--rollout``: the number of rollout replicas.

As the toml file name suggests, this example uses `Qwen3-8B <https://huggingface.co/Qwen/Qwen3-8B>`_ model with:

- 4-way tensor parallelism for policy model
- 2-way tensor parallelism for rollout model

and total 8 GPUs are used since 1 policy and 2 rollout replicas are specified.

If everything goes well, you should see the training process like this:

.. code-block:: bash

    [rank0]:[cosmos] 2025-06-09 22:14:29,220 - cosmos - INFO - Step: 1/4670, Loss: 0.00000
    [rank1]:[cosmos] 2025-06-09 22:14:29,219 - cosmos - INFO - Step: 1/4670, Loss: 0.00000
    ...

.. note::

    You may encounter loss values of 0.0 because the GRPO advantage is zero. Since it is a toy math example, it is expected.


Manual launch
:::::::::::::

Under the hood, the previous `one-click` script do the followings:

1. Prepare the dataset:
    - Download the `GSM8K <https://huggingface.co/datasets/openai/gsm8k/>`_ dataset
    - Extract prompt and reference answer using column names specified in the config file
    
    .. code-block:: yaml

        # file content of `qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml` 
        ...
        [train.train_policy]
        ...
        dataset_name = "openai/gsm8k" # the dataset name
        dataset_subset = "main" # the subset name
        dataset_train_split = "train" # the train split name
        prompt_column_name = "question" # column for prompt
        response_column_name = "answer" # column for reference answer
        ...

2. Launch processes:
    1. A controller process for coordinating the training process
    2. A policy replica with 4 processes
    3. Two rollout replicas each with 2 processes


Alternatively, all these processes can be launched manually by running the following commands:

.. code-block:: bash

    # 1. Launch the controller process
    ./tools/launch_controller.sh --port 8000 --config configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml

    # Set env-var so that the following replicas know where the controller is located.
    export COSMOS_CONTROLLER_HOST=localhost:8000

    # 2. Launch the policy replica
    CUDA_VISIBLE_DEVICES=0,1,2,3 COSMOS_CONTROLLER_HOST=localhost:8000 ./tools/launch_replica.sh --type policy --ngpus 4

    # 3. Launch the rollout replicas
    CUDA_VISIBLE_DEVICES=4,5 ./tools/launch_replica.sh --type rollout --ngpus 2
    CUDA_VISIBLE_DEVICES=6,7 ./tools/launch_replica.sh --type rollout --ngpus 2
