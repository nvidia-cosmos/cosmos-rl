Single node example
==============================

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
