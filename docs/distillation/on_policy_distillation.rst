On-Policy Distillation
======================

Overview
--------

On-policy distillation is a training approach where a student model learns from a teacher model's logits distribution through knowledge distillation during the rollout phase. This technique enables efficient transfer of knowledge from larger or better-performing teacher models to smaller or faster student models on targeted datasets.

In Cosmos RL, on-policy distillation is integrated into the training pipeline where:

1. The student model generates completions during rollout
2. The teacher model receives all completions the student model generates to get the logits and probability distributions
3. The student model receives the teacher's logit probability distributions and is trained to match the teacher's distribution using simple reserve KL or Jensen-Shannon Divergence (JSD) loss
4. This happens within the standard training loop alongside other optimization objectives

Quick Start
-----------

Example Commands
::::::::::::::::

DeepMath Dataset (Qwen3-8B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start an on-policy distillation job for the Qwen3-8B model on the DeepMath dataset:

.. code-block:: bash

    cosmos-rl --config configs/qwen3/qwen3-8b-distill-deepmath.toml tools/dataset/deepmath_distill.py

Breaking down the command:

- ``cosmos-rl``: Main CLI entry point
- ``--config``: Path to the TOML configuration file
- ``tools/dataset/deepmath_distill.py``: Custom dataset and reward function script that handles data loading and evaluation

Countdown Dataset (Qwen2.5-1.5B)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start an on-policy distillation job for the Qwen2.5-1.5B model on the Countdown dataset:

.. code-block:: bash

    cosmos-rl --config configs/qwen2-5/qwen2-5-1.5b-distill-countdown.toml tools/dataset/countdown_distill.py

This example uses a smaller model (1.5B) with a different task (Countdown numbers game) compared to the DeepMath example.

Configuration File Structure
-----------------------------

The configuration file (TOML format) contains multiple sections. Here's the example section in the toml config for distillation:

.. code-block:: toml

    [distillation]
    enable = true
    model_name_or_path = "Qwen/Qwen3-8B"
    compile = true
    mini_batch = 8
    master_dtype = "float32"
    param_dtype = "bfloat16"
    logprob_dtype = "float32"
    fsdp_reduce_dtype = "float32"
    fsdp_offload = false
    fsdp_reshard_after_forward = "default"
    batch_size_per_replica = 16
    top_k = 0
    jsd_beta = 1
    include_prompt = false
    trainer_token_ids_from_teacher = true
    rollout_top_k_recompute = false

    [distillation.parallelism]
    n_init_replicas = 1
    tp_size = 1
    cp_size = 1
    dp_shard_size = 2
    pp_size = 1
    dp_replicate_size = 1

Distillation Parameters
-----------------------

Core Configuration
::::::::::::::::::

**enable** (boolean)
    Enable/disable distillation during training. Set to ``true`` to activate the distillation loss.

**model_name_or_path** (string)
    Path or HuggingFace model identifier for the teacher model. This is the model from which logits will be extracted during rollout.
    
    Example: ``"Qwen/Qwen3-8B"``

**mini_batch** (integer)
    Batch size at each GPU for each forward execution when the teacher model generates logits and logprobs using the student-generated prompt and completion. Smaller values reduce peak memory usage during teacher inference.
    
    Default: ``1``

**batch_size_per_replica** (integer)
    Total number of samples processed per teacher model replica per distillation step. This may be first split to each GPU in the replica and then further split into multiple ``mini_batch`` forward passes for memory efficiency. For example, if ``batch_size_per_replica=32`` and ``mini_batch=8`` and there are 2 GPUs in the replica, the teacher will make 2 forward passes per GPU.
    
    Formula: ``number_of_teacher_forward_passes = batch_size_per_replica // (number_of_gpus_in_replica * mini_batch)``
    
    Default: ``8``

Parallelism Configuration
:::::::::::::::::::::::::

The ``[distillation.parallelism]`` section configures GPU parallelism for distillation:

**n_init_replicas** (integer)
    Number of parallel distillation workers. Keep at ``1`` unless running multi-node distillation.

**tp_size** (integer)
    Tensor parallelism degree. Splits model parameters across GPUs. Use for very large models.

**dp_shard_size** (integer)
    Data parallelism shard size. Number of GPUs for data parallelism.

Advanced Options
::::::::::::::::

Teacher Model Setting Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**compile**, **master_dtype**, **param_dtype**, **logprob_dtype**, **fsdp_reduce_dtype**, **fsdp_offload**, **fsdp_reshard_after_forward**
    These parameters work the same as in the ``[train]`` section for normal model settings. They control compilation, mixed precision forward, FSDP behavior, and memory management for the teacher model. See the ``[train]`` section documentation for detailed explanations.

Loss & Sampling Algorithm Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**jsd_beta** (float)
    Interpolation coefficient between ``0.0`` and ``1.0`` of the Generalized Jensen-Shannon Divergence loss. 
    
    - When beta is ``0.0``, the loss is the KL divergence
    - When beta is ``1.0``, the loss is the Inverse KL divergence
    - Values between 0 and 1 interpolate between these two extremes
    
    Default: ``0.5``

**top_k** (integer)
    Controls the distillation loss formulation:
    
    - When ``0``: Uses simple reverse KL for loss (as described in `On-Policy Distillation <https://thinkingmachines.ai/blog/on-policy-distillation>`_)
    - When ``> 0``: Uses the generalized Jensen-Shannon Divergence loss for knowledge distillation using ``F.kl_div``, restricting to top-k most likely tokens. See Eq. (1) of `this paper <https://huggingface.co/papers/2306.13649>`_ for the definition
    
    Default: ``0``

Model Training Framework Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**include_prompt** (boolean)
    Include the prompt tokens in the distillation loss computation. When ``false``, only completion tokens are considered.
    
    - Set to ``true`` if you want the student to also match teacher's prompt embeddings
    - Set to ``false`` (default) to focus on generation quality
    
    Default: ``false``

**trainer_token_ids_from_teacher** (boolean)
    Whether the trainer gets all top_k token ids directly from its redis-connected teacher model during distillation rather than from the rollout structure. This can simplify the rollout payload when being transferred in the framework.
    
    Note: When ``top_k <= 0``, this parameter is automatically set to ``false``.
    
    Default: ``true``

**rollout_top_k_recompute** (boolean)
    Whether to recompute all top-k logprobs with top-k token ids after the full sequence is generated during rollout for distillation. This can ensure the completion generation process doesn't keep large top-k values that would degrade generation efficiency.
    
    Default: ``false``

Launching on SLURM
------------------

You can launch an on-policy distillation task on SLURM using the Cosmos RL dispatch helper:

.. code-block:: bash

    python $PATH_TO_COSMOS_RL_ROOT/tools/slurm/dispatch_job.py \
        --ngpu-per-node 8 \
        --config-path $PATH_TO_COSMOS_RL_ROOT/configs/qwen3/qwen3-8b-distill-deepmath.toml \
        --output-root-path $PATH_TO_COSMOS_RL_ROOT/s_output \
        --cosmos-container $SQSH_PATH \
        --slurm-partition batch \
        --slurm-account sw_aidot \
        --repo-root-path $PATH_TO_COSMOS_RL_ROOT \
        ./cosmos_rl/tools/dataset/deepmath_distill.py

Notes
:::::

- Set ``$PATH_TO_COSMOS_RL_ROOT`` to your local Cosmos RL repository root.
- Set ``$SQSH_PATH`` to the container image path (``.sqsh``).
- You can swap the config and dataset entry script for other distillation tasks.

Launching on Lepton
-------------------

For cloud-based training on NVIDIA Lepton, you can clone an existing reference job and customize it for your needs.

Reference Job
:::::::::::::

Start with the Qwen8B DeepMath distillation reference job:

`Qwen8B DeepMath Distillation Job <https://dashboard.dgxc-lepton.nvidia.com/workspace/b5k2m9x7/compute/jobs/archived/qwen8b-distill-deepmath-checkbase-476b/replicas/list>`_

Steps to Clone and Customize
:::::::::::::::::::::::::::::

1. **Clone the Job**: 
   - Click the "Clone" or "Create from Template" button in the job details view
   - This will create a copy of the job configuration with all settings pre-populated

2. **Customize Configuration**:
   - Modify the job name and description for your experiment
   - Update the toml configuration file content and path (``config.toml``)
   - Update the launch entry script or module if needed (``cosmos_rl.tools.dataset.deepmath_distill``)
   - Adjust resource allocation (GPU count, memory) based on your model size and requirements

3. **Submit and Monitor**:
   - Click "Submit Job" to launch the training
   - Monitor training progress through the Lepton dashboard and wandb logs


See Also
--------

- :doc:`../quickstart/single_node_example` - Basic training setup
- :doc:`../parallelism/index` - Distributed training configuration
- :doc:`../rollout/index` - Rollout configuration details
