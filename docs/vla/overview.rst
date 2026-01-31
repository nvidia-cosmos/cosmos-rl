Overview
=================

Cosmos-RL provides native support for Vision-Language-Action (VLA) model reinforcement learning, enabling embodied AI agents to learn robotic manipulation tasks through interaction with simulators.

VLA Models
----------

Cosmos-RL supports two major series of VLA models:

OpenVLA Series
^^^^^^^^^^^^^^

OpenVLA is a vision-language-action model built on the Prismatic framework. Cosmos-RL supports two variants:

- **OpenVLA**: The original OpenVLA model architecture
- **OpenVLA-OFT**: OpenVLA with Online Fine-Tuning support, optimized for RL training

**Key Features:**

- Based on SigLIP/Dinov2 + LLaMA architecture
- Action prediction through language model token generation
- Norm statistics for action normalization
- Compatible with HuggingFace model hub

**Model Configuration:**

.. code-block:: toml

   [policy]
   model_name_or_path = "Haozhan72/Openvla-oft-SFT-libero10-trajall"

   [vla]
   vla_type = "openvla-oft"
   training_chunk_size = 16

PI Series (PI0.5)
^^^^^^^^^^^^^^^^^

PI0.5 is a diffusion-based VLA model that uses flow-based action prediction for robotic manipulation.

**Key Features:**

- Based on PaliGemma model with Diffusion
- Flow-based action generation with configurable denoising steps
- Expert network for action prediction
- Supports multiple noise methods: flow_sde, flow_cps, flow_noise

**Model Configuration:**

.. code-block:: toml

   [policy]
   model_name_or_path = "sunshk/pi05_libero_pytorch"
   
   [vla]
   vla_type = "pi05"
   training_chunk_size = 16

   [custom.pi05]
   num_steps = 5  # denoise steps
   action_chunk = 10
   action_env_dim = 7
   noise_method = "flow_sde"
   train_expert_only = true

Simulators
----------

Cosmos-RL integrates with multiple robotics simulators for VLA training and evaluation:

LIBERO (MuJoCo-based)
^^^^^^^^^^^^^^^^^^^^^

LIBERO (Lifelong roBotic lEarning benchmaRk with lOng-horizon tasks) is a MuJoCo-based simulation environment for long-horizon manipulation tasks.

**Supported Task Suites:**

- libero_spatial: 10 spatial reasoning tasks
- libero_object: 10 object interaction tasks
- libero_goal: 10 goal-oriented tasks
- libero_10: 10 long-horizon manipulation tasks
- libero_90: 90 diverse manipulation tasks
- libero_all: 130 tasks from all suites

**Features:**

- CPU/GPU rendering support
- Multi-environment parallel rollout
- Task initialization from dataset states
- Each environment can run different tasks simultaneously
- Action space: 7-DoF (position, rotation, gripper)

**Configuration:**

.. code-block:: toml

   [validation]
   dataset.name = "libero"
   dataset.subset = "libero_10"
   dataset.split = "val"

   [vla]
   num_envs = 8  # parallel environments per rank

**Data Source:** `LIBERO GitHub <https://github.com/Lifelong-Robot-Learning/LIBERO>`_

BEHAVIOR-1K (IsaacSim-based)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BEHAVIOR-1K is a large-scale benchmark built on OmniGibson/IsaacSim for household manipulation tasks.

**Features:**

- GPU-accelerated physics simulation
- 1000+ diverse household tasks
- Photorealistic rendering
- Task descriptions from BEHAVIOR benchmark
- Action space in B1K Neurips'25 challenge: 23-DoF 

**Configuration:**

.. code-block:: toml

   [validation]
   dataset.name = "b1k"
   dataset.subset = "b1k"
   
   [vla]
   num_envs = 4
   height = 256
   width = 256

**Requirements:**

- NVIDIA GPU with RT Cores (L20, L40, or RTX series)
- OmniGibson installation
- BEHAVIOR-1K dataset

**Data Source:** Available through OmniGibson

Reinforcement Learning Algorithms
----------------------------------

Cosmos-RL supports multiple RL algorithms optimized for VLA training:

GRPO (Group Relative Policy Optimization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GRPO is a policy gradient algorithm designed for efficient on-policy learning without requiring a critic network.

**Key Features:**

- No value function / critic network needed
- Group-based advantage estimation using relative rewards
- Efficient for VLA training with sparse rewards
- Supports reward filtering to remove outliers

**Algorithm Configuration:**

.. code-block:: toml

   [train.train_policy]
   type = "grpo"
   trainer_type = "grpo_vla"  # or "grpo_pi05" for PI models
   variant = "dapo"
   temperature = 1.6
   epsilon_low = 0.2
   epsilon_high = 0.28
   lower_bound_ratio = 10.0
   kl_beta = 0.0

**How It Works:**

1. Generate multiple rollouts per task
2. Compute relative advantages within each group (task)
3. Use policy gradient with clipped objective
4. Update policy based on advantage signals
5. Change to dapo variant if need asymmetric clipping or dynamic sampling


Quick Start
-----------

**1. Configure the training recipe** by editing toml files under ``configs/openvla-oft/`` or ``configs/pi05/``.

**2. Prepare the simulator environment:**

For LIBERO:

.. code-block:: bash

   cd /your/workspace/cosmos-rl
   uv sync --extra vla
   # or
   pip install -e .[vla]

   ROBOT_PLATFORM=LIBERO \
   uv run cosmos-rl --config configs/openvla-oft/openvla-oft-7b-fsdp2-8p8r-colocate.toml \
          --log-dir logs \
          cosmos_rl/tools/dataset/libero_grpo.py

For BEHAVIOR-1K:

.. code-block:: bash

   cd /your/workspace/cosmos-rl
   apt install python3-dev # if necessary
   uv sync
   source .venv/bin/activate

   cd /your/workspace
   git clone -b v3.7.2 https://github.com/StanfordVL/BEHAVIOR-1K.git
   cd BEHAVIOR-1K
   uv add pip cffi==1.17.1 # if necessary
   apt install -y libsm6 libxt6 libglu1-mesa
   UV_LINK_MODE=hardlink ./setup.sh --omnigibson --bddl --joylo --confirm-no-conda --accept-nvidia-eula

**3. Prepare the pretrained model:**

For PI0.5:

- We prepared PI0.5 models ranked 2nd in the Neurips'25 BEHAVIOR-1K challenge, converted to PyTorch format
- Download from `fwd4xl/pi05-b1k-pt12-cs32-v1 <https://huggingface.co/fwd4xl/pi05-b1k-pt12-cs32-v1>`_

**4. Launch training:**

For OpenVLA-OFT:

.. code-block:: bash

   ROBOT_PLATFORM=LIBERO \
   uv run cosmos-rl --config configs/openvla-oft/openvla-oft-7b-fsdp2-8p8r-colocate.toml \
         cosmos_rl/tools/dataset/libero_grpo.py

For PI0.5:

.. code-block:: bash

   uv run cosmos-rl --config configs/pi05/pi05-b1k-grpo-colocate.toml \
         cosmos_rl/tools/dataset/b1k_grpo.py


References
----------

- `OpenVLA Paper <https://arxiv.org/abs/2406.09246>`_
- `PI05 <http://arxiv.org/abs/2504.16054>`_
- `LIBERO Benchmark <https://github.com/Lifelong-Robot-Learning/LIBERO>`_
- `BEHAVIOR Benchmark <https://behavior.stanford.edu/challenge/index.html>`_
- `SimpleVLA-RL <https://github.com/bytedance/SimpleVLA-RL>`_
- `GRPO Algorithm <https://arxiv.org/abs/2402.03300>`_
