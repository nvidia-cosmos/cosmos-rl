Elastic Scaling and Fault Tolerance
=============================================

Cosmos-RL supports **elastic scaling** and **fault tolerance** for RL jobs. This document provides a detailed explanation of how these features work.

Elastic Scaling
---------------

Cosmos-RL allows running multiple **replicas** across **multiple nodes**, for both **Policy** and **Rollout** components. Each replica shares the same parallelism configuration. When a replica starts, only the controller is aware of it, and it will publish commands to existing replicas to perform operations such as building global NCCL meshes, weight updates, etc.

Cosmos-RL supports **elastic launching** of replicas, enabling users to dynamically add or remove replicas during execution. The system handles necessary coordination automatically.

### Replica Initialization Modes

There are two initialization modes based on the configuration field `n_init_replicas`:

1. **`n_init_replicas = N > 1`**  
   Cosmos-RL will wait until `N` replicas have joined before proceeding, and will treat later replicas as dynamic.

2. **`n_init_replicas = 1` (default)**  
   Cosmos-RL immediately treats the first launched replica as active and dynamically integrates subsequent replicas.

Policy and Rollout components each maintain their own `n_init_replicas` setting, defaulting to `1`.

### Policy Replica Initialization Flow

1. Users can launch policy replicas one by one.
2. Training begins only **after** `N` replicas have launched.
3. The **first** replica performs weight initialization (e.g., from checkpoint or Hugging Face model).
4. Once `N` replicas are active:
   - The controller sends a `BuildMesh` command.
   - A selected weight-initialized replica broadcasts weights to others.
5. The policy group is now ready to begin the RL workflow.

### Rollout Replica Initialization Flow

1. Rollout replicas are also launched one by one.
2. Only the **first** rollout replica starts generating rollouts initially.
3. It synchronizes weights from an initialized policy replica.
4. Once `N` rollout replicas are active:
   - The controller sends a `BuildMesh` command.
   - The first rollout replica broadcasts weights to others.
5. All rollout replicas now participate in rollout generation.

.. note::
   For `n_init_replicas = 1`, the flow is the same with `N = 1`.

### Dynamically Launched Replicas

Cosmos-RL supports adding new replicas during runtime. These replicas will be integrated into the RL workflow as follows:

- **Policy Side**
  - Controller sends a `BuildMesh` command.
  - A chosen initialized replica sends weights to the new one (via policy-policy unicast).

- **Rollout Side**
  - Controller sends a `BuildMesh` command.
  - A chosen initialized rollout replica broadcasts weights to the new one (via rollout-rollout broadcast).

After that, the new replica is fully integrated.

Demo: Elastic Scaling in Action
-------------------------------

### Step 1: Launch Controller

::

  ./tools/launch_controller.sh --port 8080 --config ./configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml

### Step 2: Launch Initial Replicas (Assuming 8-GPU machine)

::

  # Launch 1 policy replica
  CUDA_VISIBLE_DEVICES=0,1 COSMOS_CONTROLLER_HOST=localhost:8080 ./tools/launch_replica.sh --ngpus 2 --type policy

  # Launch 1 rollout replica
  CUDA_VISIBLE_DEVICES=2,3 COSMOS_CONTROLLER_HOST=localhost:8080 ./tools/launch_replica.sh --ngpus 2 --type rollout

Wait a moment to observe the initial RL workflow and check the status in the controller’s web UI.

### Step 3: Add More Replicas Dynamically

::

  # Add 1 more policy replica
  CUDA_VISIBLE_DEVICES=4,5 COSMOS_CONTROLLER_HOST=localhost:8080 ./tools/launch_replica.sh --ngpus 2 --type policy

  # Add 1 more rollout replica
  CUDA_VISIBLE_DEVICES=6,7 COSMOS_CONTROLLER_HOST=localhost:8080 ./tools/launch_replica.sh --ngpus 2 --type rollout

You will see the new replicas appear in the web UI. Now, all four replicas (2 policy + 2 rollout) are active and working in sync.

Fault Tolerance
---------------

Cosmos-RL maintains heartbeat communication between the controller and all replicas. 

If either:

- a replica fails to send heartbeats within `COSMOS_HEARTBEAT_TIMEOUT` (default is 5 minutes) seconds
- NCCL operation times out

the controller will consider it offline and remove it from the NCCL mesh. So it won't block the ongoing RL workflow.
