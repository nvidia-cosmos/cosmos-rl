
Multi-node example
==================

Manual deployment
::::::::::::::::::

Similar to single-node example, a controller process should be launched first and any extra *policy* and *rollout* replica can join at any time.

In this example, 4 policy replica with each of 4 GPUs and two rollout replica with each of 2 GPUs are launched.

.. code-block:: bash

    ### On node A
    # 1. Launch the controller process
    ./cosmos_rl/launcher/launch_controller.sh --port 8000 --config configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml

    export COSMOS_CONTROLLER_HOST=localhost:8000

    # 2. Launch one policy replica of 4 GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/launch_replica.sh --type policy --ngpus 4

    # 3. Launch two rollout replicas of 2 GPUs
    CUDA_VISIBLE_DEVICES=4,5 ./tools/launch_replica.sh --type rollout --ngpus 2
    CUDA_VISIBLE_DEVICES=6,7 ./tools/launch_replica.sh --type rollout --ngpus 2

    ### On node B
    # Launch two more policy replicas of 4 GPUs
    export COSMOS_CONTROLLER_HOST={node-A-host-or-ip}:8000
    CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/launch_replica.sh --type policy --ngpus 4
    CUDA_VISIBLE_DEVICES=4,5,6,7 ./tools/launch_replica.sh --type policy --ngpus 4

.. note::

    | In case one single replica needs more GPUs than 8 (across multiple nodes),

    .. code-block:: bash

        --rdzv-endpoint={MASTER_ADDR}:{MASTER_PORT}

    | should be specified for `launch_replica.sh` to correctly form a valid replica.

DGXC-Lepton job example
:::::::::::::::::::::::

DGXC-Lepton Job takes care of the number of GPU required by each policy and rollout replica. It will calculate and allocate the number of total nodes necessary to launch the job.

>>> cosmos-rl \
        --config ./configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml \
        --lepton-mode \
        --lepton-job-name cosmos-multi-node-test \
        --lepton-container-image {your-docker-image} \
        --lepton-resource-shape gpu.8xh100-sxm \
        --lepton-node-group {your-node-group} \
        --lepton-secret HUGGING_FACE_HUB_TOKEN={your-hugging-face-token} \
        --policy 8 \
        --rollout 4 \
        /workspace/cosmos_rl/tools/dataset/math_dapo.py

Total `8*4 + 4*2 = 40` GPUs are used. (5 `8xH100` nodes will be allocated for this job)

For more details about DGXC-Lepton, please refer to :doc:`../multinodes/dgxc_lepton`

