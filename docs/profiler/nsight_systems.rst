Nsight Systems
==============

Cosmos-RL can launch selected replicas under NVIDIA Nsight Systems and use
``cudaProfilerStart`` / ``cudaProfilerStop`` to capture only a small training
window. This is useful for the asynchronous, disaggregated RL flow because the
target process must be wrapped by ``nsys profile`` at launch time, while the
interesting policy step is usually selected later by the controller.

Configuration
-------------

Enable Nsight Systems in the TOML config:

.. code-block:: toml

   [profiler]
   enable_nsys = true
   nsys_target_roles = ["policy"]
   nsys_output_dir = "./outputs/nsys_profile"
   nsys_output_prefix = "qwen_grpo"
   nsys_trace = "cuda,nvtx,osrt,cudnn,cublas"
   nsys_extra_args = ["--sample=none"]

   [profiler.sub_profiler_config]
   wait_steps = 1
   warmup_steps = 1
   active_steps = 2
   rank_filter = [0]

When ``enable_nsys`` is true, ``launch_replica.sh`` wraps matching roles with:

.. code-block:: bash

   nsys profile \
     --capture-range=cudaProfilerApi \
     --capture-range-end=stop \
     --trace=cuda,nvtx,osrt,cudnn,cublas \
     --output=<nsys_output_dir>/<prefix>_<role>_%h_%p \
     torchrun ...

Only ranks in ``rank_filter`` call ``cudaProfilerStart`` / ``cudaProfilerStop``.
The capture window follows ``wait_steps + warmup_steps + active_steps`` and does
not change the model forward, backward, optimizer, rollout, or reward logic.

NVTX Markers
------------

Cosmos-RL emits lightweight NVTX ranges around the major policy and rollout
phases. These annotations are best-effort and automatically become no-ops when
CUDA/NVTX is unavailable.

Common ranges include:

- ``cosmos.nsys.capture ...`` for the active CUDA Profiler API capture window
- ``cosmos.policy.step_training ...`` for RL policy optimization steps
- ``cosmos.policy.sft_step ...`` for SFT policy optimization steps
- ``cosmos.rollout.rollout_generation ...`` for rollout generation calls

Keep ``nvtx`` in ``nsys_trace`` if you want these ranges to appear in the
Nsight Systems timeline.

Runtime Trigger
---------------

For RL policy replicas, you can still use the existing CLI command to choose a
policy replica at runtime:

.. code-block:: bash

   python -m cosmos_rl.cli.cli profile set <replica-name> \
     --active-steps 2 \
     --rank-filter 0 \
     -cp 8000 -ch localhost

The process must already have been launched with ``enable_nsys = true`` or the
``--nsys`` launcher flag. The CLI arms the runtime capture window; the launcher
provides the required ``nsys profile`` parent process.

Launcher Override
-----------------

For a quick smoke test without editing TOML, pass ``--nsys`` directly:

.. code-block:: bash

   COSMOS_CONTROLLER_HOST=localhost:8000 \
   ./cosmos_rl/launcher/launch_replica.sh \
     --type policy \
     --ngpus 8 \
     --config config.toml \
     --nsys \
     --nsys-output-dir ./outputs/nsys_smoke \
     --nsys-extra-arg --sample=none

Notes
-----

- Nsight Systems must be installed and ``nsys`` must be in ``PATH``.
- Capturing all ranks and all roles can produce very large reports. Start with
  one policy rank and a small ``active_steps`` value.
- For rollout or reference profiling, add the role to ``nsys_target_roles``.
  Additional runtime CUDA capture points may be needed if you want a narrower
  rollout-only window than whole-process capture.
