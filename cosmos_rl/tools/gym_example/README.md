> [!IMPORTANT]
> ## 🚀 [Cosmos 3 Has Arrived](https://github.com/nvidia/cosmos)
>
> Cosmos 3 is NVIDIA's next-generation foundation model platform for Physical AI. Compared with Cosmos-RL, Cosmos 3 unifies reasoning, world prediction, simulation, transfer, and action generation within a single model family and ecosystem.
>
> Rather than relying on separate models for reasoning, prediction, transfer, and policy learning, a single Cosmos 3 model can understand the world, reason about physical interactions, predict future outcomes, transform observations across domains, and generate actions for embodied agents. This unified architecture enables stronger performance across a broad range of Physical AI applications, including robotics, autonomous vehicles, and smart spaces.
>
> This repository is no longer under active development and will receive only limited maintenance updates. Future model releases, features, documentation, and community support will be focused on Cosmos 3.
>
> 👉 Visit the new Cosmos home: https://github.com/nvidia/cosmos
>
> There you will find the latest Cosmos 3 models, technical reports, tutorials, benchmarks, and ecosystem updates.
>
> Thank you for your support of Cosmos-RL. We encourage all users to migrate to Cosmos 3 for the latest state-of-the-art Physical AI capabilities.

# Gymnasium Classic Control example

A minimal end-to-end demo of running cosmos-rl with a non-LLM RL
workload, using the [Gymnasium Classic Control suite](https://gymnasium.farama.org/environments/classic_control/)
as the reference environment.

This example is also the **first upstream consumer of the
trajectory-iteration contract** (`TrajectoryExpansionMixin` /
`TrajectoryPacker`). `GymTrainer` composes the mixin in **rollout
mode** (`chunk_size = None`, override `_train_one_rollout`) because
gym tensor trajectories are small enough that per-rollout iteration
is the right shape.

## What this demonstrates

* The trajectory-iteration contract on a tiny CPU-runnable workload:
  `GymDataPacker` satisfies the `TrajectoryPacker` Protocol;
  `GymTrainer` composes `TrajectoryExpansionMixin` and walks
  rollouts through `_begin_training_step` -> N x `_train_one_rollout`
  -> `_finalize_training_step`.
* The Gym API extension hooks (`register_tokenizer_loader`,
  `register_local_model_config`, `TensorDataPacker`,
  `IdentityWeightMapper`) wiring a tiny MLP policy and a
  `gymnasium.Env` into the standard cosmos-rl pipeline.
* Both **discrete** (CartPole-v1) and **continuous** (Pendulum-v1)
  action spaces.
* The **rollout-side `RolloutGenerationMixin`** (`cosmos_rl.rollout.
  generation_mixin`).  `GymRolloutBackend` is the first upstream
  consumer: it composes the mixin and overrides four hooks
  (`_prepare_sample`, `_collate_batch`, `_generate`, `_postprocess`)
  instead of writing a bespoke `rollout_generation`, and gets
  background per-prompt setup overlap "for free" when
  `[rollout].prefetch_rollout = true` is set in the config.

## Layout

```
cosmos_rl/tools/gym_example/
+-- __init__.py            re-exports the example surface
+-- README.md              this file
+-- gym_policy.py          GymPolicy + GymMLPConfig + register_gym_policy()
+-- gym_rollout.py         GymRolloutEngine + rollout_episode()
+-- gym_data_packer.py     GymDataPacker (TensorDataPacker + TrajectoryPacker)
+-- gym_algo.py            compute_returns + compute_simple_pg_loss (toy PG)
+-- gym_trainer.py         GymTrainer(TrajectoryExpansionMixin, Trainer)
+-- gym_rollout_backend.py GymRolloutBackend(RolloutBase)
+-- gym_entry.py           launch entry: GymSeedDataset + gym_episode_reward
+-- configs/
    +-- cartpole_colocated.toml       primary, Redis transport
    +-- pendulum_colocated.toml       continuous-action variant
```

## Install

```bash
pip install "cosmos_rl[gym]"        # primary path (CartPole / Pendulum)
```

## Standalone (no controller) sanity check

The policy and rollout engine can be exercised directly without
spinning up the full cosmos-rl controller / dispatcher, useful for
local iteration and debugging:

```python
import gymnasium as gym
from cosmos_rl.tools.gym_example import (
    GymMLPConfig, GymPolicy, GymRolloutEngine,
)

policy = GymPolicy(GymMLPConfig(obs_dim=4, action_dim=2, discrete=True))
engine = GymRolloutEngine(
    env_factory=lambda: gym.make("CartPole-v1"),
    policy=policy,
    max_steps=200,
)
traj = engine.run({"seed": 42})
print({k: v.shape for k, v in traj.items()})
# {'observations': (200, 4), 'actions': (200,), 'rewards': (200,),
#  'terminated': (200,), 'truncated': (200,), 'episode_length': (1,)}
```

## Trajectory-iteration contract quickstart

The trainer-side path uses the upstream
`TrajectoryExpansionMixin` exactly as a downstream consumer would:

```python
from cosmos_rl.tools.gym_example import GymDataPacker, GymTrainer
from cosmos_rl.dispatcher.data.packer.trajectory_packer import TrajectoryPacker

# GymDataPacker satisfies the TrajectoryPacker Protocol:
assert isinstance(GymDataPacker(), TrajectoryPacker)
# num_transitions(rollout) -> int (=== episode_length)
# iter_transitions(rollout) -> Iterator[{observation, action, reward}]
# iter_chunks   inherited from the protocol's default body
# iter_rollouts inherited from the protocol's default body

# GymTrainer composes TrajectoryExpansionMixin in rollout mode:
assert GymTrainer.chunk_size is None
# _begin_training_step          -> per-step setup (zero_grad, scratch buffers)
# _train_one_rollout            -> per-trajectory forward + backward
# _finalize_training_step       -> grad clip, optimizer step, return metrics
```

## Supported launch matrix

|                          | colocated, single replica | colocated, multi replica | disaggregated |
|--------------------------|:-------------------------:|:------------------------:|:-------------:|
| Pytest end-to-end        | yes                       | n/a                      | n/a           |
| `cosmos-rl --config ...` | **yes**                   | no                       | no            |

The trajectory-iteration contract is validated by
`tests/test_gym_example.py` (CPU, no controller); the launcher
integration is validated by hand-running the colocated single-replica
config below.  Disaggregated and multi-replica are out of scope: the
toy trainer's `sync_all_states` is a no-op (returns `0` parameters
transferred) and `map_w_from_policy_to_rollout` is empty, so any
launch that relies on inter-replica or P2R weight transport will
silently produce stale weights on the rollout side.

## Wiring it into cosmos-rl

```bash
cosmos-rl --config cosmos_rl/tools/gym_example/configs/cartpole_colocated.toml \
          cosmos_rl/tools/gym_example/gym_entry.py
```

`gym_entry.py` registers the trainer (`gym_pg`), the rollout backend
(`gym`), and the gym MLP policy loader on import, then hands a
`GymSeedDataset`, `GymDataPacker`, and `gym_episode_reward` to
cosmos-rl's `launch_worker`.

The dataset is **not** training data: for an RL workload the
"dataset" is a stream of initial conditions (here, JSON-encoded
seeds) that the dispatcher hands to the rollout backend. The
rollout backend then drives `gymnasium.Env` via `GymRolloutEngine`
to produce per-episode trajectories.

## Pendulum-v1

```bash
cosmos-rl --config cosmos_rl/tools/gym_example/configs/pendulum_colocated.toml \
          cosmos_rl/tools/gym_example/gym_entry.py
```

`GymPolicy` automatically swaps in a Gaussian (mean / log_std) head
for continuous-action environments.

## Going beyond Classic Control

Because the rollout engine is just a thin driver around a user-supplied
`env_factory`, swapping in a more interesting environment (e.g.
`gym.make("LunarLander-v2")`) is a one-line change. Match the
config's `obs_dim` / `action_dim` / `discrete` fields to the new
environment's `observation_space` / `action_space` and you're done.

## Toy semantics

This trainer is deliberately toy. It exists to validate the
trajectory-iteration contract end-to-end on CPU, not to be a
competitive RL implementation. Not in scope:

* Real PG algorithms (PPO / A2C / REINFORCE-with-baseline). The toy
  loss is MSE between predicted and sampled actions, weighted by
  return.
* Real persistence (export_safetensors / model_load_from_hf /
  model_resume_from_checkpoint are no-ops with a warning).
* Real validation step.
* Real LR scheduler.
* Disaggregated weight sync between rollout and policy ranks.

Each of these is a clean follow-up if a real workload wants to
adopt the gym example as a starting point.

## Profiling

When the [profiler](../profiler/README.md) tooling lands, this example
ships with no extra wiring needed: the rollout engine already emits
`[Trace]` lines through `cosmos_rl.utils.trace.format_trace()` (when
the trace utility MR is also installed) and the analyzer picks them up
automatically.
