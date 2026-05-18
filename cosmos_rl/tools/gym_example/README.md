# Gymnasium Classic Control example

A minimal end-to-end demo of running cosmos-rl with a non-LLM RL
workload, using the [Gymnasium Classic Control suite](https://gymnasium.farama.org/environments/classic_control/)
as the reference environment.

## What this demonstrates

* The Gym API extension hooks (`register_tokenizer_loader`,
  `register_local_model_config`, `TensorDataPacker`,
  `IdentityWeightMapper`) wiring a tiny MLP policy and a
  `gymnasium.Env` into the standard cosmos-rl pipeline.
* Two payload-transfer profiles (Redis and UCXX) selected via
  `[custom].payload_transfer` so colocated and disaggregated launches
  share the same code.
* Both **discrete** (CartPole-v1) and **continuous** (Pendulum-v1)
  action spaces.

## Layout

```
cosmos_rl/tools/gym_example/
├── __init__.py            re-exports the example surface
├── README.md              this file
├── gym_policy.py          GymPolicy + GymMLPConfig + register_gym_policy()
├── gym_rollout.py         GymRolloutEngine + rollout_episode()
├── gym_data_packer.py     GymDataPacker (TensorDataPacker subclass)
└── configs/
    ├── cartpole_colocated.toml       primary, Redis transport
    ├── cartpole_disaggregated.toml   UCXX transport
    └── pendulum_colocated.toml       continuous-action variant
```

## Install

```bash
pip install "cosmos_rl[gym]"        # primary path (CartPole / Pendulum)
pip install "cosmos_rl[gym,ucxx]"   # add UCXX transport for disaggregated mode
```

## Standalone (no controller) sanity check

The policy and rollout engine can be exercised directly without
spinning up the full cosmos-rl controller / dispatcher — useful for
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

## Wiring it into cosmos-rl

```python
from cosmos_rl.tools.gym_example import (
    GymDataPacker, GymPolicy, register_gym_policy,
)
from cosmos_rl.policy.model.identity_weight_mapper import IdentityWeightMapper

# 1. Register the .toml -> NoOpTokenizer + GymMLPConfig handlers.
register_gym_policy()

# 2. Use IdentityWeightMapper for the (single-rank, non-sharded) MLP.
#    Hand the policy and mapper to ModelRegistry.register_model() in
#    your custom_class.py, and set:
#       [policy] model_name_or_path = "configs/cartpole_colocated.toml"
#       [policy] model_class        = "GymPolicy"
#       [policy] data_packer_class  = "GymDataPacker"
```

The full launcher integration lives in your project's `custom_class.py`
because cosmos-rl uses dotted-path imports for trainer / data-packer /
weight-mapper resolution; the example components above are designed to
be drop-in.

## Pendulum-v1

Swap the config and pass `discrete=False`:

```toml
[model]
obs_dim    = 3
action_dim = 1
discrete   = false

[env]
name      = "Pendulum-v1"
max_steps = 200
```

`GymPolicy` automatically swaps in a Gaussian (mean / log_std) head
for continuous-action environments.

## Going beyond Classic Control

Because the rollout engine is just a thin driver around a user-supplied
`env_factory`, swapping in a more interesting environment (e.g.
`gym.make("LunarLander-v2")`) is a one-line change.  Match the
config's `obs_dim` / `action_dim` / `discrete` fields to the new
environment's `observation_space` / `action_space` and you're done.

## Profiling

When the [profiler](../profiler/README.md) tooling lands, this example
ships with no extra wiring needed — the rollout engine already emits
`[Trace]` lines through `cosmos_rl.utils.trace.format_trace()` (when
the trace utility MR is also installed) and the analyzer picks them up
automatically.
