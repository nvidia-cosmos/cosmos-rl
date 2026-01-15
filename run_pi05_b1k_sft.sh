source .venv/bin/activate

# --- Debug alignment env (cosmos-rl <-> openpi-comet) ---
# Shared dump directory for:
#   - unprocessed_sample.pkl (raw BehaviorLeRobotDataset item)
#   - cosmos_model_input.pkl / cosmos_model_output.pkl
#   - openpi_model_input.pkl / openpi_model_output.pkl
#
# For real training runs, keep debug disabled:
#export PI_DEBUG_ENABLE="${PI_DEBUG_ENABLE:-1}"
#export PI_DEBUG_DUMP_DIR="${PI_DEBUG_DUMP_DIR:-/workspace/pi_debug_align}"
#export PI_DEBUG_SEED="${PI_DEBUG_SEED:-0}"
# Force cosmos-rl to always use the same raw unprocessed sample from PI_DEBUG_DUMP_DIR.
#export PI_DEBUG_FORCE_UNPROCESSED="${PI_DEBUG_FORCE_UNPROCESSED:-0}"

cosmos-rl --config configs/pi05/pi05_behavior_sft.toml ./tools/dataset/behavior_sft.py



# apt-get install -y python3.10-dev python3-dev

# GIT_LFS_SKIP_SMUDGE=1 uv sync
# GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

# source .venv/bin/activate

# cd /workspace/noise_vla/Code/BEHAVIOR-1K
# uv pip install -e bddl
# uv pip install -e OmniGibson[eval]

# cd /workspace/openpi-comet
# cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/


