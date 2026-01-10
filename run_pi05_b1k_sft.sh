source .venv/bin/activate
cosmos-rl --config configs/pi05/pi05_behavior_sft.toml ./tools/dataset/behavior_sft.py


# # PATH_TO_BEHAVIOR_1K=/workspace/noise_vla/Code/BEHAVIOR-1K
# cd $PATH_TO_BEHAVIOR_1K
# uv pip install -e bddl
# uv pip install -e OmniGibson[eval]