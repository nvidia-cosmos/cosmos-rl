export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TOKENIZERS_PARALLELISM=0

# export DEBUG=true
# export SAVE_FIRST_INPUT=1

# CONFIG=configs/pi05/pi05-libero10-grpo.toml

# ROBOT_PLATFORM=LIBERO PYTHONPATH=../LIBERO \
# uv run cosmos-rl --config $CONFIG \
#         --policy 4 \
#         --rollout 4 \
#         cosmos_rl/tools/dataset/libero_grpo.py

export COSMOS_CUDA_SYNC_DEBUG=1
export CUDA_LAUNCH_BLOCKING=1

CONFIG=configs/pi05/pi05-libero10-grpo-colocate.toml

ROBOT_PLATFORM=LIBERO PYTHONPATH=../LIBERO \
uv run cosmos-rl --config $CONFIG \
        cosmos_rl/tools/dataset/libero_grpo.py