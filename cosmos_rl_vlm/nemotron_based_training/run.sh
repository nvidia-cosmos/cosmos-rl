# You should use pre-built docker image

source /opt/venv/cosmos_rl/bin/activate

# install new version of cosmos-rl
pip uninstall -y cosmos-rl

# when open-source cosmos-rl is ready
git clone https://github.com/nvidia-cosmos/cosmos-rl && cd cosmos-rl && pip install -e . && cd ..

# If you want to use wandb to log metric, install it
pip install wandb

COSMOS_HEARTBEAT_TIMEOUT=600 COSMOS_LOG_LEVEL=DEBUGG COSMOS_NCCL_TIMEOUT_MS=60000000 COSMOS_GLOO_TIMEOUT=60000 cosmos-rl --config nemotron_based_vlm.toml --port 8000 --rdzv-port 29345 launcher.py 