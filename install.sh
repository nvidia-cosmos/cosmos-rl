# apt-get install -y tmux
# tmux new

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
apt-get install -y python3.10-dev python3-dev
apt -o Dpkg::options::="--force-overwrite" install -y libnvidia-gl-570 --no-install-recommends

cd /workspace
git clone https://github.com/Dao-AILab/flash-attention.git
git clone https://github.com/littlespray/cosmos-rl.git

cd cosmos-rl
git checkout remotes/origin/feat/support-pi0
uv sync --extra vla
uv pip install wandb
source .venv/bin/activate


cd /workspace
cd flash-attention
python setup.py install

cd /workspace/cosmos-rl

hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir /workspace/RLinf-Pi05-LIBERO-SFT
hf download sunshk/pi05_libero_pytorch --local-dir /workspace/pi05_libero_pytorch
cp /workspace/pi05_libero_pytorch/*.model /workspace/RLinf-Pi05-LIBERO-SFT/
cp /workspace/pi05_libero_pytorch/*.json /workspace/RLinf-Pi05-LIBERO-SFT/
cp /workspace/pi05_libero_pytorch/*.py /workspace/RLinf-Pi05-LIBERO-SFT/
mkdir /workspace/RLinf-Pi05-LIBERO-SFT/assets
mv /workspace/RLinf-Pi05-LIBERO-SFT/physical-intelligence /workspace/RLinf-Pi05-LIBERO-SFT/assets/physical-intelligence
