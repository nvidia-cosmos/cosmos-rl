import os
import unittest
import subprocess
import tempfile

config_content = """
redis = "12808"

[train]
resume = false
epoch = 1
output_dir = "./outputs/qwen2-5-3b-p-fsdp1-tp1-r-tp1-pp1-grpo"
epsilon = 1e-6
optm_name = "AdamW"
optm_lr = 1e-6
optm_impl = "fused"
optm_weight_decay = 0.01
optm_betas = [ 0.9, 0.999,]
optm_warmup_steps = 20
optm_grad_norm_clip = 1.0
async_tp_enabled = false
compile = false
param_dtype = "bfloat16"
fsdp_reduce_dtype = "float32"
fsdp_offload = false
fsdp_reshard_after_forward = "default"
train_batch_per_replica = 32
sync_weight_interval = 1

[rollout]
gpu_memory_utilization = 0.7
enable_chunked_prefill = false
max_response_length = 2048
n_generation = 16
batch_size = 4
quantization = "none"


[policy]
model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
model_max_length = 4096
model_gradient_checkpointing = false

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_rl"
experiment_name = "None"

[train.train_policy]
type = "grpo"
dataset.name = "JiaxinTsao/math_examples"
prompt_column_name = "prompt"
response_column_name = "result"
dataset.split = "train"
reward_function = "boxed_math"
temperature = 0.9
epsilon_low = 0.2
epsilon_high = 0.2
kl_beta = 0.0
mu_iterations = 1
min_filter_prefix_tokens = 1

[train.ckpt]
enable_checkpoint = false
save_freq = 20
save_mode = "async"

[rollout.parallelism]
n_init_replicas = 1
tp_size = 1
pp_size = 1

[policy.parallelism]
n_init_replicas = 1
tp_size = 1
cp_size = 1
dp_shard_size = 1
pp_size = 1
dp_replicate_size = 1

"""


class TestCustomArgs(unittest.TestCase):
    def test_custom_args(self):
        # save the config file to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            tmpfile.write(config_content)
            tmpfile_toml = tmpfile.name

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = f"{tmpfile_toml}"
        custom_dataset_script = os.path.join(
            current_dir, "custom_dataset/custom_gsm8k_grpo.py"
        )
        cmd = f"cosmos-rl --config {config_file} {custom_dataset_script} --foo cosmos cosmos_rl"
        process = subprocess.Popen(cmd, shell=True)

        process.wait()
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
