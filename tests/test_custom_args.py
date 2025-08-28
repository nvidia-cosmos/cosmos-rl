import os
import unittest
import subprocess

class TestCustomArgs(unittest.TestCase):
    def test_custom_args(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(current_dir, "../configs/qwen2-5/qwen2-5-3b-p-fsdp1-tp1-r-tp1-pp1-grpo.toml")
        custom_dataset_script = os.path.join(current_dir, "custom_dataset/custom_gsm8k_grpo.py")
        cmd = f"cosmos-rl --config {config_file} {custom_dataset_script} --foo cosmos cosmos_rl"
        process = subprocess.Popen(cmd, shell=True)

        process.wait()
        self.assertEqual(process.returncode, 0)

if __name__ == "__main__":
    unittest.main()