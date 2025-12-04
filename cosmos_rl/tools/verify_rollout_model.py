import os
import sys
import torch
import numpy as np
import toml
import traceback
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.model.vla import VLAModel, VLAArgs
from cosmos_rl.policy.model.vla_utils import create_vla_config
from cosmos_rl.utils.saved_batch_loader import SavedBatchLoader, SavedBatchIterator
from cosmos_rl.utils.logging import logger
import argparse
from cosmos_rl.utils import util
from cosmos_rl.policy.model.base import ModelRegistry

def verify_model(config_path):
    # 1. Load Config
    logger.info(f"Loading config from {config_path}")
    try:
        with open(config_path, "r") as f:
            config_dict = toml.load(f)
        config = Config.from_dict(config_dict)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    os.environ["ROBOT_PLATFORM"] = "LIBERO"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Create VLA Config and Args
    model_path = config.policy.model_name_or_path
    logger.info(f"Model path: {model_path}")
    
    vla_config, _, tokenizer = create_vla_config(
        model_path,
        cosmos_config=config,
        model=config.vla.vla_type
    )

    cosmos_default_dtype = util.str2torch_dtype(config.train.master_dtype)
    model = ModelRegistry.build_model(config).to(cosmos_default_dtype)

    # 3. Load Model
    logger.info("Loading VLAModel...")
    model.load_from_checkpoint(
        model_name_or_path=model_path,
        parallel_dims=None,
        device=device
    )
    model.to(device)
    model.eval()

    # for name, param in model.named_parameters():
    #     logger.info(f"{name} param.requires_grad: {param.requires_grad}, param.shape: {param.shape}, param.dtype: {param.dtype}")
    
    logger.info("✅ Model loaded successfully")

    logger.info("=" * 80)
    logger.info(f"[TEST] inv_freq saved in model: {model.model.language_model.model.rotary_emb.inv_freq}")
    logger.info("=" * 80)

    # 4. Load Saved Batch and Verify (following vla_rollout.py pattern)
    logger.info("=" * 80)
    logger.info("[TEST] Loading saved batch from SimpleVLA-RL to compare logprobs")
    logger.info("=" * 80)

    batch_dir = '/root/SimpleVLA-RL/saved_training_batches'
    try:
        # Use same pattern as grpo_trainer.py
        loader = SavedBatchLoader(batch_dir=batch_dir, episodes_per_step=128, device='cpu')
        iterator = SavedBatchIterator(loader)
        
        # Load first batch using iterator
        policy_inputs_all, advantages_all, meta_info = next(iterator)
        logger.info(f"[TEST] Loaded {len(policy_inputs_all)} total episodes from saved batch")

        if policy_inputs_all and len(policy_inputs_all) > 0:
            # Use first episode from saved batch
            saved_episode = policy_inputs_all[0]
            
            logger.info(f"[TEST] Episode 0: {saved_episode.num_steps} steps, task_id={saved_episode.task_id}, trial_id={saved_episode.trial_id}")
            
            # Extract data from first step
            step_0 = saved_episode.per_step_data[0]
            test_input_ids = step_0['input_ids'].unsqueeze(0).to(device)  # (1, seq_len)
            test_attention_mask = step_0['attention_mask'].unsqueeze(0).to(device)
            
            # pixel_values is a list of tensors in RLPolicyInput
            pixel_values_tensor = saved_episode.pixel_values[0]
            test_pixel_values = pixel_values_tensor.unsqueeze(0).to(device)  # (1, 6, 224, 224)
            
            test_responses = step_0['responses'].unsqueeze(0).to(device)  # (1, 56)
            saved_old_log_prob = step_0['old_log_prob']  # (56,)
            
            logger.info(f"[TEST] Input shapes:")
            logger.info(f"  input_ids: {test_input_ids.shape}, {test_input_ids.dtype}")
            logger.info(f"  attention_mask: {test_attention_mask.shape}, {test_attention_mask.dtype}")
            logger.info(f"  pixel_values: {test_pixel_values.shape}, {test_pixel_values.dtype}")
            logger.info(f"  responses: {test_responses.shape}, {test_responses.dtype}")
            logger.info(f"  saved_old_log_prob: {saved_old_log_prob.shape}, {saved_old_log_prob.dtype}")
            
            # Decode text if tokenizer available
            if tokenizer:
                prompt_text = tokenizer.decode(test_input_ids[0], skip_special_tokens=False)
                logger.info(f"  Prompt: {prompt_text[:100]}...")
            
            # Test with different temperatures
            test_temps = [1.6]
            
            for temp in test_temps:
                logger.info(f"\n[TEST] Testing with temperature={temp}")
                
                # Compute logprobs using model's forward_with_trajectory_structure
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    with torch.no_grad():
                        test_outputs = model.forward_with_trajectory_structure(
                            input_ids=test_input_ids,
                            pixel_values=test_pixel_values,
                            attention_mask=test_attention_mask,
                            labels=test_responses,
                            temperature=temp,
                            proprio=None
                        )
                        rollout_logits = test_outputs.logits
                        rollout_log_probs = test_outputs.logprobs.squeeze(0)  # (56,)
                
                logger.info(f"  Computed logprobs:")
                logger.info(f"    logits shape: {rollout_logits.shape}, dtype: {rollout_logits.dtype}")
                logger.info(f"    log_probs shape: {rollout_log_probs.shape}, dtype: {rollout_log_probs.dtype}")
                logger.info(f"    log_probs[0:10]: {rollout_log_probs[0:10]}")
                
                logger.info(f"  Saved logprobs from SimpleVLA-RL:")
                logger.info(f"    saved_old_log_prob[0:10]: {saved_old_log_prob[0:10]}")
                
                # Compute differences
                diff = (rollout_log_probs.cpu() - saved_old_log_prob).abs()
                logger.info(f"  Absolute differences:")
                logger.info(f"    Mean: {diff.mean().item():.6f}")
                logger.info(f"    Max: {diff.max().item():.6f}")
                logger.info(f"    Min: {diff.min().item():.6f}")
                logger.info(f"    Std: {diff.std().item():.6f}")
                logger.info(f"    diff[0:10]: {diff[0:10]}")
            
            logger.info("=" * 80)
        else:
            logger.warning("[TEST] No saved batches available")

    except Exception as e:
        logger.error(f"[TEST] Failed to verify model: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify VLA Rollout Model")
    parser.add_argument("config", help="Config file")
    args = parser.parse_args()
    verify_model(args.config)
