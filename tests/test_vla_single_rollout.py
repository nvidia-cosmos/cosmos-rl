#!/usr/bin/env python3
"""
Test script for single-card VLA inference with LIBERO simulation.
This helps debug VLA action generation quality.

Usage:
    python tests/test_vla_single_rollout.py --model_path <path> --task_suite LIBERO_10 --task_id 0
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path

# Set ROBOT_PLATFORM before any imports
os.environ["ROBOT_PLATFORM"] = "libero"

# Add cosmos-rl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cosmos_rl.rollout.vla_rollout.libero_utils import get_libero_env, normalize_gripper_action, invert_gripper_action, save_rollout_video
from transformers import AutoProcessor, AutoConfig
from cosmos_rl.policy.model.vla_utils import create_vla_config


def load_vla_model(model_path: str, device: str = "cuda:0", vla_type: str = "openvla-oft"):
    """Load VLA model on single GPU"""
    print(f"Loading VLA model from {model_path}")
    print(f"VLA Type: {vla_type}")
    
    # Resolve model path
    from cosmos_rl.utils.util import resolve_model_path
    try:
        resolved_path = resolve_model_path(model_path)
        if resolved_path != model_path:
            print(f"Resolved to: {resolved_path}")
        model_path = resolved_path
    except Exception as e:
        print(f"Using original path (resolution failed: {e})")
    
    # Detect VLA type from model path if not specified
    if "oft" in model_path.lower():
        vla_type = "openvla-oft"
    elif "openvla" in model_path.lower():
        vla_type = "openvla"
    
    print(f"Detected VLA type: {vla_type}")
    
    # Load config
    vla_config, processor, tokenizer = create_vla_config(
        model_path,
        cosmos_config=None,
        correct_pad_token=True,
        correct_gemma2=True,
        model=vla_type,  # Pass VLA type to ensure processor is created
        use_proprio=False,
        action_dim=7,
        num_images_in_input=1
    )
    
    print(f"Model config: use_proprio={vla_config.use_proprio}, action_dim={vla_config.proprio_dim}")
    if hasattr(vla_config, 'norm_stats') and vla_config.norm_stats:
        print(f"Norm stats loaded: {list(vla_config.norm_stats.keys())}")
    
    print(f"Processor: {type(processor).__name__ if processor else 'None'}")
    print(f"Tokenizer: {type(tokenizer).__name__ if tokenizer else 'None'}")
    
    # Load model using the VLA-specific approach
    print("Loading model weights...")
    
    # Import the correct model class based on VLA type
    if vla_type == "openvla-oft":
        # Use custom loading to avoid prismatic import issues
        from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        model = OpenVLAForActionPrediction.from_pretrained(
            model_path,
            config=vla_config,
            torch_dtype=torch.bfloat16,
        )
        
        # Set number of images
        if hasattr(model, 'vision_backbone'):
            model.vision_backbone.set_num_images_in_input(1)
        
        # Load dataset statistics if available
        import json
        dataset_stats_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_stats_path):
            with open(dataset_stats_path, "r") as f:
                norm_stats = json.load(f)
            model.norm_stats = norm_stats
            print(f"✅ Loaded norm_stats from dataset_statistics.json")
        elif hasattr(vla_config, 'norm_stats') and vla_config.norm_stats:
            model.norm_stats = vla_config.norm_stats
            print(f"✅ Using norm_stats from config")
        else:
            print(f"⚠️  No norm_stats found!")
            
    else:  # openvla
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            config=vla_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    return model, processor, tokenizer, vla_config


def prepare_vla_input(obs, instruction, processor, tokenizer, device):
    """Prepare observation and instruction for VLA model (matching SimpleVLA-RL exactly)"""
    from PIL import Image
    
    # Extract image
    image = obs.get("agentview_image", obs.get("image"))
    
    if image is None:
        raise ValueError("No image found in observation")
    
    # Convert numpy array to PIL Image and ensure RGB
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    
    # Create prompt (EXACT format from SimpleVLA-RL)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    
    # Process with PrismaticProcessor (positional args, NOT keyword args!)
    # This is the correct call signature matching SimpleVLA-RL
    batch_feature = processor(prompt, image)
    
    # Extract features
    input_ids = batch_feature["input_ids"]
    attention_mask = batch_feature["attention_mask"]
    pixel_values = batch_feature["pixel_values"]
    
    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    
    # For LIBERO with use_proprio=False, do NOT pass proprio
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "proprio": None,  # LIBERO does not use proprio
    }


def run_single_rollout(
    model_path: str,
    task_suite: str = "LIBERO_10",
    task_id: int = 0,
    max_steps: int = 600,
    device: str = "cuda:0",
    save_video: bool = True,
    vla_type: str = "openvla-oft"
):
    """Run a single rollout with VLA model"""
    
    # Load model
    model, processor, tokenizer, vla_config = load_vla_model(model_path, device, vla_type)
    
    # Initialize environment
    print(f"\nInitializing {task_suite} task {task_id}...")
    
    # Load task from LIBERO
    from libero.libero import benchmark
    from libero.libero.benchmark import get_benchmark
    
    task_suite_names = {
        "LIBERO_10": "libero_10",
        "LIBERO_90": "libero_90",
        "LIBERO_SPATIAL": "libero_spatial",
        "LIBERO_OBJECT": "libero_object",
        "LIBERO_GOAL": "libero_goal"
    }
    
    benchmark_name = task_suite_names.get(task_suite)
    if benchmark_name is None:
        raise ValueError(f"Unknown task suite: {task_suite}")
    
    # Get benchmark and task
    benchmark_suite = get_benchmark(benchmark_name)()
    task = benchmark_suite.get_task(task_id)
    
    env, task_description = get_libero_env(
        task=task,
        model_family="vla",
        resolution=256
    )
    
    print(f"Task: {task_description}")
    
    # Reset environment
    obs = env.reset()
    done = False
    step = 0
    
    # Storage for video
    video_frames = []
    action_log = []
    
    print(f"\nStarting rollout (max {max_steps} steps)...")
    print("=" * 80)
    
    while step < max_steps and not done:
        # Save frame
        if "agentview_image" in obs:
            img = obs["agentview_image"][::-1, ::-1]
            video_frames.append(img)
        
        # Prepare input
        vla_input = prepare_vla_input(obs, task_description, processor, tokenizer, device)
        
        # Generate action
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if hasattr(model, 'generate_action_verl'):
                # OpenVLA-OFT
                # Determine unnorm_key from norm_stats
                unnorm_key = None
                if hasattr(model, 'norm_stats') and model.norm_stats:
                    unnorm_key = list(model.norm_stats.keys())[0]  # Use first key
                
                actions, responses = model.generate_action_verl(
                    input_ids=vla_input["input_ids"],
                    pixel_values=vla_input["pixel_values"],
                    proprio=vla_input["proprio"],
                    attention_mask=vla_input["attention_mask"],
                    padding_idx=tokenizer.pad_token_id,
                    do_sample=True,
                    unnorm_key=unnorm_key,
                    temperature=1.6,  # Matching SimpleVLA-RL LIBERO config
                )
            elif hasattr(model, 'generate_action'):
                # OpenVLA
                actions, responses = model.generate_action(
                    input_ids=vla_input["input_ids"],
                    pixel_values=vla_input["pixel_values"],
                    attention_mask=vla_input["attention_mask"],
                    do_sample=True,
                    temperature=1.0,
                )
            else:
                raise ValueError("Model has no generate_action or generate_action_verl method")
        
        # Convert to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        # actions shape should be [1, num_chunks, action_dim] or [num_chunks, action_dim]
        if actions.ndim == 3:
            actions = actions[0]  # Remove batch dimension
        
        print(f"[Step {step:3d}] Generated {len(actions)} action chunks, shape: {actions.shape}")
        print(f"            First action: {actions[0]}")
        print(f"            Response: {responses[0] if isinstance(responses, list) else 'N/A'}")
        
        # Execute each action in the chunk
        for chunk_idx, action in enumerate(actions):
            # Process action for LIBERO
            normalized_action = normalize_gripper_action(action, binarize=True)
            processed_action = invert_gripper_action(normalized_action)
            
            # Step environment
            obs, reward, done, info = env.step(processed_action.tolist())
            step += 1
            
            # Log action
            action_log.append({
                'step': step,
                'raw_action': action.copy(),
                'processed_action': processed_action.copy(),
                'gripper': processed_action[-1],
                'reward': reward,
                'done': done
            })
            
            # Print compact action info
            gripper_state = "CLOSE" if processed_action[-1] > 0 else "OPEN"
            print(f"            └─ Chunk {chunk_idx}: pos={processed_action[:3]}, gripper={gripper_state}")
            
            if done:
                print(f"\n{'='*80}")
                print(f"Episode finished at step {step}")
                print(f"Success: {done}")
                print(f"{'='*80}\n")
                break
        
        if step % 80 == 0:  # Print separator every 10 VLA inferences (80 env steps)
            print("-" * 80)
    
    # Close environment
    env.close()
    
    # Summary
    print(f"\nRollout Summary:")
    print(f"  Total steps: {step}")
    print(f"  Success: {done}")
    print(f"  Frames collected: {len(video_frames)}")
    
    # Analyze actions
    if action_log:
        actions_array = np.array([log['raw_action'] for log in action_log])
        print(f"\nAction Statistics:")
        print(f"  Mean: {actions_array.mean(axis=0)}")
        print(f"  Std:  {actions_array.std(axis=0)}")
        print(f"  Min:  {actions_array.min(axis=0)}")
        print(f"  Max:  {actions_array.max(axis=0)}")
        
        grippers = [log['gripper'] for log in action_log]
        print(f"\nGripper Statistics:")
        print(f"  Open (< 0): {sum(1 for g in grippers if g < 0)} / {len(grippers)}")
        print(f"  Close (> 0): {sum(1 for g in grippers if g > 0)} / {len(grippers)}")
    
    # Save video
    if save_video and video_frames:
        output_dir = "./rollouts/test"
        os.makedirs(output_dir, exist_ok=True)
        
        video_path = os.path.join(
            output_dir,
            f"{task_suite}_task{task_id}_steps{step}_success{done}.mp4"
        )
        
        print(f"\nSaving video to {video_path}...")
        import imageio
        video_writer = imageio.get_writer(video_path, fps=30)
        for img in video_frames:
            video_writer.append_data(img)
        video_writer.close()
        print(f"✅ Video saved!")
    
    return done, step, action_log


def main():
    parser = argparse.ArgumentParser(description="Test VLA single rollout")
    parser.add_argument("--model_path", type=str, required=True, help="Path to VLA model")
    parser.add_argument("--task_suite", type=str, default="LIBERO_10", 
                       choices=["LIBERO_10", "LIBERO_90", "LIBERO_SPATIAL", "LIBERO_OBJECT", "LIBERO_GOAL"],
                       help="LIBERO task suite")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID within suite")
    parser.add_argument("--max_steps", type=int, default=600, help="Maximum steps per episode")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--vla_type", type=str, default="openvla-oft", 
                       choices=["openvla-oft", "openvla"],
                       help="VLA model type")
    parser.add_argument("--no_video", action="store_true", help="Disable video saving")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VLA Single Rollout Test")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"VLA Type: {args.vla_type}")
    print(f"Task: {args.task_suite} #{args.task_id}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    success, steps, action_log = run_single_rollout(
        model_path=args.model_path,
        task_suite=args.task_suite,
        task_id=args.task_id,
        max_steps=args.max_steps,
        device=args.device,
        save_video=not args.no_video,
        vla_type=args.vla_type
    )
    
    print(f"\n{'='*80}")
    print(f"Final Result: {'✅ SUCCESS' if success else '❌ FAILED'} ({steps} steps)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

