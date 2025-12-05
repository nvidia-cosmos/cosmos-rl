# Custom GRPO Training Flow Guide

## Overview

This guide explains how to define a custom GRPO flow, including customizing both policy trainer and rollout generation parts.

## Quick Start

To create a complete custom GRPO training flow:

1. **Implement a Custom GRPO Trainer** by extending `GRPOTrainer` and register it with a unique `train_policy.trainer_type` type
2. **Implement a Custom Rollout Engine** by extending `RolloutBase` and register it with a unique `rollout.backend` type
3. **Specify the custom types** in your configuration file (e.g., `train_plocy.trainer_type` and `rollout.backend`)
4. **Launch the training** using the existing launcher with your custom configuration

The framework will automatically use your custom implementations based on the registered types. All the usages are the same as the usages in existing framework.

## How to Create a Custom GRPO Trainer

### Step 1: Implement the Custom GRPO Trainer

By extending the base `GRPOTrainer` class. You can override methods in `GRPOTrainer` to customize the training behavior as needed, such as the loss function, logprob computation, training step, lr scheduler, and more. Create a new Python file to define your custom trainer class that extends the `GRPOTrainer` class.

**Example**
Class `CustomGRPOTrainer` in `cosmos_rl/tools/example/custom_grpo_trainer.py`


**Key Methods You Can Override:**

1. **`loss_fn()`** - Customize the GRPO loss computation logic
   - Takes current token logprobs, old token logprobs, reference token logprobs, advantages, sequence lengths, etc.
   - Returns a tuple of (scalar loss, per-token loss, kl loss)
   - The scalar loss is used for back-propagation; others are for logging

2. **`compute_logprobs()`** - Modify how per-token log probabilities are calculated
   - Takes minibatch data including input ids like information, model logits, and a flag indicating if logits are full or pre-selected
   - Returns per-token log probabilities, logprob masks, and metrics dictionary (e.g., entropy metrics)

3. **`step_training()`** - Customize the entire training step logic
   - Takes rollouts, current step number, total steps, remaining samples, inter-policy NCCL communicator, master replica flag, and checkpoint flag
   - Returns a dictionary of training metrics for logging and reporting
   - Argument `rollouts` is exactly the results put by rollout backend and can be an identifier to the sample data for training

4. **`build_lr_schedulers()`** - Customize learning rate scheduler creation
   - Returns an `LRSchedulersContainer` with your custom learning rate schedulers

5. **`build_optimizers()`** - Customize optimizer creation
   - Build and configure optimizers for the model parameters
   - Normally no need to customize and use the default Adam optimizers

6. **`model_load_from_hf()`** - Customize model loading from HuggingFace
   - Override how the model is loaded and initialized from HuggingFace Hub
   - Normally no need to customize and use the default implementation

7. **`model_resume_from_checkpoint()`** - Customize checkpoint resumption
   - Override how the model state is restored from saved checkpoints
   - Normally no need to customize and use the default implementation

8. **`export_safetensors()`** - Customize model export and saving
   - Override how model weights are saved to disk
   - Normally no need to customize and use the default implementation

### Step 2: Register the Custom Trainer with TrainerRegistry

Use the `@TrainerRegistry.register()` decorator to register your custom trainer with a unique `trainer_type` identifier. This registration allows the framework to automatically instantiate your custom trainer when the corresponding type is specified in the configuration.

**Registration Pattern:**
```
@TrainerRegistry.register(trainer_type="your_custom_trainer_name")
class YourCustomTrainer(GRPOTrainer):
    # Your implementation
```

The `trainer_type` you specify here will be used in your configuration file as `train_policy.trainer_type` to select this trainer.

### Step 3: Specify in Configuration

In your TOML configuration file, specify your custom `trainer_type` in the `[train.train_policy]` section:

**Configuration Parameters:**
- `trainer_type` - Set this to the name you registered with `@TrainerRegistry.register()`

**Example Configuration:**
Reference example configurations can be found in paths like `configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml`



## How to Customize the Rollout Process

### Step 1: Create a Custom Rollout Class

To customize the rollout generation process, implement a custom rollout engine to extend the `RolloutBase` class with needed methods override. This allows you to control how prompts or data inputs are processed and how completions or data outputs are generated during the rollout phase.


**Example**
Class `ExampleHFRollout` in `cosmos_rl/rollout/example_custom_rollout/example_custom_rollout.py`


**Key Methods You Can Override:**
Your custom rollout engine should implement the following key methods:

1. **`__init__()`** - Initialize the rollout engine
   - Takes `config`, `parallel_dims`, `device`, and optional kwargs
   - Call `super().__init__()` to initialize the base class
   - Set up any custom attributes needed for your rollout engine

2. **`post_init_hook()`** - Additional initialization after model parallelization
   - Called after the base initialization to perform additional setup
   - Main setup functionalities can be implemented here

3. **`rollout_generation()`** - The core generation method (required)
   - Takes a list of `RLPayload` objects containing inputs and metadata for rollout generation
   - Takes a `torch.cuda.Stream` for asynchronous execution on the stream
   - Takes a `BaseDataPacker` for data formatting and processing
   - Takes a `DataFetcherBase` for access training dataset and validation dataset to fetch data if needed
   - Takes an `is_validation` flag to distinguish validation from training generation
   - Returns a list of `RolloutResult` objects containing:
     - `prompt` - The original prompt
     - `completions` - List of generated completion strings, or list of any objects such as tensors and videos for native tensor mode, video mode, or any other non-text modes
     - `completion_logprobs` - (Optional) Log probabilities for each token
     - `completion_token_ids` - (Optional) Token IDs for completions

4. **`init_engine()`** - Initialize the underlying model/engine
   - Loads the model from HuggingFace or other sources
   - Build the exact rollout engine for generation execution
   - Sets up tokenizer and generation configurations
   - Configures separate setting parameters for training and validation generation according to the fields in `[rollout]` of the configuration toml
   - Maybe set up the simulation functionalities if needed
   - Sets `self._engine_initialized = True` when complete

5. **`get_underlying_model()`** - Return the underlying model
   - Returns the model instance for weight synchronization or inspection

6. **`set_underlying_model()`** - Set the underlying model
   - Takes a `torch.nn.Module` to replace the current model
   - Used for weight updates from the policy worker in the colocated and shared model case

### Step 2: Register the Custom Rollout Backend with RolloutRegistry

Then register the custom rollout backend with the `RolloutRegistry`.
Use the `@RolloutRegistry.register()` decorator with a unique identifier:

```
@RolloutRegistry.register("your_custom_rollout")
class YourCustomRollout(RolloutBase):
    # Your implementation
```

The identifier you specify (e.g., "example_hf") will be used in your configuration as the `rollout.backend`.


### Step 3: Specify in Configuration

In your TOML configuration file, set the `rollout.backend` to match your registered identifier:

```
[rollout]
backend = "your_custom_rollout"
```

**Example Configuration:**
Reference example configurations can be found in paths like `configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml`

## Launch the Custom Training Flow

Execute the launcher with both the updated configuration file including the custom `train_policy.trainer_type` and `rollout_backend`. Also, use an entry script where the custom grpo trainer and custom rollout backend have been registered and imported.

```bash
cosmos-rl --config configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml tools/example/custom_grpo_trainer.py
```

**Portability to both `colocated` and `disaggregated` mode**
The above launching example is in colocated mode by default, where the policy and rollout replica shared the same GPUs within the same process. The customization can seamlessly ports to disaggregated case where the policy and rollout replica are on separate GPUs and processes. By only 
replacing the line `mode = "colocated"` with `mode = "disaggregated"` in `configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml`, the disaggregated workflow can be launched using exactly the same command. The customization is totally portable and can be directly used in either `colocated` or `disaggregated` mode.

## Implementation Notes

**Model Processing:**
- Ensure the model is compatible with the parallelism configuration
- Handle the parallelism processing if needed

**Data Processing:**
In `rollout_generation`:
- Process each `RLPayload` to get its prompts if any and then do generation
- If `prompt_idx` in `RLPayload`, can directly use `prompt_idx` to index the input data for generation, such as in native tensor mode with non-text format
- Use `data_packer.get_rollout_output()` to post-process the generated samples including inputs and outputs of the generation, we may replace the whole sample with some unique identifier

In `step_training`:
- Use `data_packer.get_policy_input()` to pre-process the rollout results, its arguments are from the format of `data_packer.get_rollout_output()` returns, so it can transfer the replaced unique identifier back to the complete sample data if any
- `data_packer.get_policy_input()` and `data_packer.get_rollout_output()` are vice versa, and should be defined in custom `data_packer` like in `tools/dataset/post_completion.py`


**Reference Implementation:**
For a custom rollout generation example, refer to `cosmos_rl/rollout/example_custom_rollout/example_custom_rollout.py`.

For a custom grpo trainer example, refer to `cosmos_rl/tools/example/custom_grpo_trainer.py`. 

For an example configuration file using the above custom implementations, refer to `configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml`.

For an example data processing between policy trainer and rollout backend, refer to `tools/dataset/post_completion.py`.


## Summary

By implementing both a custom GRPO trainer and a custom rollout engine, you gain complete control over the entire GRPO training pipeline while leveraging the existing framework infrastructure.

The key to customization is the registry system:
- **Register your trainer** with `@TrainerRegistry.register(trainer_type="...")`
- **Register your rollout** (optional) with `@RolloutRegistry.register(rollout_backend_type="...")`
- **Specify in config** using the registered type names
- **Launch** and the framework automatically uses your custom implementations

This design allows you to experiment with novel RL algorithms while maintaining compatibility with the distributed training infrastructure.

