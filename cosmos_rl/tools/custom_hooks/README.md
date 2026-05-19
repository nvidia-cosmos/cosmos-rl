# Custom Logger Functions and Training Hooks

Cosmos-RL provides a flexible system for integrating custom logging and monitoring through `custom_logger_fns` and `hook_fns`. This allows you to integrate external systems (e.g., TAO, MLflow, custom monitoring) without modifying the core training code.

## Custom Logger Functions

Custom logger functions are called after each training step and validation to report metrics.

**Signature:**
```python
def custom_logger_fn(report_data: Dict[str, Any], step: int) -> None:
    """
    Args:
        report_data: Dictionary containing training/validation metrics
        step: Current training step
    """
    pass
```

**Example:**
```python
from cosmos_rl.launcher.worker_entry import main
from cosmos_rl.tools.custom_hooks import create_status_logger

# Create a custom logger that sends status to an external endpoint
status_logger = create_status_logger(
    endpoint="http://monitoring-server/api/status",
    component_name="My SFT Training"
)

main(custom_logger_fns=[status_logger])
```

## Training and Validation Hooks

Hooks provide fine-grained control over the training lifecycle. They are called at specific points during training and validation.

**Available Hooks:**

| Hook Name | When Called | report_data Contents |
|-----------|-------------|---------------------|
| `pre_training_hook` | Before training loop starts | total_epochs, total_steps, start_epoch, start_step |
| `pre_training_step_hook` | Before each training step | current_epoch, current_step, total_steps |
| `post_training_step_hook` | After each training step | current_epoch, current_step, total_steps, + training metrics |
| `post_training_hook` | After training completes | final_epoch, final_step, total_steps, final_val_loss |
| `pre_validation_hook` | Before validation starts | current_epoch, is_last_step |
| `pre_per_step_validation_hook` | Before each validation batch | current_epoch, batch_index |
| `post_per_step_validation_hook` | After each validation batch | current_epoch, batch_index, val_score |
| `post_validation_hook` | After validation completes | current_epoch, val_avg_loss |

**Hook Signature:**
```python
def hook_fn(worker, report_data: Dict[str, Any]) -> None:
    """
    Args:
        worker: The SFTPolicyWorker instance (access to trainer, config, etc.)
        report_data: Dictionary containing context-specific data
    """
    pass
```

**Example with TAO-like Integration:**
```python
from cosmos_rl.launcher.worker_entry import main
from cosmos_rl.tools.custom_hooks import TAOStatusLogger

# Create TAO-style status logger
tao_logger = TAOStatusLogger(
    status_endpoint="http://tao-server/api/status",
    experiment_name="sft-experiment-001"
)

# Launch with both custom logger and hooks
main(
    custom_logger_fns=[tao_logger.log_status],
    hook_fns=tao_logger.get_hooks(),
)
```

**Using Individual Hook Creators:**
```python
from cosmos_rl.launcher.worker_entry import main
from cosmos_rl.tools.custom_hooks import (
    create_status_logger,
    create_training_hooks,
    create_validation_hooks,
)

# Create custom logger
logger_fn = create_status_logger(
    endpoint="http://localhost:8080/status",
    component_name="Training Monitor"
)

# Create hooks
training_hooks = create_training_hooks(
    status_endpoint="http://localhost:8080/status",
    heartbeat_interval=10  # Send heartbeat every 10 steps
)
validation_hooks = create_validation_hooks(
    status_endpoint="http://localhost:8080/status"
)

# Merge all hooks
all_hooks = {**training_hooks, **validation_hooks}

main(
    custom_logger_fns=[logger_fn],
    hook_fns=all_hooks,
)
```

## Benefits of Using Hooks and Custom Loggers

1. **Separation of Concerns**: Keep external system integrations separate from core training logic
2. **Flexibility**: Add/remove monitoring without code changes to the trainer
3. **Timeout Prevention**: Send heartbeats during long validation to prevent external system timeouts
4. **Pluggable Architecture**: Switch between different monitoring systems by changing the entry script

## Example Scripts

- `tao_sft_example.py` - TAO-compatible SFT example with custom logger and hooks

For detailed implementation examples, see `custom_loggers_and_hooks.py`.
