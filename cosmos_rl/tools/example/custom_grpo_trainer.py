# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import GRPOTrainer, compute_loss
from typing import Optional, Dict, Any, List, Tuple
import torch
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.optm import (
    LRSchedulersContainer,
)
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger


"""
Following is an example of defining a custom GRPO trainer by extending the base GRPOTrainer.
You can override the methods in GRPOTrainer to customize the training behavior as needed.
For example, you can override the loss function, the logprob computation, the training step,
etc. Here we provide a simple example that reuses most of the existing logic in GRPOTrainer.
"""

"""
To customize the rollout generation process as well, please refer to `cosmos_rl/rollout/example_custom_rollout/example_custom_rollout.py`. 
In that file, we demonstrate how to implement a custom rollout engine and register it to the RolloutRegistry to be used in the rollout worker.
You can mimic that implementation to implement your own rollout engine to cusomize the rollout generation process.
Then combined with this custom GRPO trainer, you can have a complete custom GRPO training flow with both custom rollout and custom trainer.
"""

"""
Only by implementing this custom GRPO trainer and registering it with the TrainerRegistry with a unique trainer_type,
and implementing a custom rollout engine like `ExampleHFRollout` as mentioned above and registering it with the RolloutRegistry with a unique rollout_backend_type,
then a complete custom GRPO training flow with both custom rollout and custom trainer can be created with existing cosmos-rl framework logic.

The custom trainer_type and rollout_backend_type can then be specified in the cosmos config file like in `configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml`.
Then the custom GRPO training flow can be launched with the existing launcher logic in cosmos-rl using the specified config file.

All you need to do is:
1. Implement the custom GRPO trainer by extending GRPOTrainer as shown below and register it with a unique trainer_type.
2. Implement the custom rollout engine by extending RolloutBase as shown in `example_custom_rollout.py` and register it with a unique rollout_backend_type.
3. Specify the custom trainer_type and rollout_backend_type in the cosmos config file, like in `configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml`.
4. Launch the training flow with the existing launcher, eg. `cosmos-rl --config configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml tools/example/custom_grpo_trainer.py`.

Run `cosmos-rl --config configs/qwen2-5/qwen2-5-3b-p-fsdp2-colocated-grpo.toml tools/example/custom_grpo_trainer.py`, then the custom GRPO training flow with both custom rollout and custom trainer will be launched.
"""


@TrainerRegistry.register(trainer_type="custom_grpo_example")
class CustomGRPOTrainer(GRPOTrainer):
    """
    Custom GRPO Trainer class that extends the base GRPOTrainer.
    This class can be customized further as needed.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Custom GRPO Trainer.
        Args:
            *args: Positional arguments for the base GRPOTrainer.
            **kwargs: Keyword arguments for the base GRPOTrainer.
        """
        super().__init__(*args, **kwargs)

    def loss_fn(
        self,
        current_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
        old_per_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
        ref_per_token_logps: Optional[
            torch.Tensor
        ],  # per-token logprobs of shape `[n_tokens_of_logprobs]`
        current_advantages: torch.Tensor,  # of shape `[batch_size, max_len]`
        cu_seqlens: torch.Tensor,  # of shape `[batch_size + 1]`
        config: CosmosConfig,
        logprob_masks: torch.Tensor,  # of shape `[batch_size, max_len]`
        dp_group: Optional[torch.distributed.ProcessGroup] = None,
        ddp_comm: HighAvailabilitylNccl = None,
        rollout_per_token_logps: Optional[
            List[List[float]]
        ] = None,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the GRPO loss. This method can be overridden for custom loss computation.
        Here we simply reuse the existing compute_loss function in GRPOTrainer for example.
        Returns:
            A tuple of (loss, per_token_loss, kl_loss)
            The first element is the scalar loss tensor which will be used for backpropagation.
            The second element is the per-token loss tensor of shape `[n_tokens_of_logprobs]`.
            The third element is the KL divergence loss tensor.
            The second and third loss basically serve for logging purpose.
        """
        logger.debug("Using custom GRPO loss function in CustomGRPOTrainer.")
        return compute_loss(
            current_token_logps,
            old_per_token_logps,
            ref_per_token_logps,
            current_advantages,
            cu_seqlens,
            config,
            logprob_masks,
            dp_group,
            ddp_comm,
            rollout_per_token_logps,
            **kwargs,
        )

    def compute_logprobs(
        self,
        minibatch: Dict[str, Any],
        logits: torch.Tensor,
        is_full_logits: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the per-token log probabilities and advantages for the given minibatch.
        This method can be overridden for custom logprob computation.
        Here we simply call the superclass method for example.
        Args:
            minibatch: a dictionary containing the input_ids and logprob_masks
            logits: the logits of the model
            is_full_logits: whether the logits are full logits or have been index-selected for memory efficiency
        Returns:
            logps: the per-token log probabilities for the minibatch
            logprob_masks: the logprob_masks for the minibatch
            metrics: a dict of collected metrics, e.g. entropy
        """
        logger.debug("Using custom compute_logprobs in CustomGRPOTrainer.")
        return super().compute_logprobs(
            minibatch,
            logits,
            is_full_logits=is_full_logits,
            **kwargs,
        )

    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Perform a single training step using the provided rollouts.
        This method can be overridden for custom training step logic.
        Here we simply call the superclass method for example.
        Args:
            rollouts: A list of Rollout objects containing the training data or unique identifiers for the training data.
            current_step: The current training step.
            total_steps: The total number of training steps.
            remain_samples_num: The number of remaining rollout generated samples to process in the whole training.
            inter_policy_nccl: The NCCL communicator for inter-policy communication.
            is_master_replica: Whether this replica is the master replica.
            do_save_checkpoint: Whether to save a checkpoint after this step.
        Returns:
            A dictionary of training metrics used for logging and reporting.
        """
        logger.debug(f"Starting training step {current_step}/{total_steps}")
        return super().step_training(
            rollouts,
            current_step,
            total_steps,
            remain_samples_num,
            inter_policy_nccl,
            is_master_replica,
            do_save_checkpoint=do_save_checkpoint,
            **kwargs,
        )

    def build_lr_schedulers(self) -> LRSchedulersContainer:
        """
        Build the lr schedulers for the trainer.
        This method can be overridden for custom lr scheduler building.
        Here we simply call the superclass method for example.
        Returns:
            LRSchedulersContainer: The container holding the learning rate schedulers.
        """
        logger.debug("Building LR schedulers using CustomGRPOTrainer.")
        return super().build_lr_schedulers()

    """
    If needed, you can add more custom methods or override existing methods from GRPOTrainer here.
    The basic methods need custom and override usually include:
        build_optimizers, 
        build_lr_schedulers, step_training, export_safetensors,
        step_training, 
        export_safetensors,
        model_load_from_hf, 
        model_resume_from_checkpoint, 
        loss_fn, 
        compute_logprobs.
    """


from cosmos_rl.tools.dataset.gsm8k_grpo import (
    GSM8kDataset,
    GSM8kValDataset,
    GSM8kDataPacker,
    custom_reward_fn,
)
from cosmos_rl.launcher.worker_entry import main as launch_worker


if __name__ == "__main__":
    """
    Example of launching a custom GRPO training flow using the above registered CustomGRPOTrainer.
    This example reuses the launching logic from tools/dataset/gsm8k_grpo.py.
    """
    # It is best practice to pass the dataset as a factory function
    launch_worker(
        dataset=GSM8kDataset(),
        val_dataset=GSM8kValDataset(),
        reward_fns=[custom_reward_fn],
        data_packer=GSM8kDataPacker(),
        val_data_packer=GSM8kDataPacker(),
    )
