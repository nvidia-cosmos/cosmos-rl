Customization
================================

Customized dataset
-------------------

A `torch.utils.data.Dataset` object with `query_reference_answer` method is all you need:

.. code-block:: python

    class CustomDataset(torch.utils.data.Dataset):
        def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
            pass

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        def query_reference_answer(self, idx) -> Optional[Any]:
            """
            Query the reference answer for reward computation.
            """
            return self.data[idx]["reference_answer"]


Here we attach the `BytedTsinghua-SIA/DAPO-Math-17k <https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k>`_ dataset as an example:

.. code-block:: python

    from typing import Optional, Any, List, Dict
    from torch.utils.data import Dataset
    from datasets import load_dataset
    from cosmos_rl.policy.config import Config
    from cosmos_rl.dispatcher.algo.reward import direct_math_reward_fn, overlong_reward_fn
    from transformers import AutoTokenizer
    from torch.utils.data import ConcatDataset

    class MathDapoDataset(Dataset):
        def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
            '''
            This method is optional and get called by launcher after being mounted
            `config`: config;
            `tokenizer`: tokenizer;
            '''
            self.config = config
            self.tokenizer = tokenizer

            # This demo is only for DAPO-Math-17k dataset
            assert config.train.train_policy.dataset.name == "BytedTsinghua-SIA/DAPO-Math-17k"
            self.dataset = load_dataset(config.train.train_policy.dataset.name, config.train.train_policy.dataset.subset)

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx: int) -> tuple[str, str]:
            '''
            For DecoderOnlyLLMDataPacker, it should either return:
            - raw text prompt to be converted into input_ids by both rollout and policy models;
            - conversation format:
            ```
            [
                {
                    "role": "system",
                    "content": "You are an AI math expert, you will be given a question and required to answer. "
                }
                ...
            ]
            ```
            '''
            assert hasattr(self, "tokenizer"), "`self.tokenizer` should be set by the launcher"
            conversation = self.dataset[idx]["prompt"]
            assert isinstance(
                conversation, list
            ), f"Prompt should be a string, but got {type(conversation)}， {conversation}"
            
            # Apply chat-template to get the raw prompt in text
            return self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

        def get_reference_answer(self, idx: int) -> Any:
            '''
            This is mandatory for GRPO to get a reference answer for reward computation.
            '''
            return self.dataset[idx]["reward_model"]["ground_truth"]

In this example, the dataset fetch each row and return the raw text prompt by applying chat-template. This demonstrates how user can customize the dataset to fit their needs.

.. note::
    It is assumed here that decoder only LLM data packer is used, so we must either return the raw text prompt or the conversation format.

How to tell the launcher to use your customized dataset?
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Since we have already defined our customized dataset in previous step, we need to override the launcher entry point to pass the custom dataset.

Save this file to `./custom_entry.py`

.. code-block:: python

    from typing import Optional, Any, List, Dict
    from torch.utils.data import Dataset
    from datasets import load_dataset
    from cosmos_rl.launcher.worker_entry import main as launch_worker
    from cosmos_rl.policy.config import Config
    from cosmos_rl.dispatcher.algo.reward import direct_math_reward_fn, overlong_reward_fn
    from transformers import AutoTokenizer
    from torch.utils.data import ConcatDataset

    class MathDapoDataset(Dataset):
        ...

    if __name__ == "__main__":
        # `dataset` argument can be:
        #   - a dataset instance
        #   - a factory function that returns a dataset instance
        #
        # It is best practice to pass the dataset as a factory function
        # so that the dataset can be loaded on demand. (Not all workers need it)
        def get_dataset_factory(config: Config) -> Dataset:
            return MathDapoDataset()

        launch_worker(
            dataset=get_dataset_factory,
        )


 Append your customized launcher entry point to `cosmos-rl` command:

>>> cosmos-rl --config configs/qwen3/qwen3-8b-p-tp4-r-tp2-pp1-grpo.toml \
    --policy 1 \
    --rollout 2 \
    custom_entry.py

Check `./tools/dataset/ <#>`_ for more pre-defined customized datasets. 

Customized reward
-------------------

Similar to customized dataset, override the launcher entry point to pass the custom reward functions:

.. code-block:: python

    ...

    def custom_reward_fn(to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs) -> float:
        assert isinstance(reference, str), "Reference answer should be a string"
        # For demo purpose, we just return a random float
        # In practice, you should implement your own reward function
        return random.random()

    if __name__ == "__main__":
        launch_worker(
            #...
            reward_fns=[custom_reward_fn],
            #...
        )

- ``to_be_evaluated``: rollout generation
- ``reference``: reference answer from the dataset interface

Final reward will be the sum of returned float values from all reward functions.

Customized Data Packer
----------------------
Check `decoder_only_llm_packer.py <#>`_ to see how to implement a customized data packer for your own model.

Here we just reuse the pre-deined LLM data packer to demonstrate how to pass your data packer.

.. code-block:: python

    from typing import Optional, Any, List, Dict
    from torch.utils.data import Dataset
    from datasets import load_dataset
    from cosmos_rl.launcher.worker_entry import main as launch_worker
    from cosmos_rl.policy.config import Config
    from cosmos_rl.dispatcher.algo.reward import gsm8k_reward_fn
    from transformers import AutoTokenizer
    from cosmos_rl.dispatcher.data.packer import DataPacker, DecoderOnlyLLMDataPacker
    from cosmos_rl.utils.modelscope import modelscope_load_dataset

    class GSM8kDataset(Dataset):
        def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
            '''
            This method is optional and get called by launcher after being mounted
            `config`: config;
            `tokenizer`: tokenizer;
            '''
            self.config = config
            self.tokenizer = tokenizer
            modelscope_dataset_if_enabled = modelscope_load_dataset('AI-ModelScope/gsm8k', subset_name='main', split='train')
            if modelscope_dataset_if_enabled is None:
                self.dataset = load_dataset("openai/gsm8k", "main", split="train")
            else:
                self.dataset = modelscope_dataset_if_enabled


        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx: int) -> tuple[str, str]:
            '''
            For DecoderOnlyLLMDataPacker, it should either return:
            - raw text prompt to be converted into input_ids by both rollout and policy models;
            - conversation format:
            ```
            [
                {
                    "role": "system",
                    "content": "You are an AI math expert, you will be given a question and required to answer. "
                }
                ...
            ]
            ```
            '''
            assert hasattr(self, "tokenizer"), "`self.tokenizer` should be set by the launcher"
            question = self.dataset[idx]["question"]
            assert isinstance(
                question, str
            ), f"Prompt should be a string, but got {type(question)}， {question}"
            # Convert to templated prompt
            conversation = [
                {
                    "role": "system",
                    "content": """You are an AI math expert, you will be given a question and required to answer.
    Final answer should be like
    ```
    #### [ANS]
    ``` where [ANS] is your answer"""
                },
                {
                    "role": "user",
                    "content": question,
                }
            ]
            prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt

        def get_reference_answer(self, idx: int) -> Any:
            '''
            This is mandatory for GRPO to get a reference answer for reward computation.
            '''
            return self.dataset[idx]["answer"]

    class DemoDataPacker(DataPacker):
        '''
        This is a demo data packer that wraps the underlying data packer of the selected model.
        This is meaningless for this example, but useful for explaining:
            - how dataset data is processed and collated into a mini-batch for rollout engine;
            - how rollout output is processed and collated into a mini-batch for policy model;
        '''
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Check source code of DecoderOnlyLLMDataPacker to see how it's implemented
            self.underlying_data_packer = DecoderOnlyLLMDataPacker()

        def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
            '''
            This method is optional and get called by launcher after being mounted
            `config`: config;
            `tokenizer`: tokenizer;
            '''
            super().setup(config, tokenizer, *args, **kwargs)
            self.underlying_data_packer.setup(config, tokenizer, *args, **kwargs)

        def get_rollout_input(self, item: Any) -> Any:
            '''
            Convert dataset item into what rollout engine (e.g. vllm) expects
            '''
            return self.underlying_data_packer.get_rollout_input(item)

        def rollout_collate_fn(self, items: List[Any]) -> Any:
            '''
            Collate the rollout inputs into a mini-batch for rollout engine
            '''
            return self.underlying_data_packer.rollout_collate_fn(items)

        def get_policy_input(self, item: Any, rollout_output: str) -> Any:
            '''
            Process samples & rollout output before collating them into a mini-batch
            '''
            return self.underlying_data_packer.get_policy_input(item, rollout_output)

        def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
            '''
            Compute the maximum sequence length of the mini-batch
            '''
            return self.underlying_data_packer.policy_compute_max_len(processed_samples)

        def policy_collate_fn(self, processed_samples: List[Any], computed_max_len: int) -> Dict[str, Any]:
            '''
            Collate the mini-batch into the kwargs required by the policy model
            '''
            return self.underlying_data_packer.policy_collate_fn(processed_samples, computed_max_len)

    if __name__ == "__main__":
        def get_dataset_factory(config: Config) -> Dataset:
            return GSM8kDataset()
 
        launch_worker(
            #...
            dataset=get_dataset_factory,
            data_packer=DemoDataPacker(),
            #...
        )


Customized Model
-----------------

To customize the model, one needs to implement:

- A new model class that inherits from `cosmos_rl.policy.model.base.BaseModel`
- A `WeightMapper` class that inherits from `cosmos_rl.policy.model.base.WeightMapper`
- A `DataPacker` class that inherits from `cosmos_rl.dispatcher.data.packer.DataPacker`

Let's take `deepseek_v3` as an example.

.. code-block:: python

    from cosmos_rl.launcher.worker_entry import main as launch_worker
    from deepseek_v3 import DeepseekV3MoEModel
    from deepseek_v3.weight_mapper import DeepseekV3MoEWeightMapper
    from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
        DecoderOnlyLLMDataPacker,
    )
    from cosmos_rl.policy.model.base import ModelRegistry

    if __name__ == "__main__":
        # Register the model into the registry
        ModelRegistry.register_model(
            # Model class to register
            DeepseekV3MoEModel,
            # Weight mapper for this model
            DeepseekV3MoEWeightMapper,
        )

        launch_worker()

First import the model class, weight mapper class, and data packer class from the external source code. 

Then register the model into the registry via `ModelRegistry.register_model`.

User can launch a external model job with: 

>>> cosmos-rl --config ./configs/deepseek-v3/moonlight-moe-13b-tp4-sft.toml 
    cosmos_rl.tools.model.moonlight_launcher


