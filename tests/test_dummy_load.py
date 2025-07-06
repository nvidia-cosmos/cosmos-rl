import os
import unittest
import time
import types
import torch
import torch.nn as nn
import vllm
from vllm.device_allocator.cumem import CuMemAllocator
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm import LLM, SamplingParams
from vllm.utils import DeviceMemoryProfiler

if vllm.__version__ >= "0.9.0":
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
else:
    from vllm.model_executor.model_loader.loader import (
        _process_weights_after_loading as process_weights_after_loading,
    )
    from vllm.model_executor.model_loader.loader import DefaultModelLoader
from vllm.utils import GiB_bytes


class CustomModelLoader(DefaultModelLoader):
    def reload_weights(self, vllm_config: VllmConfig, model: nn.Module) -> None:
        device_config = vllm_config.device_config
        model_config = vllm_config.model_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self.get_all_weights(model_config, model)
            )
            self.counter_after_loading_weights = time.perf_counter()
            print(
                f"Loading weights took {self.counter_after_loading_weights - self.counter_before_loading_weights:.2f} seconds",
            )
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}"
                    )

            process_weights_after_loading(model, model_config, target_device)


def patch_vllm_model_to_reload_weight(llm: LLM):
    def add_method(obj, method, method_name):
        setattr(obj, method_name, types.MethodType(method, obj))

    def reload_model_worker(self) -> None:
        if self.vllm_config.model_config.enable_sleep_mode:
            allocator = CuMemAllocator.get_instance()
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be " "used for one instance per process."
            )
            context = allocator.use_memory_pool(tag="weights")
        else:
            from contextlib import nullcontext

            context = nullcontext()
        with context:
            self.model_runner.reload_model()

    add_method(
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker,
        reload_model_worker,
        "reload_model",
    )

    def reload_model_runner(self) -> None:
        print("Starting to realod weight for current vllm model")
        with DeviceMemoryProfiler(self.device) as m:
            time_before_load = time.perf_counter()
            loader = CustomModelLoader(self.vllm_config.load_config)
            if not hasattr(loader, "reload_weights"):
                raise ValueError("Model loader does not support reloading weights")
            loader.reload_weights(self.vllm_config, self.model)
            time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        print(
            f"Model weight reloading took {self.model_memory_usage / GiB_bytes:.4f} GiB and {time_after_load - time_before_load:.6f} seconds",
            self.model_memory_usage / GiB_bytes,
            time_after_load - time_before_load,
        )

    add_method(
        llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner,
        reload_model_runner,
        "reload_model",
    )


class TestDummyLoad(unittest.TestCase):
    model_id_list = [
        "google/gemma-3-1b-pt",
        "Qwen/Qwen2.5-3B-Instruct",
    ]
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    def test_dummy_load(self):
        prompts = [
            "The problem involves a chess tournament with 12 participants where each participant's victory lists expand through transitive closure, and each twelfth list includes someone not in the eleventh list. The key steps to determine the number of drawn games are as follows:\n\n1. **Understanding the Lists**: Each participant's lists grow by including those they defeated and the transitive closure of those defeats. The twelfth list must include someone new compared to the eleventh, indicating the transitive closure takes exactly 12 steps.\n\n2. **Cycle Structure**: To satisfy the condition, the tournament must be structured such that each participant's victory chain forms a cycle. For example, participant 1 beats 2, 2 beats 3, ..., 12 beats 1. This cycle ensures each list grows by one participant per step.\n\n3. **Transitive Closure**: In a 12-player cycle, each participant's lists grow incrementally, adding one new participant each step. The twelfth list includes all participants, while the eleventh list misses one, satisfying the problem's condition.\n\n4. **Drawn Games Calculation**: \n   - Total games in the tournament: \\( \\binom{12}{2} = 66 \\).\n   - Decisive games (cycle): 12 (each participant beats exactly one other).\n   - Drawn games: Total games - Decisive games = \\( 66 - 12 = 54 \\).\n\nThus, the number of drawn games played in the tournament is \\(\\boxed{54}\\).",
        ]
        sampling_params = SamplingParams(
            seed=42, temperature=0.8, top_p=0.95, max_tokens=1024
        )
        for model_id in self.model_id_list:
            if model_id == "google/gemma-3-1b-pt" and vllm.__version__ <= "0.9.1":
                # In vLLM 0.9.1 and earlier, Gemma models do not set the normalizer buffer as non-persistent.
                # This causes incorrect results in dummy load mode. Skip this test for Gemma models.
                continue

            audo_load_llm = LLM(
                model=model_id,
                enable_sleep_mode=False,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                seed=42,
                enforce_eager=True,
                distributed_executor_backend="external_launcher",
                disable_custom_all_reduce=True,
                enable_prefix_caching=False,
                disable_log_stats=True,
                skip_tokenizer_init=False,
            )
            results = audo_load_llm.generate(prompts, sampling_params)
            audo_load_results = results[0].outputs[0].text
            del audo_load_llm

            dummy_load_llm = LLM(
                model=model_id,
                enable_sleep_mode=False,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                load_format="dummy",
                seed=42,
                enforce_eager=True,
                distributed_executor_backend="external_launcher",
                disable_custom_all_reduce=True,
                enable_prefix_caching=False,
                disable_log_stats=True,
                skip_tokenizer_init=False,
            )
            patch_vllm_model_to_reload_weight(dummy_load_llm)
            dummy_load_llm.llm_engine.vllm_config.load_config.load_format = "auto"
            dummy_load_llm.collective_rpc("reload_model")
            results = dummy_load_llm.generate(prompts, sampling_params)
            dummy_load_results = results[0].outputs[0].text
            del dummy_load_llm

            self.assertEqual(dummy_load_results, audo_load_results)


if __name__ == "__main__":
    unittest.main()
