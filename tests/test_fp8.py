import unittest

from torch import nn
from cosmos_rl.utils.fp8.fp8_util import (
    FP8ModelConverter,
    IS_TORCH_COMPATIBLE_WITH_FP8,
)
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.config import FP8Config
from cosmos_rl.utils.parallelism import ParallelDims

from torchao.float8.float8_linear import Float8Linear


class DemoModel(nn.Module):
    def __init__(self, input_dim, output_dim, intermediate_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, x):
        return self.linear2(self.linear(self.norm(x)))

    def fqn_filter_for_fp8(self):
        return []


@unittest.skipIf(
    not IS_TORCH_COMPATIBLE_WITH_FP8, "FP8 is not supported for this version of PyTorch"
)
class TestFp8(unittest.TestCase):
    def _test_fp8_model_converter_delayed_scaling(self, quant_recipe, dp_shard):
        demo_model = DemoModel(input_dim=1024, output_dim=512, intermediate_dim=1024)
        config = CosmosConfig()
        fp8_config = FP8Config(
            enable_fp8=True, quant_recipe=quant_recipe, fp8_recipe="delayed_scaling"
        )
        config.train.fp8 = fp8_config

        # Mock the parallel_dims
        tp_size = 2
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=dp_shard,
            cp=1,
            tp=tp_size,
            pp=1,
            world_size=dp_shard * tp_size,
            pp_dynamic_shape=False,
        )
        converter = FP8ModelConverter(config, parallel_dims)
        converter.convert_model(demo_model)

    def test_fp8_model_converter_delayed_scaling(self):
        params = [
            ("rowwise", 1),
            ("tensorwise", 1),
            ("rowwise", 2),
            ("tensorwise", 2),
        ]
        for quant_recipe, dp_shard in params:
            with self.subTest(quant_recipe=quant_recipe, dp_shard=dp_shard):
                # right now expect NotImplementedError for delayed scaling
                with self.assertRaises(NotImplementedError):
                    self._test_fp8_model_converter_delayed_scaling(
                        quant_recipe=quant_recipe, dp_shard=dp_shard
                    )

    def test_fp8_model_converter(self):
        params = [
            ("rowwise", "dynamic_scaling", 1),
            ("tensorwise", "dynamic_scaling", 1),
            ("rowwise", "dynamic_scaling", 2),
            ("tensorwise", "dynamic_scaling", 2),
        ]
        for quant_recipe, fp8_recipe, dp_shard in params:
            with self.subTest(
                quant_recipe=quant_recipe, fp8_recipe=fp8_recipe, dp_shard=dp_shard
            ):
                self._test_fp8_model_converter(
                    quant_recipe=quant_recipe,
                    fp8_recipe=fp8_recipe,
                    dp_shard=dp_shard,
                )

    def _test_fp8_model_converter(self, quant_recipe, fp8_recipe, dp_shard):
        demo_model = DemoModel(input_dim=1024, output_dim=512, intermediate_dim=1024)
        config = CosmosConfig()
        fp8_config = FP8Config(
            enable_fp8=True, quant_recipe=quant_recipe, fp8_recipe=fp8_recipe
        )
        config.train.fp8 = fp8_config

        # Mock the parallel_dims
        tp_size = 2
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=dp_shard,
            cp=1,
            tp=tp_size,
            pp=1,
            world_size=dp_shard * tp_size,
            pp_dynamic_shape=False,
        )
        config.train.fp8.quant_recipe = quant_recipe
        config.train.fp8.fp8_recipe = fp8_recipe

        converter = FP8ModelConverter(config, parallel_dims)
        converter.convert_model(demo_model)

        # Check the model is converted
        assert isinstance(
            demo_model.linear, Float8Linear
        ), f"Got type: {type(demo_model.linear)}"
        assert isinstance(
            demo_model.linear2, Float8Linear
        ), f"Got type: {type(demo_model.linear2)}"


if __name__ == "__main__":
    unittest.main()
