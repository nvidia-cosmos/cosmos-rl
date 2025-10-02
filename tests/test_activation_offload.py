import torch
import unittest

from cosmos_rl.utils.activation_offloading import get_act_offloading_ctx_manager


class DemoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_0 = torch.nn.Linear(2048, 2048)
        self.relu = torch.nn.ReLU()
        self.linear_1 = torch.nn.Linear(2048, 1024)

    def forward(self, x):
        return self.linear_1(self.relu(self.linear_0(x)))


class TestActivationOffload(unittest.TestCase):
    def _test_activation_offload(self, async_offload=False):
        device = torch.device("cuda")
        with torch.device(device):
            torch.manual_seed(2025)
            model = DemoModel()

            torch.manual_seed(2025)

            ref_model = DemoModel()

        input_data = torch.randn(1024, 2048).to(device)

        # With activation offload
        with get_act_offloading_ctx_manager(model, True, use_streams=async_offload):
            output = model(input_data)

        result = output.sum()
        result.backward()

        # Without activation offload
        with get_act_offloading_ctx_manager(model, False, use_streams=async_offload):
            output_ref = ref_model(input_data)

        result_ref = output_ref.sum()
        result_ref.backward()

        for param, param_ref in zip(model.parameters(), ref_model.parameters()):
            assert param.data_ptr() != param_ref.data_ptr()
            assert torch.equal(param.grad, param_ref.grad)

    def test_activation_offload_sync(self):
        self._test_activation_offload(async_offload=False)

    def test_activation_offload_async(self):
        self._test_activation_offload(async_offload=True)


if __name__ == "__main__":
    unittest.main()
