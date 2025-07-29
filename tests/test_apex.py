import unittest


class TestApex(unittest.TestCase):
    def test_fused_adam(self):
        import apex
        import torch

        model = torch.nn.Linear(10, 10).cuda()
        opt = apex.optimizers.FusedAdam(model.parameters(), lr=0.001)

        input_data = torch.randn(5, 10).cuda()
        output = model(input_data)
        loss = output.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    # Run the tests
    unittest.main()
