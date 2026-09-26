import unittest
from dataclasses import replace

import torch

from parrot import ModelConfig, Parrot


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0),
    "requires a Hopper or newer CUDA GPU",
)
class CudaTrainingTests(unittest.TestCase):
    def test_grouped_experts_match_reference_loss_and_gradients(self):
        cfg = ModelConfig(
            vocab_size=32, dim=32, n_layers=2, n_heads=2, n_kv_heads=1,
            head_dim=16, kv_latent_dim=16, n_experts=4, expert_dim=32,
            shared_expert_dim=16, max_seq_len=16, window_size=4,
            moe_backend="pytorch",
        )
        torch.manual_seed(23)
        reference = Parrot(cfg).to(device="cuda", dtype=torch.bfloat16)
        grouped = Parrot(replace(cfg, moe_backend="training_cuda")).to(
            device="cuda", dtype=torch.bfloat16
        )
        grouped.load_state_dict(reference.state_dict(), strict=False)
        with torch.no_grad():
            for reference_block, grouped_block in zip(
                reference.blocks, grouped.blocks
            ):
                experts = reference_block.moe.experts
                grouped_block.moe.expert_gate.copy_(torch.stack([
                    expert.gate.weight.T for expert in experts
                ]))
                grouped_block.moe.expert_up.copy_(torch.stack([
                    expert.up.weight.T for expert in experts
                ]))
                grouped_block.moe.expert_down.copy_(torch.stack([
                    expert.down.weight.T for expert in experts
                ]))

        ids = torch.randint(cfg.vocab_size, (4, 32), device="cuda")
        expected = reference(
            ids, labels=ids, bag_size=4, supervised_logits_only=True
        )
        actual = grouped(
            ids, labels=ids, bag_size=4, supervised_logits_only=True
        )
        expected.loss.backward()
        actual.loss.backward()
        torch.testing.assert_close(actual.loss, expected.loss, atol=2e-2, rtol=2e-2)
        for reference_block, grouped_block in zip(
            reference.blocks, grouped.blocks
        ):
            expected_gate_grad = torch.stack([
                expert.gate.weight.grad.T
                for expert in reference_block.moe.experts
            ])
            torch.testing.assert_close(
                grouped_block.moe.expert_gate.grad,
                expected_gate_grad,
                atol=3e-2,
                rtol=3e-2,
            )
