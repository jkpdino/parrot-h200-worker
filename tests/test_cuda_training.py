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
                grouped_block.moe.expert_gate_up.copy_(torch.stack([
                    torch.cat((expert.gate.weight, expert.up.weight), dim=0).T
                    for expert in experts
                ]))
                grouped_block.moe.expert_down.copy_(torch.stack([
                    expert.down.weight.T for expert in experts
                ]))
                grouped_block.moe.shared.gate_up.weight.copy_(torch.cat((
                    reference_block.moe.shared.gate.weight,
                    reference_block.moe.shared.up.weight,
                )))
                grouped_block.moe.shared.down.weight.copy_(
                    reference_block.moe.shared.down.weight
                )

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
            expected_gate_up_grad = torch.stack([
                torch.cat((expert.gate.weight.grad, expert.up.weight.grad), dim=0)
                for expert in reference_block.moe.experts
            ])
            torch.testing.assert_close(
                grouped_block.moe.expert_gate_up.grad,
                expected_gate_up_grad.transpose(1, 2),
                atol=3e-2,
                rtol=3e-2,
            )
            expected_down_grad = torch.stack([
                expert.down.weight.grad
                for expert in reference_block.moe.experts
            ])
            torch.testing.assert_close(
                grouped_block.moe.expert_down.grad,
                expected_down_grad.transpose(1, 2),
                atol=3e-2,
                rtol=3e-2,
            )
            torch.testing.assert_close(
                grouped_block.moe.router.weight.grad,
                reference_block.moe.router.weight.grad,
                atol=3e-2,
                rtol=3e-2,
            )
        for expected, actual in zip(reference.depth_mix, grouped.depth_mix):
            torch.testing.assert_close(
                actual.query.grad, expected.query.grad, atol=3e-2, rtol=3e-2
            )
            torch.testing.assert_close(
                actual.norm.weight.grad,
                expected.norm.weight.grad,
                atol=3e-2,
                rtol=3e-2,
            )
