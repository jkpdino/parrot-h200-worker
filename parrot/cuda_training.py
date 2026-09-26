"""CUDA training operators for Parrot's sparse experts."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def grouped_experts(module, flat, indices, weights):
    """Evaluate routed experts with grouped GEMMs and a non-atomic combine."""
    tokens, dim = flat.shape
    top_k = indices.shape[1]
    expert_ids = indices.reshape(-1)
    token_ids = torch.arange(tokens, device=flat.device).repeat_interleave(top_k)
    order = expert_ids.argsort()
    sorted_experts = expert_ids[order]
    sorted_tokens = token_ids[order]
    grouped_input = flat[sorted_tokens].contiguous()
    counts = torch.zeros(
        module.cfg.n_experts, device=flat.device, dtype=torch.int32
    )
    counts.scatter_add_(
        0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.int32)
    )
    offsets = counts.cumsum(0).to(torch.int32)

    gate, up = torch._grouped_mm(
        grouped_input, module.expert_gate_up, offs=offsets
    ).chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    routed = torch._grouped_mm(hidden, module.expert_down, offs=offsets)
    routed = routed * weights.reshape(-1)[order, None].to(routed.dtype)

    # Every sorted row maps to a unique token/slot assignment. Restore that
    # order with a one-to-one scatter, then reduce the two routes without the
    # expensive atomic index_add used by the original implementation.
    assignment_order = torch.empty_like(routed).index_copy(0, order, routed)
    combined = assignment_order.view(tokens, top_k, dim).sum(1)
    return module.shared(flat) + combined
