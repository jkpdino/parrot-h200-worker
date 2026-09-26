"""Autograd-enabled fused Metal SwiGLU; GEMMs use PyTorch autograd."""
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

_LIBRARIES = {}


def _library(dtype):
    if dtype not in _LIBRARIES:
        scalar = {torch.float32: "float", torch.bfloat16: "bfloat", torch.float16: "half"}[dtype]
        source = Path(__file__).with_name("kernels").joinpath("training.metal").read_text()
        source = source.replace("device float", f"device {scalar}").replace(
            "device const float", f"device const {scalar}")
        source = source.replace("STORAGE", scalar).replace("SCORE", "float")
        _LIBRARIES[dtype] = torch.mps.compile_shader(source)
    return _LIBRARIES[dtype]


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        if gate.device.type != "mps" or gate.dtype not in (
            torch.float32, torch.bfloat16, torch.float16
        ):
            raise ValueError("Metal training requires MPS float32, bfloat16, or float16")
        gate, up = gate.contiguous(), up.contiguous()
        out = torch.empty_like(gate)
        _library(gate.dtype).swiglu_forward(out, gate, up, threads=gate.numel())
        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        dg, du = torch.empty_like(gate), torch.empty_like(up)
        _library(gate.dtype).swiglu_backward(
            dg, du, grad.contiguous(), gate, up, threads=gate.numel())
        return dg, du


def swiglu(gate, up):
    return _SwiGLU.apply(gate, up)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, eps):
        value = value.contiguous()
        dim = value.shape[-1]
        rows = value.numel() // dim
        output = torch.empty_like(value)
        inverse_rms = torch.empty(rows, device=value.device, dtype=torch.float32)
        _library(value.dtype).rms_training_forward(
            output, inverse_rms, value, weight, rows, dim, eps,
            threads=(rows * 256, 1, 1), group_size=(256, 1, 1),
        )
        ctx.save_for_backward(value, weight, inverse_rms)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, weight, inverse_rms = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dim = value.shape[-1]
        rows = value.numel() // dim
        grad_value = torch.empty_like(value)
        grad_weight = torch.empty_like(weight)
        lib = _library(value.dtype)
        lib.rms_training_backward_input(
            grad_value, grad_output, value, weight, inverse_rms, rows, dim,
            threads=(rows * 256, 1, 1), group_size=(256, 1, 1),
        )
        lib.rms_training_grad_weight(
            grad_weight, grad_output, value, inverse_rms,
            rows, dim, threads=(dim * 256, 1, 1), group_size=(256, 1, 1),
        )
        return grad_value, grad_weight, None


def rms_norm(value, weight, eps):
    return _RMSNorm.apply(value, weight, eps)


class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        batches, logit_positions, vocab = logits.shape
        target_positions = targets.shape[1]
        if target_positions not in (logit_positions, logit_positions - 1):
            raise ValueError("logits must contain the target positions and at most one extra position")
        bag_size = targets.shape[-1]
        rows = batches * target_positions
        row_loss = torch.empty(rows, device=logits.device, dtype=torch.float32)
        row_maximum = torch.empty_like(row_loss)
        row_inverse_sum = torch.empty_like(row_loss)
        row_valid = torch.empty(rows, device=logits.device, dtype=torch.int32)
        _library(logits.dtype).cross_entropy_forward(
            row_loss, row_maximum, row_inverse_sum, row_valid, logits, targets,
            batches, logit_positions, target_positions, vocab, bag_size,
            threads=(rows * 256, 1, 1), group_size=(256, 1, 1),
        )
        denominator = row_valid.sum().clamp_min(1).float()
        ctx.save_for_backward(
            logits, targets, row_maximum, row_inverse_sum, row_valid, denominator
        )
        return row_loss.sum() / denominator

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss):
        logits, targets, row_maximum, row_inverse_sum, row_valid, denominator = ctx.saved_tensors
        batches, logit_positions, vocab = logits.shape
        target_positions = targets.shape[1]
        bag_size = targets.shape[-1]
        rows = batches * target_positions
        grad_logits = torch.zeros_like(logits)
        _library(logits.dtype).cross_entropy_backward(
            grad_logits, logits, targets, row_maximum, row_inverse_sum,
            row_valid, denominator, grad_loss.contiguous(),
            batches, logit_positions, target_positions, vocab, bag_size,
            threads=rows * vocab,
        )
        return grad_logits, None


def cross_entropy(logits, targets):
    return _CrossEntropy.apply(logits, targets)


class _DepthMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, norm_weight, eps, *sources):
        values = torch.stack(sources)
        source_count, *shape, dim = values.shape
        tokens = values.numel() // (source_count * dim)
        scores = torch.empty(source_count, tokens, device=values.device, dtype=torch.float32)
        inverse_rms = torch.empty_like(scores)
        weights = torch.empty_like(scores)
        output = torch.empty_like(values[0])
        lib = _library(values.dtype)
        lib.depth_training_scores(
            scores, inverse_rms, values, query, norm_weight,
            source_count, tokens, dim, eps,
            threads=(source_count * tokens * 256, 1, 1),
            group_size=(256, 1, 1),
        )
        lib.depth_training_softmax(
            weights, scores, source_count, tokens, threads=tokens,
        )
        lib.depth_training_mix(
            output, weights, values,
            source_count, tokens, dim, threads=tokens * dim,
        )
        ctx.save_for_backward(
            values, output, scores, inverse_rms, weights, query, norm_weight
        )
        ctx.source_count = source_count
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        values, output, scores, inverse_rms, weights, query, norm_weight = ctx.saved_tensors
        source_count = ctx.source_count
        dim = values.shape[-1]
        tokens = output.numel() // dim
        grad_output = grad_output.contiguous()
        grad_scores = torch.empty_like(scores)
        grad_values = torch.empty_like(values)
        grad_query = torch.empty_like(query)
        grad_norm_weight = torch.empty_like(norm_weight)
        lib = _library(values.dtype)
        lib.depth_training_grad_scores(
            grad_scores, grad_output, values, output, weights,
            source_count, tokens, dim,
            threads=(source_count * tokens * 256, 1, 1),
            group_size=(256, 1, 1),
        )
        lib.depth_training_grad_values(
            grad_values, grad_output, values, scores, inverse_rms, weights,
            grad_scores, query, norm_weight, source_count, tokens, dim,
            threads=source_count * tokens * dim,
        )
        lib.depth_training_grad_parameters(
            grad_query, grad_norm_weight, values, inverse_rms, grad_scores,
            query, norm_weight, source_count, tokens, dim,
            threads=(dim * 256, 1, 1), group_size=(256, 1, 1),
        )
        return (grad_query, grad_norm_weight, None, *grad_values.unbind(0))


def depth_mix(module, sources):
    return _DepthMix.apply(module.query, module.norm.weight, module.norm.eps, *sources)


class _GroupScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, flat, indices, positions, capacity, n_experts):
        tokens, dim = flat.shape
        top_k = indices.shape[1]
        grouped = flat.new_zeros(n_experts, capacity, dim)
        ctx.save_for_backward(indices, positions)
        ctx.shape = (tokens, dim, top_k, capacity)
        _library(flat.dtype).group_scatter(
            grouped, flat, indices, positions, tokens, dim, top_k, capacity,
            threads=tokens * top_k * dim,
        )
        return grouped

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grouped):
        indices, positions = ctx.saved_tensors
        tokens, dim, top_k, capacity = ctx.shape
        grad_flat = grad_grouped.new_empty(tokens, dim)
        _library(grad_grouped.dtype).group_scatter_backward(
            grad_flat, grad_grouped.contiguous(), indices, positions,
            tokens, dim, top_k, capacity, threads=tokens * dim,
        )
        return grad_flat, None, None, None, None


class _GroupGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, expert_output, indices, positions, weights):
        experts, capacity, dim = expert_output.shape
        tokens, top_k = indices.shape
        output = expert_output.new_empty(tokens, dim)
        _library(expert_output.dtype).group_gather(
            output, expert_output, indices, positions, weights,
            tokens, dim, top_k, capacity, threads=tokens * dim,
        )
        ctx.save_for_backward(expert_output, indices, positions, weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        expert_output, indices, positions, weights = ctx.saved_tensors
        experts, capacity, dim = expert_output.shape
        tokens, top_k = indices.shape
        grad_output = grad_output.contiguous()
        grad_expert = torch.zeros_like(expert_output)
        grad_weights = torch.empty_like(weights)
        lib = _library(expert_output.dtype)
        lib.group_gather_backward_output(
            grad_expert, grad_output, indices, positions, weights,
            tokens, dim, top_k, capacity, threads=tokens * top_k * dim,
        )
        lib.group_gather_backward_weight(
            grad_weights, grad_output, expert_output, indices, positions,
            tokens * top_k, dim, top_k, capacity,
            threads=(tokens * top_k * 256, 1, 1), group_size=(256, 1, 1),
        )
        return grad_expert, None, None, grad_weights


def grouped_experts(module, flat, indices, weights):
    """Run sparse experts as three batched GEMMs, retaining exact overflow."""
    cfg = module.cfg
    tokens = flat.shape[0]
    assignments = tokens * cfg.top_k
    expert_ids = indices.reshape(-1)
    positions = torch.empty(assignments, device=flat.device, dtype=torch.int32)
    counts = torch.zeros(cfg.n_experts, device=flat.device, dtype=torch.int32)
    _library(flat.dtype).assignment_positions(
        positions, counts, expert_ids, assignments, threads=assignments
    )
    # A single synchronized scalar read replaces the former per-expert overflow
    # branch. Every assignment then uses the same batched, differentiable path.
    capacity = max(8, (int(counts.max().item()) + 7) // 8 * 8)
    grouped = _GroupScatter.apply(flat, indices, positions, capacity, cfg.n_experts)

    hidden = swiglu(
        torch.bmm(grouped, module.expert_gate.transpose(1, 2)),
        torch.bmm(grouped, module.expert_up.transpose(1, 2)),
    )
    expert_output = torch.bmm(hidden, module.expert_down.transpose(1, 2))
    routed = _GroupGather.apply(
        expert_output, indices, positions, weights.to(flat.dtype)
    )
    return routed + module.shared(flat)
