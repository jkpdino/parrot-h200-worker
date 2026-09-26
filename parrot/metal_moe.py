"""Direct PyTorch-MPS bridge for the inference-only Metal MoE kernels."""
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .model import MoE

_LIBRARIES = {}
_MAX_METAL_EXPERTS = 32


def _scratch(owner, name: str, shape, like: Tensor, dtype=None) -> Tensor:
    """Reuse inference scratch without registering it in model state."""
    dtype = like.dtype if dtype is None else dtype
    cached = owner.__dict__.get(name)
    if (cached is None or tuple(cached.shape) != tuple(shape) or
            cached.device != like.device or cached.dtype != dtype):
        cached = torch.empty(shape, device=like.device, dtype=dtype)
        owner.__dict__[name] = cached
    return cached


def _library(dtype: torch.dtype):
    if dtype not in _LIBRARIES:
        source = Path(__file__).with_name("kernels").joinpath("moe_bf16.metal").read_text()
        if dtype == torch.float16:
            source = source.replace("_bf16", "_fp16").replace("bfloat", "half")
        elif dtype != torch.bfloat16:
            raise RuntimeError("The Metal MoE backend requires MPS BF16 or FP16")
        _LIBRARIES[dtype] = torch.mps.compile_shader(source)
    return _LIBRARIES[dtype]


def metal_rms_norm(module, x: Tensor) -> Tensor:
    """Fused inference RMSNorm for contiguous MPS BF16/FP16 tensors."""
    if not x.is_contiguous():
        x = x.contiguous()
    dim = x.shape[-1]
    rows = x.numel() // dim
    output = torch.empty_like(x)
    suffix = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    lib = _library(x.dtype)
    if rows <= 16:
        getattr(lib, f"rms_norm_decode_{suffix}")(
            output, x, module.weight, rows, dim, module.eps,
            threads=(rows * 256, 1, 1), group_size=(256, 1, 1),
        )
    else:
        getattr(lib, f"rms_norm_{suffix}")(
            output, x, module.weight, rows, dim, module.eps,
        )
    return output


def metal_route(module: "MoE", flat: Tensor, collect_stats: bool):
    """Fused FP32-softmax top-2 routing and expert-slot assignment."""
    tokens, dim = flat.shape
    experts = module.cfg.n_experts
    if experts > _MAX_METAL_EXPERTS:
        raise RuntimeError(
            f"The fused Metal router supports at most {_MAX_METAL_EXPERTS} experts"
        )
    probabilities = torch.empty(
        (tokens, experts) if collect_stats else (1,),
        device=flat.device, dtype=torch.float32,
    )
    weights = torch.empty((tokens, 2), device=flat.device, dtype=torch.float32)
    indices = torch.empty((tokens, 2), device=flat.device, dtype=torch.int64)
    positions = torch.empty(tokens * 2, device=flat.device, dtype=torch.int32)
    counts = torch.zeros(experts, device=flat.device, dtype=torch.int32)
    suffix = "bf16" if flat.dtype == torch.bfloat16 else "fp16"
    getattr(_library(flat.dtype), f"moe_route_{suffix}")(
        probabilities, weights, indices, positions, counts,
        flat, module.router.weight, tokens, dim, experts, int(collect_stats),
        threads=(tokens, 1, 1),
        group_size=(min(256, tokens), 1, 1),
    )
    return probabilities if collect_stats else None, weights, indices, positions, counts


def pack_weights(module: "MoE") -> tuple[Tensor, Tensor, Tensor]:
    """Pack routed experts once; shared weights remain in their native tensors."""
    cached = getattr(module, "_metal_weight_cache", None)
    if cached is not None and not torch.is_grad_enabled() and not module.training:
        return cached[1]
    version = tuple(
        (parameter._version, parameter.device, parameter.dtype, parameter.data_ptr())
        for expert in module.experts
        for parameter in (expert.gate.weight, expert.up.weight, expert.down.weight)
    )
    if cached is not None and cached[0] == version:
        return cached[1]
    packed = (
        torch.stack([expert.gate.weight.detach() for expert in module.experts]).contiguous(),
        torch.stack([expert.up.weight.detach() for expert in module.experts]).contiguous(),
        torch.stack([expert.down.weight.detach() for expert in module.experts]).contiguous(),
    )
    module._metal_weight_cache = (version, packed)
    return packed


def pack_combined_weights(module: "MoE") -> tuple[Tensor, Tensor, Tensor]:
    """Pack fused gate/up weights for batched MPS inference."""
    cached = getattr(module, "_metal_combined_cache", None)
    if cached is not None and not torch.is_grad_enabled() and not module.training:
        return cached[1]
    parameters = [
        parameter
        for expert in module.experts
        for parameter in (expert.gate.weight, expert.up.weight, expert.down.weight)
    ] + [module.shared.gate.weight, module.shared.up.weight]
    version = tuple(
        (parameter._version, parameter.device, parameter.dtype, parameter.data_ptr())
        for parameter in parameters
    )
    if cached is not None and cached[0] == version:
        return cached[1]
    packed = (
        torch.stack([
            torch.cat((expert.gate.weight.detach(), expert.up.weight.detach()), dim=0)
            for expert in module.experts
        ]).contiguous(),
        torch.stack([expert.down.weight.detach() for expert in module.experts]).contiguous(),
        torch.cat((module.shared.gate.weight.detach(),
                   module.shared.up.weight.detach()), dim=0).contiguous(),
    )
    module._metal_combined_cache = (version, packed)
    return packed


def _metal_swiglu(gate_up: Tensor, hidden_dim: int) -> Tensor:
    rows = gate_up.numel() // (2 * hidden_dim)
    hidden = torch.empty(rows * hidden_dim, device=gate_up.device, dtype=gate_up.dtype)
    lib = _library(gate_up.dtype)
    suffix = "bf16" if gate_up.dtype == torch.bfloat16 else "fp16"
    getattr(lib, f"swiglu_from_gate_up_{suffix}")(
        hidden, gate_up, rows, hidden_dim
    )
    return hidden.view(*gate_up.shape[:-1], hidden_dim)


def _fused_shared(module: "MoE", flat: Tensor, shared_gate_up: Tensor) -> Tensor:
    gate_up = torch.nn.functional.linear(flat, shared_gate_up)
    hidden = _metal_swiglu(gate_up, module.cfg.shared_expert_dim)
    return torch.nn.functional.linear(hidden, module.shared.down.weight)


def metal_experts(module: "MoE", flat: Tensor, indices: Tensor,
                  weights: Tensor) -> Tensor:
    """Execute selected and shared experts without leaving PyTorch's MPS stream."""
    if flat.device.type != "mps" or flat.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError("The Metal MoE backend requires MPS BF16 or FP16")
    if torch.is_grad_enabled():
        raise RuntimeError("The Metal MoE backend is inference-only")
    tokens, dim = flat.shape
    cfg = module.cfg
    routed_gate, routed_up, routed_down = pack_weights(module)
    hidden = torch.empty(
        (tokens, cfg.top_k + 1, cfg.expert_dim),
        device=flat.device,
        dtype=flat.dtype,
    )
    output = torch.empty_like(flat)
    lib = _library(flat.dtype)
    suffix = "bf16" if flat.dtype == torch.bfloat16 else "fp16"
    getattr(lib, f"moe_swiglu_hidden_{suffix}")(
        hidden, flat, indices, routed_gate, routed_up,
        module.shared.gate.weight, module.shared.up.weight,
        tokens, dim, cfg.expert_dim, cfg.shared_expert_dim,
    )
    getattr(lib, f"moe_down_accumulate_{suffix}")(
        output, hidden, indices, weights, routed_down,
        module.shared.down.weight,
        tokens, dim, cfg.expert_dim, cfg.shared_expert_dim,
    )
    return output


def metal_experts_decode(module: "MoE", flat: Tensor,
                         addend: Tensor | None = None,
                         defer_sum: bool = False) -> Tensor:
    """Route and execute three expert lanes in two Metal dispatches."""
    if flat.device.type != "mps" or flat.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError("The Metal MoE backend requires MPS BF16 or FP16")
    if torch.is_grad_enabled():
        raise RuntimeError("The Metal MoE backend is inference-only")
    cfg = module.cfg
    if cfg.top_k != 2 or cfg.n_experts > _MAX_METAL_EXPERTS:
        raise RuntimeError(
            "The fused decode kernel requires top-2 of at most "
            f"{_MAX_METAL_EXPERTS} experts"
        )
    if cfg.expert_dim > 512 or cfg.shared_expert_dim > 512:
        raise RuntimeError("The fused decode kernel supports expert widths up to 512")
    tokens, dim = flat.shape
    routed_gate_up, routed_down, shared_gate_up = pack_combined_weights(module)
    lanes = _scratch(
        module, "_metal_decode_lanes", (tokens, 3, dim), flat, torch.float32
    )
    output = _scratch(module, "_metal_decode_output", flat.shape, flat)
    suffix = "bf16" if flat.dtype == torch.bfloat16 else "fp16"
    lib = _library(flat.dtype)
    getattr(lib, f"moe_decode_lanes_{suffix}")(
        lanes, flat, module.router.weight,
        routed_gate_up, routed_down, shared_gate_up,
        module.shared.down.weight,
        tokens, dim, cfg.n_experts, cfg.expert_dim, cfg.shared_expert_dim,
        threads=(tokens * 3 * 768, 1, 1), group_size=(768, 1, 1),
    )
    if defer_sum:
        return lanes
    getattr(lib, f"moe_sum_decode_lanes_{suffix}")(
        output, lanes, flat if addend is None else addend.reshape_as(flat),
        tokens, dim, int(addend is not None),
        threads=(tokens * dim, 1, 1),
        group_size=(min(256, tokens * dim), 1, 1),
    )
    return output


def metal_experts_depth_decode(module: "MoE", sources: list[Tensor],
                               depth_module, output_norm,
                               addend: Tensor) -> Tensor:
    """Fuse Block AttnRes, RMSNorm, routing, and three expert lanes."""
    first = sources[0]
    tokens = first.numel() // first.shape[-1]
    dim = first.shape[-1]
    cfg = module.cfg
    flattened = [source.reshape(tokens, dim) for source in sources]
    padded = flattened + [flattened[0]] * (9 - len(flattened))
    routed_gate, routed_up, routed_down = pack_weights(module)
    lanes = torch.empty((tokens, 3, dim), device=first.device, dtype=torch.float32)
    output = torch.empty((tokens, dim), device=first.device, dtype=first.dtype)
    suffix = "bf16" if first.dtype == torch.bfloat16 else "fp16"
    lib = _library(first.dtype)
    getattr(lib, f"moe_depth_decode_lanes_{suffix}")(
        lanes, module.router.weight, routed_gate, routed_up, routed_down,
        module.shared.gate.weight, module.shared.up.weight,
        module.shared.down.weight, *padded,
        depth_module.query, depth_module.norm.weight, output_norm.weight,
        tokens, dim, cfg.n_experts, cfg.expert_dim, cfg.shared_expert_dim,
        len(sources), depth_module.norm.eps, output_norm.eps,
        threads=(tokens * 3 * 512, 1, 1), group_size=(512, 1, 1),
    )
    getattr(lib, f"moe_sum_decode_lanes_{suffix}")(
        output, lanes, addend.reshape(tokens, dim), tokens, dim, 1,
        threads=(tokens * dim, 1, 1),
        group_size=(min(256, tokens * dim), 1, 1),
    )
    return output.view_as(first)


def metal_local_attention_decode(module, x: Tensor, cache, position: int,
                                 cos: Tensor, sin: Tensor,
                                 addend: Tensor | None = None,
                                 defer_sum: bool = False) -> Tensor:
    """Fused local GQA decode followed by output projection."""
    cfg = module.cfg
    if cfg.head_dim != 96 or cfg.window_size > 2048:
        raise RuntimeError("The fused local attention kernel requires head_dim=96 and window<=2048")
    batch = x.shape[0]
    head_output = _scratch(
        module, "_metal_decode_heads", (batch, cfg.n_heads, cfg.dim), x,
        torch.float32,
    )
    output = _scratch(module, "_metal_decode_output", (batch, cfg.dim), x)
    suffix = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    lib = _library(x.dtype)
    window_start = max(0, position - cfg.window_size + 1)
    getattr(lib, f"local_attention_decode_{suffix}")(
        head_output, x, module.q.weight, module.k.weight, module.v.weight,
        cache.key, cache.value, cos, sin,
        batch, position, window_start, cfg.dim, cfg.n_heads, cfg.n_kv_heads,
        cfg.head_dim, cfg.max_seq_len, cfg.head_dim ** -0.5, module.out.weight,
        threads=(batch * cfg.n_heads * 256, 1, 1),
        group_size=(256, 1, 1),
    )
    if defer_sum:
        return head_output
    getattr(lib, f"attention_sum_heads_decode_{suffix}")(
        output, head_output,
        x.reshape(batch, cfg.dim) if addend is None else addend.reshape(batch, cfg.dim),
        batch, cfg.n_heads, cfg.dim, int(addend is not None),
        threads=(batch * cfg.dim, 1, 1),
        group_size=(min(256, batch * cfg.dim), 1, 1),
    )
    return output.view(batch, 1, cfg.dim)


def metal_mla_attention_decode(module, x: Tensor, cache, position: int,
                               absorbed_query: Tensor,
                               absorbed_output: Tensor,
                               addend: Tensor | None = None,
                               defer_sum: bool = False) -> Tensor:
    """Fused latent-cache MLA decode followed by absorbed output projection."""
    cfg = module.cfg
    if cfg.kv_latent_dim != 128 or cfg.max_seq_len > 4096:
        raise RuntimeError("The fused MLA kernel requires latent_dim=128 and context<=4096")
    batch = x.shape[0]
    head_output = _scratch(
        module, "_metal_decode_heads", (batch, cfg.n_heads, cfg.dim), x,
        torch.float32,
    )
    output = _scratch(module, "_metal_decode_output", (batch, cfg.dim), x)
    suffix = "bf16" if x.dtype == torch.bfloat16 else "fp16"
    lib = _library(x.dtype)
    getattr(lib, f"mla_attention_decode_{suffix}")(
        head_output, x, module.kv_down.weight, absorbed_query, cache.latent,
        batch, position, cfg.dim, cfg.n_heads, cfg.kv_latent_dim,
        cfg.max_seq_len, cfg.head_dim ** -0.5, absorbed_output,
        threads=(batch * cfg.n_heads * 256, 1, 1),
        group_size=(256, 1, 1),
    )
    if defer_sum:
        return head_output
    getattr(lib, f"attention_sum_heads_decode_{suffix}")(
        output, head_output,
        x.reshape(batch, cfg.dim) if addend is None else addend.reshape(batch, cfg.dim),
        batch, cfg.n_heads, cfg.dim, int(addend is not None),
        threads=(batch * cfg.dim, 1, 1),
        group_size=(min(256, batch * cfg.dim), 1, 1),
    )
    return output.view(batch, 1, cfg.dim)


def metal_vocab_argmax(hidden: Tensor, weight: Tensor) -> Tensor:
    """Project to tied vocabulary rows and reduce directly to token IDs."""
    batch = hidden.shape[0]
    dim = hidden.shape[-1]
    vocab = weight.shape[0]
    groups = (vocab + 255) // 256
    partial_values = torch.empty(
        (batch, groups), device=hidden.device, dtype=torch.float32
    )
    partial_indices = torch.empty(
        (batch, groups), device=hidden.device, dtype=torch.int32
    )
    output = torch.empty((batch, 1), device=hidden.device, dtype=torch.int64)
    suffix = "bf16" if hidden.dtype == torch.bfloat16 else "fp16"
    lib = _library(hidden.dtype)
    getattr(lib, f"vocab_argmax_partials_{suffix}")(
        partial_values, partial_indices, hidden, weight,
        batch, dim, vocab, groups,
        threads=(batch * groups * 256, 1, 1), group_size=(256, 1, 1),
    )
    getattr(lib, f"vocab_argmax_reduce_{suffix}")(
        output, partial_values, partial_indices, batch, groups,
        threads=(batch * 256, 1, 1), group_size=(256, 1, 1),
    )
    return output


def dense_mps_experts(module: "MoE", flat: Tensor, indices: Tensor,
                      weights: Tensor) -> Tensor:
    """Compute every expert with large batched MPS operations for comparison."""
    if flat.device.type != "mps" or torch.is_grad_enabled():
        raise RuntimeError("The dense MPS backend is inference-only and requires MPS")
    routed_gate_up, routed_down, shared_gate_up = pack_combined_weights(module)
    gate_up = torch.einsum("nd,ehd->neh", flat, routed_gate_up)
    hidden = _metal_swiglu(gate_up, module.cfg.expert_dim)
    expert_outputs = torch.einsum("neh,edh->ned", hidden, routed_down)
    dispatch = torch.zeros(
        (flat.shape[0], module.cfg.n_experts), device=flat.device, dtype=weights.dtype
    ).scatter(1, indices, weights)
    routed = (expert_outputs * dispatch.unsqueeze(-1).to(expert_outputs.dtype)).sum(1)
    return routed + _fused_shared(module, flat, shared_gate_up)


def grouped_mps_experts(module: "MoE", flat: Tensor, indices: Tensor,
                        weights: Tensor, positions: Tensor | None = None,
                        counts: Tensor | None = None) -> Tensor:
    """Pack routed tokens by expert, use batched MPS GEMMs, and gather exactly."""
    if flat.device.type != "mps" or flat.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError("The grouped MPS backend requires MPS BF16 or FP16")
    if torch.is_grad_enabled():
        raise RuntimeError("The grouped MPS backend is inference-only")
    tokens, dim = flat.shape
    cfg = module.cfg
    assignments = tokens * cfg.top_k
    lib = _library(flat.dtype)
    if positions is None:
        positions = torch.empty(assignments, device=flat.device, dtype=torch.int32)
        counts = torch.zeros(cfg.n_experts, device=flat.device, dtype=torch.int32)
        lib.moe_count_assignments(positions, counts, indices, assignments)
    # Twice the balanced assignment count handles ordinary skew and gives MPS
    # a matrix shape it executes efficiently. Overflow remains exact.
    capacity = max(8, ((2 * assignments + cfg.n_experts - 1)
                       // cfg.n_experts + 7) // 8 * 8)
    routed_gate_up, routed_down, shared_gate_up = pack_combined_weights(module)
    grouped = torch.zeros(
        (cfg.n_experts, capacity, dim), device=flat.device, dtype=flat.dtype
    )
    suffix = "bf16" if flat.dtype == torch.bfloat16 else "fp16"
    getattr(lib, f"moe_scatter_grouped_{suffix}")(
        grouped, flat, indices, positions, assignments, dim, capacity
    )
    overflow_hidden = torch.empty(
        (assignments, cfg.expert_dim), device=flat.device, dtype=flat.dtype
    )
    getattr(lib, f"moe_overflow_hidden_{suffix}")(
        overflow_hidden, flat, indices, positions, routed_gate_up,
        assignments, dim, cfg.expert_dim, capacity,
    )
    gate_up = torch.bmm(grouped, routed_gate_up.transpose(1, 2))
    hidden = _metal_swiglu(gate_up, cfg.expert_dim)
    expert_output = torch.bmm(hidden, routed_down.transpose(1, 2))
    shared_output = _fused_shared(module, flat, shared_gate_up)
    output = torch.empty_like(flat)
    getattr(lib, f"moe_gather_grouped_{suffix}")(
        output, expert_output, overflow_hidden, shared_output, indices, positions,
        weights, routed_down, tokens, dim, cfg.expert_dim, capacity,
    )
    return output


def metal_depth_mix(module, sources: list[Tensor]) -> Tensor:
    """Fused inference-only Block AttnRes scoring and mixing on MPS."""
    first = sources[0]
    if first.device.type != "mps" or first.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError("The Metal depth backend requires MPS BF16 or FP16")
    if torch.is_grad_enabled():
        raise RuntimeError("The Metal depth backend is inference-only")
    if not 1 <= len(sources) <= 9:
        raise RuntimeError("Metal depth mixing supports one to nine sources")
    tokens = first.numel() // first.shape[-1]
    dim = first.shape[-1]
    flattened = [source.reshape(tokens, dim) for source in sources]
    padded = flattened + [flattened[0]] * (9 - len(flattened))
    output = _scratch(module, "_metal_norm_output", (tokens, dim), first)
    lib = _library(first.dtype)
    suffix = "bf16" if first.dtype == torch.bfloat16 else "fp16"
    if tokens <= 16:
        getattr(lib, f"depth_mix_decode_{suffix}")(
            output, *padded, module.query, module.norm.weight,
            tokens, dim, len(sources), module.norm.eps,
            threads=(tokens * 256, 1, 1), group_size=(256, 1, 1),
        )
    else:
        scores = torch.empty(
            (len(sources), tokens), device=first.device, dtype=torch.float32
        )
        getattr(lib, f"depth_scores_{suffix}")(
            scores, *padded, module.query, module.norm.weight,
            tokens, dim, len(sources), module.norm.eps,
        )
        getattr(lib, f"depth_mix_{suffix}")(
            output, scores, *padded, tokens, dim, len(sources),
        )
    return output.view_as(first)


def metal_depth_mix_norm(module, sources: list[Tensor], output_norm) -> Tensor:
    """Fuse small-batch Block AttnRes mixing with its consumer RMSNorm."""
    first = sources[0]
    tokens = first.numel() // first.shape[-1]
    dim = first.shape[-1]
    if tokens > 16 or dim > 768:
        return output_norm(metal_depth_mix(module, sources))
    if first.device.type != "mps" or first.dtype not in (
        torch.bfloat16, torch.float16
    ):
        return output_norm(module(sources))
    if torch.is_grad_enabled():
        raise RuntimeError("The fused depth/norm backend is inference-only")
    flattened = [source.reshape(tokens, dim) for source in sources]
    padded = flattened + [flattened[0]] * (9 - len(flattened))
    output = _scratch(module, "_metal_norm_output", (tokens, dim), first)
    suffix = "bf16" if first.dtype == torch.bfloat16 else "fp16"
    getattr(_library(first.dtype), f"depth_mix_norm_decode_{suffix}")(
        output, *padded, module.query, module.norm.weight, output_norm.weight,
        tokens, dim, len(sources), module.norm.eps, output_norm.eps,
        threads=(tokens * 256, 1, 1), group_size=(256, 1, 1),
    )
    return output.view_as(first)


def metal_depth_mix_norm_pending(module, sources: list[Tensor],
                                 contributions: Tensor,
                                 base: Tensor | None, output_norm):
    """Resolve deferred head/expert contributions inside depth mix and RMSNorm."""
    first = sources[0]
    tokens = first.numel() // first.shape[-1]
    dim = first.shape[-1]
    flattened = [source.reshape(tokens, dim) for source in sources]
    padded = flattened + [flattened[0]] * (9 - len(flattened))
    output = _scratch(module, "_metal_norm_output", (tokens, dim), first)
    resolved = _scratch(module, "_metal_resolved_output", (tokens, dim), first)
    suffix = "bf16" if first.dtype == torch.bfloat16 else "fp16"
    getattr(_library(first.dtype), f"depth_mix_norm_pending_{suffix}")(
        output, resolved, contributions,
        flattened[0] if base is None else base.reshape(tokens, dim),
        *padded, module.query, module.norm.weight, output_norm.weight,
        tokens, dim, len(sources), contributions.shape[1], int(base is not None),
        module.norm.eps, output_norm.eps,
        threads=(tokens * 256, 1, 1), group_size=(256, 1, 1),
    )
    return output.view_as(first), resolved.view_as(first)
