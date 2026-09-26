"""PyTorch training, prefill, and cached generation for the Parrot pilot."""
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


_ROPE_CACHE: dict[tuple, tuple[Tensor, Tensor]] = {}
_LOCAL_MASK_CACHE: dict[tuple, list[Tensor]] = {}


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768  # Mistral v0.3; derive from tokenizer for others.
    dim: int = 768
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 96
    kv_latent_dim: int = 128
    global_every: int = 4
    window_size: int = 2048  # Includes the current position.
    max_seq_len: int = 4096  # Model positions, including during TST.
    n_experts: int = 16
    top_k: int = 2
    expert_dim: int = 512
    shared_expert_dim: int = 256
    moe_backend: Literal[
        "pytorch", "metal", "dense_mps", "grouped_mps", "training_mps"
    ] = "pytorch"
    depth_backend: Literal["pytorch", "metal"] = "pytorch"
    residual: Literal["attnres", "standard"] = "attnres"
    depth_block_size: int = 4  # Attention and MoE each count as one module.
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    aux_loss_weight: float = 0.01
    collect_router_stats: bool = True
    checkpoint_modules: bool = False
    local_query_chunk: int = 256

    def __post_init__(self):
        for name in ("vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
                     "head_dim", "kv_latent_dim", "global_every", "window_size",
                     "max_seq_len", "n_experts", "top_k", "expert_dim",
                     "shared_expert_dim", "depth_block_size", "local_query_chunk"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_heads % self.n_kv_heads or self.head_dim % 2:
            raise ValueError("n_heads must divide by n_kv_heads; head_dim must be even")
        if self.top_k > self.n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        if self.residual not in ("attnres", "standard"):
            raise ValueError("residual must be attnres or standard")
        if self.moe_backend not in (
            "pytorch", "metal", "dense_mps", "grouped_mps", "training_mps"
        ):
            raise ValueError("invalid moe_backend")
        if self.depth_backend not in ("pytorch", "metal"):
            raise ValueError("depth_backend must be pytorch or metal")
        if self.moe_backend in ("metal", "grouped_mps"):
            if self.top_k != 2:
                raise ValueError(f"{self.moe_backend} requires top_k=2")
            if self.n_experts > 32:
                raise ValueError(
                    f"{self.moe_backend} supports at most 32 routed experts"
                )
            if any(width % 4 for width in (
                self.dim, self.expert_dim, self.shared_expert_dim
            )):
                raise ValueError(f"{self.moe_backend} requires widths divisible by four")
        depth_sources = 1 + (
            2 * self.n_layers + self.depth_block_size - 1
        ) // self.depth_block_size
        if self.depth_backend == "metal" and (self.dim % 4 or depth_sources > 9):
            raise ValueError("metal depth mixing requires dim divisible by four and at most nine sources")
        if self.norm_eps <= 0 or self.rope_theta <= 0 or self.aux_loss_weight < 0:
            raise ValueError("Invalid norm_eps, rope_theta, or aux_loss_weight")


@dataclass
class LocalKVCache:
    key: Tensor  # [batch, kv heads, max positions, head width], RoPE applied.
    value: Tensor


@dataclass
class MLALatentCache:
    latent: Tensor  # [batch, max positions, latent width]


@dataclass
class GenerationState:
    layers: list[LocalKVCache | MLALatentCache]
    position: int
    batch_size: int


@dataclass
class PendingResidual:
    base: Tensor | None
    contributions: Tensor  # FP32 [batch, heads-or-expert-lanes, dim]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, use_metal: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.use_metal = use_metal

    def forward(self, x: Tensor) -> Tensor:
        if (self.use_metal and x.device.type == "mps" and
                x.dtype in (torch.float32, torch.bfloat16, torch.float16)):
            if torch.is_grad_enabled():
                from .training import rms_norm
                return rms_norm(x, self.weight, self.eps)
            if x.dtype in (torch.bfloat16, torch.float16):
                from .metal_moe import metal_rms_norm
                return metal_rms_norm(self, x)
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class DepthMix(nn.Module):
    """Normalize keys only; values are unnormalized block sums."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.query = nn.Parameter(torch.zeros(cfg.dim))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps, cfg.depth_backend == "metal")

    def forward(self, sources: list[Tensor]) -> Tensor:
        if self.cfg.depth_backend == "metal":
            from .metal_moe import metal_depth_mix
            return metal_depth_mix(self, sources)
        if (sources[0].device.type == "mps" and
                sources[0].dtype in (torch.float32, torch.bfloat16, torch.float16)):
            from .training import depth_mix
            return depth_mix(self, sources)
        values = torch.stack(sources, dim=0)
        normalized = F.rms_norm(
            values, (self.cfg.dim,), self.norm.weight, self.norm.eps
        )
        scores = (normalized.float() * self.query.float()).sum(-1)
        weights = scores.softmax(dim=0).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(0)

    def normalized(self, sources: list[Tensor], output_norm: RMSNorm) -> Tensor:
        if (self.cfg.depth_backend == "metal" and
                sources[0].device.type == "mps" and
                sources[0].dtype in (torch.bfloat16, torch.float16) and
                not torch.is_grad_enabled()):
            from .metal_moe import metal_depth_mix_norm
            return metal_depth_mix_norm(self, sources, output_norm)
        return output_norm(self(sources))


class LocalGQA(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

    def _rope_tables(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Compute angles in FP32 even when the model weights are BF16.
        d = self.cfg.head_dim
        cache_key = (
            x.device.type, x.device.index, x.dtype, self.cfg.max_seq_len, d,
            self.cfg.rope_theta, torch.is_inference_mode_enabled(),
        )
        cached = _ROPE_CACHE.get(cache_key)
        if cached is None:
            freq = self.cfg.rope_theta ** (
                -torch.arange(0, d, 2, device=x.device).float() / d
            )
            angles = torch.arange(
                self.cfg.max_seq_len, device=x.device
            ).float()[:, None] * freq
            cached = (angles.cos().to(x.dtype), angles.sin().to(x.dtype))
            _ROPE_CACHE[cache_key] = cached
        return cached

    def _rope(self, x: Tensor, start_position: int = 0) -> Tensor:
        cached = self._rope_tables(x)
        end_position = start_position + x.shape[-2]
        if end_position > self.cfg.max_seq_len:
            raise ValueError("RoPE position exceeds max_seq_len")
        cos, sin = (
            cached[0][start_position:end_position],
            cached[1][start_position:end_position],
        )
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def _project(self, x: Tensor, start_position: int = 0):
        b, t, _ = x.shape
        c = self.cfg
        q = self._rope(
            self.q(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2),
            start_position,
        )
        k = self._rope(
            self.k(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2),
            start_position,
        )
        v = self.v(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        return q, k, v

    def _project_decode(self, x: Tensor, position: int):
        c = self.cfg
        packed = getattr(self, "_decode_qkv_cache", None)
        if packed is not None and not torch.is_grad_enabled() and not self.training:
            packed_weight = packed[1]
        else:
            parameters = (self.q.weight, self.k.weight, self.v.weight)
            version = tuple(
                (p._version, p.device, p.dtype, p.data_ptr()) for p in parameters
            )
            if packed is None or packed[0] != version:
                packed = (
                    version,
                    torch.cat([p.detach() for p in parameters], dim=0).contiguous(),
                )
                self._decode_qkv_cache = packed
            packed_weight = packed[1]
        q_width = c.n_heads * c.head_dim
        kv_width = c.n_kv_heads * c.head_dim
        q_linear, k_linear, v_linear = F.linear(x, packed_weight).split(
            (q_width, kv_width, kv_width), dim=-1
        )
        b = x.shape[0]
        q = self._rope(
            q_linear.view(b, 1, c.n_heads, c.head_dim).transpose(1, 2), position
        )
        k = self._rope(
            k_linear.view(b, 1, c.n_kv_heads, c.head_dim).transpose(1, 2), position
        )
        v = v_linear.view(b, 1, c.n_kv_heads, c.head_dim).transpose(1, 2)
        return q, k, v

    def _prefill(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, t, _ = x.shape
        c = self.cfg
        q, cached_k, cached_v = self._project(x)
        if t <= c.window_size and t <= c.local_query_chunk:
            y = F.scaled_dot_product_attention(
                q, cached_k, cached_v, is_causal=True, enable_gqa=True,
            )
            y = y.transpose(1, 2).reshape(b, t, -1)
            return self.out(y), cached_k, cached_v
        mask_key = (x.device.type, x.device.index, t, c.window_size,
                    c.local_query_chunk)
        masks = _LOCAL_MASK_CACHE.get(mask_key)
        if masks is None:
            masks = []
            for start in range(0, t, c.local_query_chunk):
                end = min(t, start + c.local_query_chunk)
                lo = max(0, start - c.window_size + 1)
                qi = torch.arange(start, end, device=x.device)[:, None]
                ki = torch.arange(lo, end, device=x.device)[None, :]
                masks.append((ki <= qi) & (ki > qi - c.window_size))
            _LOCAL_MASK_CACHE[mask_key] = masks
        chunks = []
        for chunk_index, start in enumerate(range(0, t, c.local_query_chunk)):
            end = min(t, start + c.local_query_chunk)
            lo = max(0, start - c.window_size + 1)
            chunks.append(F.scaled_dot_product_attention(
                q[:, :, start:end], cached_k[:, :, lo:end], cached_v[:, :, lo:end],
                attn_mask=masks[chunk_index], enable_gqa=True,
            ))
        y = torch.cat(chunks, dim=2).transpose(1, 2).reshape(b, t, -1)
        return self.out(y), cached_k, cached_v

    def forward(self, x: Tensor) -> Tensor:
        return self._prefill(x)[0]

    def prefill(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self._prefill(x)

    def decode(self, x: Tensor, cache: LocalKVCache, position: int,
               addend: Tensor | None = None,
               defer_sum: bool = False) -> Tensor:
        if x.shape[1] != 1:
            raise ValueError("decode expects one model position")
        if (x.device.type == "mps" and
                x.dtype in (torch.bfloat16, torch.float16) and
                not torch.is_grad_enabled()):
            from .metal_moe import metal_local_attention_decode
            cos, sin = self._rope_tables(x)
            return metal_local_attention_decode(
                self, x, cache, position, cos, sin, addend, defer_sum
            )
        if defer_sum:
            raise RuntimeError("deferred attention reduction requires MPS")
        q, k, v = self._project_decode(x, position)
        cache.key[:, :, position:position + 1].copy_(k)
        cache.value[:, :, position:position + 1].copy_(v)
        lo = max(0, position - self.cfg.window_size + 1)
        cached_k = cache.key[:, :, lo:position + 1]
        cached_v = cache.value[:, :, lo:position + 1]
        y = F.scaled_dot_product_attention(
            q, cached_k, cached_v, enable_gqa=True
        )
        result = self.out(y.transpose(1, 2).reshape(x.shape[0], 1, -1))
        return result if addend is None else addend + result


class GlobalMLA(nn.Module):
    """NoPE joint KV compression, with direct queries and explicit KV expansion.

    The training forward uses expanded heads. Incremental decode retains only
    kv_down(x) and absorbs the up-projections into the query and output paths.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        width = cfg.n_heads * cfg.head_dim
        self.q = nn.Linear(cfg.dim, width, bias=False)
        self.kv_down = nn.Linear(cfg.dim, cfg.kv_latent_dim, bias=False)
        self.k_up = nn.Linear(cfg.kv_latent_dim, width, bias=False)
        self.v_up = nn.Linear(cfg.kv_latent_dim, width, bias=False)
        self.out = nn.Linear(width, cfg.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.prefill(x)[0]

    def prefill(self, x: Tensor) -> tuple[Tensor, Tensor]:
        b, t, _ = x.shape
        def heads(y):
            return y.view(b, t, self.cfg.n_heads, self.cfg.head_dim).transpose(1, 2)
        q_linear, latent = self.q(x), self.kv_down(x)
        k_linear, v_linear = self.k_up(latent), self.v_up(latent)
        y = F.scaled_dot_product_attention(
            heads(q_linear), heads(k_linear), heads(v_linear),
            is_causal=True,
        )
        return self.out(y.transpose(1, 2).reshape(b, t, -1)), latent

    def _decode_weights(self) -> tuple[Tensor, Tensor]:
        c = self.cfg
        packed = getattr(self, "_decode_weight_cache", None)
        if packed is not None and not torch.is_grad_enabled() and not self.training:
            return packed[1]
        parameters = (
            self.q.weight, self.k_up.weight, self.v_up.weight, self.out.weight
        )
        version = tuple(
            (p._version, p.device, p.dtype, p.data_ptr()) for p in parameters
        )
        if packed is None or packed[0] != version:
            query = self.q.weight.detach().float().view(
                c.n_heads, c.head_dim, c.dim
            )
            key = self.k_up.weight.detach().float().view(
                c.n_heads, c.head_dim, c.kv_latent_dim
            )
            value = self.v_up.weight.detach().float().view(
                c.n_heads, c.head_dim, c.kv_latent_dim
            )
            output = self.out.weight.detach().float().view(
                c.dim, c.n_heads, c.head_dim
            ).permute(1, 2, 0)
            absorbed_query = torch.einsum(
                "hdl,hdm->hlm", key, query
            ).reshape(c.n_heads * c.kv_latent_dim, c.dim).contiguous()
            absorbed_output = torch.einsum(
                "hdl,hdm->hlm", value, output
            ).permute(2, 0, 1).reshape(
                c.dim, c.n_heads * c.kv_latent_dim
            ).to(self.q.weight.dtype).contiguous()
            absorbed_query = absorbed_query.to(self.q.weight.dtype)
            packed = (version, (absorbed_query, absorbed_output))
            self._decode_weight_cache = packed
        return packed[1]

    def decode(self, x: Tensor, cache: MLALatentCache, position: int,
               addend: Tensor | None = None,
               defer_sum: bool = False) -> Tensor:
        if x.shape[1] != 1:
            raise ValueError("decode expects one model position")
        c = self.cfg
        b = x.shape[0]
        absorbed_query, absorbed_output = self._decode_weights()
        if (x.device.type == "mps" and
                x.dtype in (torch.bfloat16, torch.float16) and
                not torch.is_grad_enabled()):
            from .metal_moe import metal_mla_attention_decode
            return metal_mla_attention_decode(
                self, x, cache, position, absorbed_query, absorbed_output,
                addend, defer_sum
            )
        if defer_sum:
            raise RuntimeError("deferred attention reduction requires MPS")
        latent = self.kv_down(x)
        cache.latent[:, position:position + 1].copy_(latent)
        cached = cache.latent[:, :position + 1]
        q_latent = F.linear(x.float(), absorbed_query.float()).view(
            b, 1, c.n_heads, c.kv_latent_dim
        ).transpose(1, 2)
        latent_heads = cached[:, None].expand(-1, c.n_heads, -1, -1)
        context = F.scaled_dot_product_attention(
            q_latent, latent_heads, latent_heads,
            scale=c.head_dim ** -0.5,
        )
        context = context.transpose(1, 2).reshape(
            b, 1, c.n_heads * c.kv_latent_dim
        )
        result = F.linear(context, absorbed_output.float()).to(x.dtype)
        return result if addend is None else addend + result


class SwiGLU(nn.Module):
    def __init__(self, dim: int, intermediate: int, metal_training: bool = False):
        super().__init__()
        self.metal_training = metal_training
        self.gate = nn.Linear(dim, intermediate, bias=False)
        self.up = nn.Linear(dim, intermediate, bias=False)
        self.down = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate(x), self.up(x)
        if self.metal_training:
            from .training import swiglu
            return self.down(swiglu(gate, up))
        return self.down(F.silu(gate) * up)


class MoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        if cfg.moe_backend == "training_mps":
            self.expert_gate = nn.Parameter(torch.empty(
                cfg.n_experts, cfg.expert_dim, cfg.dim
            ))
            self.expert_up = nn.Parameter(torch.empty(
                cfg.n_experts, cfg.expert_dim, cfg.dim
            ))
            self.expert_down = nn.Parameter(torch.empty(
                cfg.n_experts, cfg.dim, cfg.expert_dim
            ))
            nn.init.normal_(self.expert_gate, std=0.02)
            nn.init.normal_(self.expert_up, std=0.02)
            nn.init.normal_(self.expert_down, std=0.02)
            self.experts = None
        else:
            self.experts = nn.ModuleList([
                SwiGLU(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)
            ])
        self.shared = SwiGLU(cfg.dim, cfg.shared_expert_dim, cfg.moe_backend == "training_mps")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None]:
        flat = x.reshape(-1, x.shape[-1])
        fused_route = (
            self.cfg.moe_backend == "grouped_mps" and
            x.device.type == "mps" and
            x.dtype in (torch.bfloat16, torch.float16) and
            not torch.is_grad_enabled() and
            self.cfg.top_k == 2 and self.cfg.n_experts <= 32
        )
        route_counts = positions = None
        if fused_route:
            from .metal_moe import metal_route
            probabilities, weights, indices, positions, route_counts = metal_route(
                self, flat, self.cfg.collect_router_stats
            )
        else:
            # Disable autocast: both projection and softmax run in FP32.
            with torch.autocast(device_type=x.device.type, enabled=False):
                probabilities = F.linear(
                    flat.float(), self.router.weight.float()
                ).softmax(-1)
            weights, indices = probabilities.topk(self.cfg.top_k, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
        if self.cfg.moe_backend == "metal":
            from .metal_moe import metal_experts
            result = metal_experts(self, flat, indices, weights)
        elif self.cfg.moe_backend == "dense_mps":
            from .metal_moe import dense_mps_experts
            result = dense_mps_experts(self, flat, indices, weights)
        elif self.cfg.moe_backend == "grouped_mps":
            from .metal_moe import grouped_mps_experts
            result = grouped_mps_experts(
                self, flat, indices, weights, positions, route_counts
            )
        elif self.cfg.moe_backend == "training_mps" and x.device.type == "mps":
            from .training import grouped_experts
            result = grouped_experts(self, flat, indices, weights)
        else:
            result = self.shared(flat)
            for expert_id, expert in enumerate(self.experts):
                rows, slots = torch.where(indices == expert_id)
                if rows.numel():
                    contribution = expert(flat[rows]) * weights[rows, slots, None].to(result.dtype)
                    result = result.index_add(0, rows, contribution)
        # MPS bincount performs several GPU-to-CPU scalar reads. A scatter
        # reduction stays on the command stream and computes identical counts.
        if not self.cfg.collect_router_stats:
            return result.view_as(x), None, None
        if route_counts is not None:
            counts = route_counts.float()
        else:
            flat_indices = indices.flatten()
            counts = torch.zeros(
                self.cfg.n_experts, device=indices.device, dtype=torch.float32
            ).scatter_add_(
                0, flat_indices,
                torch.ones(flat_indices.shape, device=indices.device,
                           dtype=torch.float32),
            )
        utilization = counts / indices.numel()  # Sums to one; no token dropping.
        aux = self.cfg.n_experts * (utilization * probabilities.mean(0)).sum()
        return result.view_as(x), aux, utilization.detach()

    def inference(self, x: Tensor, *, direct: bool,
                  addend: Tensor | None = None,
                  defer_sum: bool = False) -> Tensor:
        """Inference without router diagnostics; direct=True favors decode GEMV."""
        flat = x.reshape(-1, x.shape[-1])
        use_metal_route = (
            x.device.type == "mps" and
            x.dtype in (torch.bfloat16, torch.float16) and
            not torch.is_grad_enabled() and
            self.cfg.moe_backend not in ("pytorch", "training_mps") and
            self.cfg.top_k == 2 and self.cfg.n_experts <= 32
        )
        if not use_metal_route:
            result = self.forward(x)[0]
            return result if addend is None else addend + result
        from .metal_moe import metal_experts, metal_experts_decode, metal_route
        if direct:
            result = metal_experts_decode(self, flat, addend, defer_sum)
            return result if defer_sum else result.view_as(x)
        if defer_sum:
            raise ValueError("deferred MoE reduction requires direct inference")
        if addend is not None:
            raise ValueError("addend is only supported by direct MoE inference")
        _, weights, indices, positions, counts = metal_route(
            self, flat, collect_stats=False
        )
        if self.cfg.moe_backend == "metal":
            result = metal_experts(self, flat, indices, weights)
        elif self.cfg.moe_backend == "grouped_mps":
            from .metal_moe import grouped_mps_experts
            result = grouped_mps_experts(
                self, flat, indices, weights, positions, counts
            )
        else:
            from .metal_moe import dense_mps_experts
            result = dense_mps_experts(self, flat, indices, weights)
        return result.view_as(x)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, index: int):
        super().__init__()
        self.attention = GlobalMLA(cfg) if (index + 1) % cfg.global_every == 0 else LocalGQA(cfg)
        self.moe = MoE(cfg)
        use_metal = cfg.depth_backend == "metal" or cfg.moe_backend != "pytorch"
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps, use_metal)
        self.moe_norm = RMSNorm(cfg.dim, cfg.norm_eps, use_metal)

    def attend(self, x):
        return self.attention(self.attention_norm(x))

    def feedforward(self, x):
        return self.moe(self.moe_norm(x))

    def feedforward_inference(self, x, *, direct: bool):
        return self.moe.inference(self.moe_norm(x), direct=direct)


@dataclass
class ModelOutput:
    logits: Tensor  # [batch, model positions, vocab]
    loss: Tensor | None
    lm_loss: Tensor | None
    aux_loss: Tensor  # Unweighted average over layers.
    expert_utilization: Tensor  # [layers, experts], fraction of assignments.


class Parrot(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.depth_mix = nn.ModuleList(
            [DepthMix(cfg) for _ in range(2 * cfg.n_layers + 1)]
            if cfg.residual == "attnres" else []
        )
        use_metal = cfg.depth_backend == "metal" or cfg.moe_backend != "pytorch"
        self.final_norm = RMSNorm(cfg.dim, cfg.norm_eps, use_metal)
        # F.linear below uses embedding.weight directly: no duplicate head.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None,
                *, bag_size: int = 1,
                supervised_logits_only: bool = False) -> ModelOutput:
        """Unpadded streams. labels are UNSHIFTED source-token IDs, or -100.

        bag_size=1: position i predicts token i+1.
        bag_size=4: mean embedding of bag i predicts each token in bag i+1.
        Set labels=input_ids for language-model training. No automatic schedule.
        """
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty [batch, source_tokens] tensor")
        if not isinstance(bag_size, int) or bag_size < 1 or input_ids.shape[1] % bag_size:
            raise ValueError("source_tokens must be divisible by positive integer bag_size")
        b, source_len = input_ids.shape
        t = source_len // bag_size
        if t > self.config.max_seq_len:
            raise ValueError("Input exceeds max_seq_len model positions")
        if labels is not None and (labels.shape != input_ids.shape or t < 2):
            raise ValueError("labels must match input_ids and contain at least two model positions")
        if supervised_logits_only and labels is None:
            raise ValueError("supervised_logits_only requires labels")
        x = self.embedding(input_ids).reshape(b, t, bag_size, -1).mean(2)
        completed, partial = [x], None
        aux_losses, utilization = [], []
        module_index = 0
        for block in self.blocks:
            for is_moe, operation in ((False, block.attend), (True, block.feedforward)):
                if self.config.residual == "attnres":
                    sources = completed + ([] if partial is None else [partial])
                    h = self.depth_mix[module_index](sources)
                else:
                    h = x
                if self.config.checkpoint_modules and self.training and torch.is_grad_enabled():
                    result = checkpoint(operation, h, use_reentrant=False)
                else:
                    result = operation(h)
                if is_moe:
                    delta, aux, usage = result
                    if self.config.collect_router_stats:
                        aux_losses.append(aux)
                        utilization.append(usage)
                else:
                    delta = result
                if self.config.residual == "attnres":
                    partial = delta if partial is None else partial + delta
                    if (module_index + 1) % self.config.depth_block_size == 0:
                        completed.append(partial)
                        partial = None
                else:
                    x = x + delta
                module_index += 1
        if self.config.residual == "attnres":
            x = self.depth_mix[-1](completed + ([] if partial is None else [partial]))
        # The last position has no shifted target. Training callers that only
        # consume the loss can skip its large vocabulary projection exactly;
        # the transformer and router auxiliary loss still use every position.
        logit_input = x[:, :-1] if supervised_logits_only else x
        logits = F.linear(self.final_norm(logit_input), self.embedding.weight)
        if self.config.collect_router_stats:
            aux_loss = torch.stack(aux_losses).mean()
            expert_utilization = torch.stack(utilization)
        else:
            aux_loss = x.new_zeros((), dtype=torch.float32)
            expert_utilization = x.new_zeros(
                (self.config.n_layers, self.config.n_experts), dtype=torch.float32
            )
        lm_loss = loss = None
        if labels is not None:
            targets = labels.reshape(b, t, bag_size)[:, 1:]
            if (self.config.moe_backend == "training_mps" and
                    logits.device.type == "mps"):
                from .training import cross_entropy
                lm_loss = cross_entropy(logits, targets.contiguous())
            else:
                # One log-softmax per position; do not replicate vocab logits per bag.
                log_probs = logits[:, :-1].float().log_softmax(-1)
                valid = targets != -100
                selected = log_probs.gather(-1, targets.masked_fill(~valid, 0))
                lm_loss = -(selected * valid).sum() / valid.sum().clamp_min(1)
            loss = lm_loss + self.config.aux_loss_weight * aux_loss
        return ModelOutput(logits, loss, lm_loss, aux_loss, expert_utilization)

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> tuple[Tensor, GenerationState]:
        """Process a prompt once and return next-token logits plus KV state."""
        if self.training:
            raise RuntimeError("prefill requires model.eval()")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be nonempty [batch, positions]")
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError("Prompt exceeds max_seq_len")
        x = self.embedding(input_ids)
        completed, partial = [x], None
        layer_caches: list[LocalKVCache | MLALatentCache] = []
        module_index = 0
        for block in self.blocks:
            if self.config.residual == "attnres":
                sources = completed + ([] if partial is None else [partial])
                normalized = self.depth_mix[module_index].normalized(
                    sources, block.attention_norm
                )
            else:
                normalized = block.attention_norm(x)
            if isinstance(block.attention, LocalGQA):
                delta, key, value = block.attention.prefill(normalized)
                key_cache = torch.empty(
                    (b, self.config.n_kv_heads, self.config.max_seq_len,
                     self.config.head_dim),
                    device=key.device, dtype=key.dtype,
                )
                value_cache = torch.empty_like(key_cache)
                key_cache[:, :, :t].copy_(key)
                value_cache[:, :, :t].copy_(value)
                layer_caches.append(LocalKVCache(key_cache, value_cache))
            else:
                delta, latent = block.attention.prefill(normalized)
                latent_cache = torch.empty(
                    (b, self.config.max_seq_len, self.config.kv_latent_dim),
                    device=latent.device, dtype=latent.dtype,
                )
                latent_cache[:, :t].copy_(latent)
                layer_caches.append(MLALatentCache(latent_cache))
            if self.config.residual == "attnres":
                partial = delta if partial is None else partial + delta
                if (module_index + 1) % self.config.depth_block_size == 0:
                    completed.append(partial)
                    partial = None
            else:
                x = x + delta
            module_index += 1

            if self.config.residual == "attnres":
                sources = completed + ([] if partial is None else [partial])
                h = self.depth_mix[module_index](sources)
            else:
                h = x
            delta = block.feedforward_inference(h, direct=False)
            if self.config.residual == "attnres":
                partial = delta if partial is None else partial + delta
                if (module_index + 1) % self.config.depth_block_size == 0:
                    completed.append(partial)
                    partial = None
            else:
                x = x + delta
            module_index += 1
        if self.config.residual == "attnres":
            x = self.depth_mix[-1](completed + ([] if partial is None else [partial]))
        logits = F.linear(self.final_norm(x[:, -1:]), self.embedding.weight)
        return logits, GenerationState(layer_caches, t, b)

    @torch.inference_mode()
    def decode_step(self, input_ids: Tensor,
                    state: GenerationState, *,
                    _next_token_only: bool = False) -> tuple[Tensor, GenerationState]:
        """Append one token to an existing generation state."""
        if self.training:
            raise RuntimeError("decode_step requires model.eval()")
        if input_ids.shape != (state.batch_size, 1):
            raise ValueError("decode_step input must be [cached batch, 1]")
        if state.position >= self.config.max_seq_len:
            raise ValueError("Generation state has reached max_seq_len")
        if len(state.layers) != self.config.n_layers:
            raise ValueError("Generation state does not match model depth")
        x = self.embedding(input_ids)
        completed, partial = [x], None
        defer_residuals = (
            self.config.residual == "attnres" and
            self.config.depth_backend == "metal" and
            self.config.moe_backend not in ("pytorch", "training_mps") and
            x.device.type == "mps" and
            x.dtype in (torch.bfloat16, torch.float16)
        )

        def normalized_input(depth: DepthMix, norm: RMSNorm) -> Tensor:
            """Resolve a deferred module output while preparing its consumer."""
            nonlocal completed, partial
            sources = completed + ([] if partial is None else [partial])
            pending = sources[-1]
            if not isinstance(pending, PendingResidual):
                return depth.normalized(sources, norm)

            # The pending item is the most recent logical depth source.  Its
            # reduction can share a dispatch with this depth mix and RMSNorm.
            from .metal_moe import metal_depth_mix_norm_pending
            placeholder = pending.base if pending.base is not None else completed[0]
            tensor_sources = sources[:-1] + [placeholder]
            normalized, resolved = metal_depth_mix_norm_pending(
                depth, tensor_sources, pending.contributions, pending.base, norm
            )
            if partial is pending:
                partial = resolved
            else:
                completed[-1] = resolved
            return normalized

        module_index = 0
        for layer_index, block in enumerate(self.blocks):
            if self.config.residual == "attnres":
                normalized = normalized_input(
                    self.depth_mix[module_index], block.attention_norm
                )
            else:
                normalized = block.attention_norm(x)
            cache = state.layers[layer_index]
            addend = partial if self.config.residual == "attnres" else None
            if isinstance(block.attention, LocalGQA):
                if not isinstance(cache, LocalKVCache):
                    raise ValueError("Local attention received an MLA cache")
                delta = block.attention.decode(
                    normalized, cache, state.position, addend=addend,
                    defer_sum=defer_residuals,
                )
            else:
                if not isinstance(cache, MLALatentCache):
                    raise ValueError("MLA attention received a local cache")
                delta = block.attention.decode(
                    normalized, cache, state.position, addend=addend,
                    defer_sum=defer_residuals,
                )
            if self.config.residual == "attnres":
                partial = (
                    PendingResidual(addend, delta) if defer_residuals else delta
                )
                if (module_index + 1) % self.config.depth_block_size == 0:
                    completed.append(partial)
                    partial = None
            else:
                x = x + delta
            module_index += 1

            if self.config.residual == "attnres":
                normalized = normalized_input(
                    self.depth_mix[module_index], block.moe_norm
                )
                addend = partial
                delta = block.moe.inference(
                    normalized, direct=True, addend=addend,
                    defer_sum=defer_residuals,
                )
                partial = (
                    PendingResidual(addend, delta) if defer_residuals else delta
                )
            else:
                delta = block.feedforward_inference(x, direct=True)
            if self.config.residual == "attnres":
                if (module_index + 1) % self.config.depth_block_size == 0:
                    completed.append(partial)
                    partial = None
            else:
                x = x + delta
            module_index += 1
        if self.config.residual == "attnres":
            normalized = normalized_input(self.depth_mix[-1], self.final_norm)
        else:
            normalized = self.final_norm(x)
        logits = F.linear(normalized, self.embedding.weight)
        output = logits.argmax(-1) if _next_token_only else logits
        state.position += 1
        return output, state

    @torch.inference_mode()
    def decode_next(self, input_ids: Tensor,
                    state: GenerationState) -> tuple[Tensor, GenerationState]:
        """Append one token and return the next greedy token without logits."""
        return self.decode_step(input_ids, state, _next_token_only=True)

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        inactive = (self.config.n_layers * (self.config.n_experts - self.config.top_k)
                    * 3 * self.config.dim * self.config.expert_dim)
        return {"total": total, "active_per_token": total - inactive}
