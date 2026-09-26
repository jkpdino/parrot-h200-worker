"""Fused CUDA training operators for Parrot."""
from __future__ import annotations

import os

import torch


def _liger_ops():
    """Import CUDA-only kernels lazily so CPU and Metal installs still work."""
    # Autotuning can temporarily retain several full expert working sets. The
    # large benchmark nearly fills an H200, so use Liger's fixed configuration.
    os.environ.setdefault("LIGER_FUSED_MOE_AUTOTUNE", "0")
    try:
        from liger_kernel.ops import (
            LigerAttnResFunction,
            LigerFusedMoEFunction,
            LigerRMSNormFunction,
            LigerSiLUMulFunction,
        )
    except ImportError as error:
        raise RuntimeError(
            "training_cuda requires the 'cuda' extra: "
            "pip install 'parrot-xl[cuda]'"
        ) from error
    return (
        LigerAttnResFunction,
        LigerFusedMoEFunction,
        LigerRMSNormFunction,
        LigerSiLUMulFunction,
    )


def rms_norm(value, weight, eps):
    """RMSNorm with fused forward and backward CUDA kernels."""
    _, _, function, _ = _liger_ops()
    return function.apply(value, weight, eps, 0.0, "llama", False, None)


def swiglu(gate, up):
    """Fuse SiLU, multiplication, and both activation gradients."""
    *_, function = _liger_ops()
    return function.apply(gate, up, 1.0, 1.0)


def depth_mix(module, sources):
    """Fuse AttnRes normalization, scoring, softmax, mixing, and backward."""
    function, *_ = _liger_ops()
    values = torch.stack(sources)
    return function.apply(
        values, module.query, module.norm.weight, module.norm.eps
    )


def grouped_experts(module, flat, indices, weights):
    """Run fused gather/GEMM/SwiGLU/GEMM/combine MoE forward and backward."""
    _, function, *_ = _liger_ops()
    routed = function.apply(
        flat,
        module.expert_gate_up,
        module.expert_down,
        indices.to(torch.int32),
        weights,
    )
    return module.shared(flat) + routed
