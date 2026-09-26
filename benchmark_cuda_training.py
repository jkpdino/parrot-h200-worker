"""CUDA full-model training benchmark used by the Runpod H200 worker."""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import torch

from parrot import ModelConfig, Parrot


@dataclass(frozen=True)
class BenchmarkConfig:
    batch: int = 256
    seq_len: int = 512
    bag_size: int = 4
    warmup: int = 3
    runs: int = 7
    dtype: str = "bfloat16"
    compile_mode: str = "max-autotune"
    require_h200: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BenchmarkConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown benchmark fields: {sorted(unknown)}")
        config = cls(**values)
        if min(config.batch, config.seq_len, config.runs) < 1 or config.warmup < 0:
            raise ValueError("batch, seq_len, and runs must be positive; warmup must be nonnegative")
        if config.bag_size not in (1, 4) or config.seq_len % config.bag_size:
            raise ValueError("bag_size must be 1 or 4 and divide seq_len")
        if config.seq_len // config.bag_size < 2:
            raise ValueError("the input must contain at least two model positions")
        if config.seq_len // config.bag_size > ModelConfig().max_seq_len:
            raise ValueError("model sequence exceeds max_seq_len")
        if config.dtype not in ("bfloat16", "float16"):
            raise ValueError("dtype must be bfloat16 or float16")
        if config.compile_mode not in ("eager", "default", "reduce-overhead", "max-autotune"):
            raise ValueError("invalid compile_mode")
        if config.warmup + config.runs > 50:
            raise ValueError("warmup + runs must not exceed 50")
        return config


def device_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    free, _ = torch.cuda.mem_get_info(index)
    return {
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "free_memory_bytes": free,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


@lru_cache(maxsize=6)
def _model(dtype_name: str, compile_mode: str):
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(42)
    eager = Parrot(ModelConfig(moe_backend="pytorch")).to(
        device="cuda", dtype=dtype
    ).train()
    if compile_mode == "eager":
        return eager, eager
    kwargs = {"backend": "inductor", "dynamic": False}
    if compile_mode != "default":
        kwargs["mode"] = compile_mode
    return eager, torch.compile(eager, **kwargs)


def benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    info = device_info()
    if config.require_h200 and "H200" not in info["name"].upper():
        raise RuntimeError(f"H200 required, worker received {info['name']}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    eager, model = _model(config.dtype, config.compile_mode)
    ids = torch.randint(
        eager.config.vocab_size,
        (config.batch, config.seq_len),
        device="cuda",
    )
    torch.cuda.reset_peak_memory_stats()
    samples = []
    last_loss = None
    try:
        for index in range(config.warmup + config.runs):
            model.zero_grad(set_to_none=True)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(
                ids,
                labels=ids,
                bag_size=config.bag_size,
                supervised_logits_only=True,
            )
            output.loss.backward()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / 1_000
            last_loss = float(output.loss.detach())
            if index >= config.warmup:
                samples.append(elapsed)
            del output
    except torch.cuda.OutOfMemoryError as error:
        eager.zero_grad(set_to_none=True)
        del ids
        torch.cuda.empty_cache()
        raise RuntimeError(f"CUDA out of memory: {error}") from error

    if last_loss is None or not torch.isfinite(torch.tensor(last_loss)):
        raise RuntimeError("nonfinite loss")
    for name, parameter in eager.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"nonfinite gradient: {name}")

    median = statistics.median(samples)
    source_tokens = config.batch * config.seq_len
    model_positions = source_tokens // config.bag_size
    result = {
        "config": asdict(config),
        "device": info,
        "parameters": eager.parameter_counts(),
        "median_seconds": median,
        "source_tokens_per_second": source_tokens / median,
        "model_positions_per_second": model_positions / median,
        "run_seconds": samples,
        "loss": last_loss,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "scope": (
            "full transformer + supervised vocabulary projection + shifted cross entropy "
            "+ router auxiliary loss + backward; excludes optimizer and gradient clearing"
        ),
    }
    del ids
    return result
