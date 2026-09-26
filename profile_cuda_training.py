"""Profile one steady-state Parrot CUDA training step after compilation."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from benchmark_cuda_training import BenchmarkConfig, _model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=960)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--bag-size", type=int, default=4)
    parser.add_argument("--n-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="benchmark_cuda_h200_profile.json")
    parser.add_argument("--trace", default="benchmark_cuda_h200_trace.json")
    args = parser.parse_args()

    config = BenchmarkConfig.from_dict({
        "batch": args.batch,
        "seq_len": args.seq_len,
        "bag_size": args.bag_size,
        "n_experts": args.n_experts,
        "top_k": args.top_k,
        "warmup": args.warmup,
        "runs": 1,
        "compile_mode": "default",
    })
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    eager, model = _model(
        config.dtype, config.compile_mode, config.n_experts, config.top_k
    )
    ids = torch.randint(
        eager.config.vocab_size,
        (config.batch, config.seq_len),
        device="cuda",
    )

    for _ in range(config.warmup):
        model.zero_grad(set_to_none=True)
        output = model(
            ids,
            labels=ids,
            bag_size=config.bag_size,
            supervised_logits_only=True,
        )
        output.loss.backward()
        torch.cuda.synchronize()
        del output

    forward_start = torch.cuda.Event(enable_timing=True)
    forward_end = torch.cuda.Event(enable_timing=True)
    backward_end = torch.cuda.Event(enable_timing=True)
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profile:
        forward_start.record()
        with torch.profiler.record_function("parrot_forward"):
            output = model(
                ids,
                labels=ids,
                bag_size=config.bag_size,
                supervised_logits_only=True,
            )
        forward_end.record()
        with torch.profiler.record_function("parrot_backward"):
            output.loss.backward()
        backward_end.record()
        backward_end.synchronize()

    forward_seconds = forward_start.elapsed_time(forward_end) / 1_000
    backward_seconds = forward_end.elapsed_time(backward_end) / 1_000
    elapsed_seconds = forward_seconds + backward_seconds

    profile.export_chrome_trace(args.trace)
    trace = json.loads(Path(args.trace).read_text())
    kernels: dict[str, list[float]] = defaultdict(list)
    gpu_activity = Counter()
    for event in trace["traceEvents"]:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        if category in ("kernel", "gpu_memcpy", "gpu_memset"):
            gpu_activity[category] += 1
        if category == "kernel":
            kernels[event["name"]].append(float(event.get("dur", 0)))
    kernel_rows = [
        {
            "name": name,
            "calls": len(durations),
            "total_us": sum(durations),
            "average_us": sum(durations) / len(durations),
        }
        for name, durations in kernels.items()
    ]
    kernel_rows.sort(key=lambda row: row["total_us"], reverse=True)
    total_kernel_us = sum(row["total_us"] for row in kernel_rows)
    for row in kernel_rows:
        row["percent_of_summed_kernel_time"] = (
            100 * row["total_us"] / total_kernel_us
            if total_kernel_us else 0
        )

    def kernel_category(name: str) -> str:
        lowered = name.lower()
        if any(key in lowered for key in ("flash_", "fmha", "attention")):
            return "attention"
        if "log_softmax" in lowered or "nll_loss" in lowered:
            return "cross_entropy"
        if any(key in lowered for key in (
            "index", "scatter", "gather", "radix", "sort", "scan",
            "topk", "nonzero", "where",
        )):
            return "routing_and_indexing"
        if any(key in lowered for key in (
            "gemm", "grouped_mm", "nvjet", "cutlass", "xmma",
        )):
            return "gemm"
        if lowered.startswith("triton_"):
            return "triton_reduction_and_elementwise"
        return "other"

    category_rows: dict[str, dict[str, float | int | str]] = {}
    for row in kernel_rows:
        category = kernel_category(str(row["name"]))
        aggregate = category_rows.setdefault(category, {
            "category": category, "calls": 0, "total_us": 0.0,
        })
        aggregate["calls"] += int(row["calls"])
        aggregate["total_us"] += float(row["total_us"])
    categories = sorted(
        category_rows.values(), key=lambda row: row["total_us"], reverse=True
    )
    for row in categories:
        row["percent_of_summed_kernel_time"] = (
            100 * float(row["total_us"]) / total_kernel_us
            if total_kernel_us else 0
        )

    source_tokens = config.batch * config.seq_len
    result = {
        "config": config.__dict__,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "elapsed_seconds": elapsed_seconds,
        "source_tokens_per_second": source_tokens / elapsed_seconds,
        "total_summed_kernel_us": total_kernel_us,
        "kernel_launches": gpu_activity["kernel"],
        "gpu_memcpy_calls": gpu_activity["gpu_memcpy"],
        "gpu_memset_calls": gpu_activity["gpu_memset"],
        "kernel_categories": categories,
        "top_cuda_kernels": kernel_rows[:50],
        "operator_table": profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=40
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        key: result[key]
        for key in (
            "forward_seconds", "backward_seconds", "elapsed_seconds",
            "source_tokens_per_second", "kernel_launches",
            "total_summed_kernel_us",
        )
    }, indent=2))
    print(result["operator_table"])
    print(json.dumps(result["top_cuda_kernels"][:20], indent=2))


if __name__ == "__main__":
    main()
