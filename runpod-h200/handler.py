"""Runpod Serverless handler for Parrot H200 training-kernel benchmarks."""
from __future__ import annotations

from typing import Any

import runpod
import torch

from benchmark_cuda_training import BenchmarkConfig, benchmark, device_info


def handler(job: dict[str, Any]):
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise ValueError("input must be an object")
    operation = payload.get("operation", "benchmark")
    if operation == "info":
        return {"device": device_info()}
    if operation not in ("benchmark", "sweep"):
        raise ValueError("operation must be info, benchmark, or sweep")

    raw_configs = payload.get("configs") if operation == "sweep" else [
        {key: value for key, value in payload.items() if key != "operation"}
    ]
    if not isinstance(raw_configs, list) or not 1 <= len(raw_configs) <= 16:
        raise ValueError("configs must contain between 1 and 16 benchmark configurations")

    results = []
    for raw in raw_configs:
        if not isinstance(raw, dict):
            raise ValueError("each benchmark configuration must be an object")
        config = BenchmarkConfig.from_dict(raw)
        try:
            results.append({"ok": True, "result": benchmark(config)})
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            results.append({"ok": False, "config": raw, "error": str(error)})
            torch.cuda.empty_cache()
    successful = [item["result"] for item in results if item["ok"]]
    best = max(
        successful, key=lambda item: item["source_tokens_per_second"], default=None
    )
    return {"results": results, "best": best}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
