# Runpod H200 training worker

This worker benchmarks Parrot's grouped CUDA training backend over the complete
forward, loss, and backward pass on an H200. It rejects other GPU models by
default and returns the fastest shape from a request containing up to 16
configurations.

Build from the repository root because the image includes the `parrot` package:

```sh
docker build --platform linux/amd64 \
  -f runpod-h200/Dockerfile \
  -t ghcr.io/jkpdino/parrot-h200:cu128-torch28-v1 .
docker push ghcr.io/jkpdino/parrot-h200:cu128-torch28-v1
```

Set the Runpod endpoint's container image to
`ghcr.io/jkpdino/parrot-h200:cu128-torch28-v1`, create a queue endpoint, and
select only the `HOPPER_141` GPU pool. Use one active worker while tuning to
retain TorchInductor's in-process cache. Send `test_input.json` as the request
body.

The timed region includes the full model forward pass, shifted cross-entropy,
router auxiliary loss, and backward pass. It excludes gradient clearing and the
optimizer. CUDA events measure GPU execution after the warmup iterations.

TorchInductor compiles the model, including the non-atomic top-2 route combine
and the multi-target `logsumexp` loss.

For persistent development, launch a normal Pod with
`runpod/pytorch:1.2.0-cu1281-torch280-ubuntu2404`, clone the repository into
`/workspace`, and invoke `benchmark()` directly. The measured 32-expert H200
maximum in the current sweep is 1,189,837 source tokens/second at batch 992,
sequence 512, `bag_size=4`, and top-2 routing.
