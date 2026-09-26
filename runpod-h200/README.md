# Runpod H200 training worker

This queue worker benchmarks Parrot's complete forward, loss, and backward pass
on an H200. It rejects other GPU models by default and returns the fastest shape
from a request containing up to 16 configurations.

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
