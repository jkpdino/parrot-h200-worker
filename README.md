# Parrot XL forward pass

Pure PyTorch implementation of the proposed 16-layer MoE decoder. Defaults use
the [Mistral 7B v0.3 tokenizer](https://huggingface.co/mistralai/Mistral-7B-v0.3)
(32,768 entries, including added tokens). The architecture
is randomly initialized; the tokenizer does not provide pretrained model weights.

```sh
python -m pip install -e .
python example.py --tiny
python -m unittest discover -s tests -v
```

The tokenizer downloads from `mistralai/Mistral-7B-v0.3` on first use. Tests use
synthetic IDs and run offline without downloading it.
Only `tokenizer.json` is loaded, with no Mistral model weights or remote Python code.
This is raw-text pretraining: the example disables automatic special tokens and
does not apply a chat template. Another tokenizer can be selected with
`python example.py --tiny --tokenizer openai-community/gpt2`; the example derives
the embedding/head size from the loaded vocabulary.

## M4 Metal inference

The inference-only `grouped_mps` backend runs routing, token grouping, SwiGLU,
gathering, RMSNorm, and Block AttnRes in custom Metal kernels while retaining
MPS matrix multiplication for expert projections. Tokens are grouped into a
fixed-capacity matrix for each expert. An exact overflow path handles arbitrary
routing skew, so no token is dropped.

```python
model = Parrot(ModelConfig(
    moe_backend="grouped_mps",
    depth_backend="metal",
    collect_router_stats=False,  # Training-only diagnostics.
)).to(
    device="mps", dtype=torch.bfloat16
).eval()
with torch.inference_mode():
    output = model(ids.to("mps"))
```

On the test machine's M4 Pro, the complete model processes a batch of one
512-token sequence in a median 97.53 ms, or **5,250 tokens/sec**. This includes
all 16 transformer layers, Block AttnRes, and the full 32,768-token logits.
Reproduce the gated benchmark with:

```sh
python3 benchmark.py --moe-backend grouped_mps --depth-backend metal \
  --dtype bfloat16 --batches 1 --seq-len 512 --warmup 3 --runs 9 \
  --skip-router-stats --min-tokens-per-second 5000
python3 benchmark_moe.py --backend grouped_mps --tokens 256 --dtype bfloat16
```

The integrated kernels are in `parrot/kernels/moe_bf16.metal`. They support BF16
and FP16 and cache packed expert weights until a parameter changes. Set
`collect_router_stats=True` when expert utilization and the load-balancing loss
are needed; `--skip-router-stats` avoids those training diagnostics during a
latency benchmark. The Metal path has no backward kernels, so it requires
`eval()` with gradients disabled. Training uses `moe_backend="pytorch"`.

### Incremental generation

`prefill()` processes the prompt once and returns next-token logits plus a
`GenerationState`. `decode_step()` appends one token without recomputing the
prefix:

```python
logits, state = model.prefill(prompt_ids)
generated = []
for _ in range(16):
    next_id = logits.argmax(-1)
    generated.append(next_id)
    logits, state = model.decode_step(next_id, state)
```

The 12 local layers cache two RoPE key heads and two value heads. The four MLA
layers retain only their 128-wide joint latent; decode absorbs the key and value
up-projections into the query and output operations. The local cache is
preallocated to `max_seq_len` for low decode latency and attention still reads
only the most recent 2,048 positions. At 4,096 positions, the BF16 cache uses
about 40 MiB per sequence: 36 MiB for preallocated local K/V and 4 MiB for MLA
latents.

On the M4 Pro, a batch-one greedy run with a 128-token prompt and 16 generated
tokens improved from 25.4 tokens/sec with prefix recomputation to **56.0
tokens/sec** with the cache, a **2.21x speedup**. Reproduce it with:

```sh
python3 benchmark_generation.py --prompt-len 128 --new-tokens 16 \
  --batch 1 --warmup 1 --runs 3 \
  --min-cached-tokens-per-second 50
```

Cached aggregate generation throughput measured 56.0, 99.8, 161.8, and 295.9
tokens/sec at batch sizes 1, 2, 4, and 8 respectively.

## Standalone Metal proof of concept

`metal/moe.metal` implements one complete MoE module as four native Metal
compute kernels: router projection, top-2 selection, SwiGLU hidden projection,
and weighted down projection. Build and run its CPU-reference check with:

```sh
./metal/build.sh
.build/moe-metal
```

Run the real model dimensions with 256 flattened tokens:

```sh
.build/moe-metal --full --tokens 256 --runs 5
```

The standalone implementation uses FP32 and simple scalar dot-product loops. It
allocates intermediate buffers on each call and implements forward inference
only. It is a correctness baseline for later tiling, SIMD-group matrix math,
FP16/BF16 storage, persistent buffers, backward kernels, and PyTorch integration.
Routed weights use the same `[out_features, in_features]` layout as
`torch.nn.Linear`; the smaller shared expert is stored in one padded expert slot.

## Training forward

```python
import torch
from parrot import ModelConfig, Parrot
from parrot.tokenizer import load_tokenizer

tokenizer = load_tokenizer()
model = Parrot(ModelConfig(
    vocab_size=tokenizer.get_vocab_size(),
    checkpoint_modules=True,
)).cuda()  # Keep master parameters FP32; use BF16 autocast.

# ids: unpadded int64 [batch, source_tokens], on the same device as the model.
# Supply unshifted labels: the model handles the target shift internally.
ids = torch.randint(tokenizer.get_vocab_size(), (1, 64), device="cuda")
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(ids, labels=ids, bag_size=4)  # TST: 16 model positions
output.loss.backward()

# Switch to bag_size=1 for ordinary next-token training, using the same weights.
# For eight accumulation microsteps, backpropagate output.loss / 8 each time.
```

`output` contains logits, combined loss, language-model loss, unweighted router
auxiliary loss, and per-layer expert utilization. Without labels, the loss fields
are `None`. The last position has logits but no next-bag target in the supplied
stream. Targets equal to `-100` are ignored; this does **not** mask input attention.
Sequences must be unpadded, and source length must divide evenly by `bag_size`.
The 4,096-position limit permits 16,384 source tokens with `bag_size=4`.
The training loop selects the 30%/70% phase schedule; no optimizer or data pipeline
is included here.

## Architecture

- Layers 4, 8, 12, 16 use causal full-context NoPE MLA: direct queries, a joint
  128-wide KV projection, and separate key/value up-projections.
- Other layers use RoPE GQA, eight query heads and two KV heads, width 96,
  with a causal 2,048-position window including the current position.
- Every layer has 16 routed SwiGLU experts of width 512, top-2 normalized
  routing, and one shared expert of width 256. Only selected routed experts
  execute for a token; no capacity limit or token dropping is used.
- Block AttnRes sums four module outputs per depth block. Each module mixes
  the embedding, completed sums, and any current partial sum using a learned
  zero-initialized query and RMS-normalized keys. A final depth mixer precedes
  final RMSNorm and the tied embedding/output head.
- `ModelConfig(residual="standard")` selects the ordinary pre-norm residual
  baseline. Configurations otherwise share parameter names for copying weights
  with `load_state_dict(..., strict=False)`; only AttnRes keys should differ.
- Router balancing uses `n_experts * sum(assignment_fraction * mean_router_prob)`;
  assignment fractions include both selected experts and sum to one. The model
  averages this loss across layers and applies `aux_loss_weight` (default 0.01).

With 32,768 vocabulary entries, the implementation has **360,458,496 total** and
**96,217,344 active** parameters by the specification's counting convention.
This matches the default Mistral v0.3 vocabulary. Active count includes the whole tied vocabulary matrix
and all shared parameters, and excludes unselected routed experts; it is not a
FLOP count.

## Scope and performance

The ordinary forward pass supports training and prefill with backward support.
The incremental API is inference-only. MLA expands its latent into heads for
training SDPA, while incremental decode keeps the compressed latent and uses
the algebraically equivalent absorbed attention calculation.

Attention uses PyTorch scaled dot-product attention. Local attention processes
query chunks against only their relevant key range, with bounded boolean masks
instead of a full sequence-square mask. SDPA backend selection depends on the
device and dtype. The default training backend dispatches experts with a Python
loop. `training_cuda` sorts the top-2 assignments once and evaluates every
expert with three differentiable grouped GEMMs; the inference-only MPS path uses
the grouped Metal kernels described above.
Module checkpointing is optional. 4K training memory and throughput on a 4090
have not been measured; the starting batch size is not a fit guarantee.

References: [Block AttnRes equations 5–6](https://arxiv.org/pdf/2603.15031),
[DeepSeek MLA](https://arxiv.org/abs/2405.04434), and
[Nous token superposition](https://nousresearch.com/token-superposition).
The MLA layout and four-token TST objective here follow the supplied pilot spec.

## Metal training kernel

`ModelConfig(moe_backend="training_mps")` enables differentiable Metal kernels
for sparse expert grouping, SwiGLU, RMSNorm, Block AttnRes depth mixing, and
cross-entropy. Expert projections and attention use PyTorch's optimized MPS
matrix multiplication and SDPA. It supports MPS FP32, FP16, and BF16 with
first-order gradients. Use the default `depth_backend="pytorch"`; the separate
Metal inference depth backend remains inference-only.

```sh
python3 -m unittest discover -s tests -v
python3 benchmark_training.py --batch 512 --seq-len 96 --bag-size 4 \
  --compile --min-source-tokens-per-second 18000
python3 benchmark_training_roofline.py
swift run -c release --package-path native-metal parrot-metal-bench \
  --rows 16384 --dim 768 --dtype fp16 --engine mps \
  --output benchmark_native_metal_mps_fp16_results.json
python3 benchmark_training.py --backend pytorch \
  --output benchmark_training_pytorch_results.json
```

The benchmark includes the complete 360,458,496-parameter transformer's forward
pass, every supervised vocabulary projection, shifted language-model loss,
router auxiliary loss, and backward pass. The final position has no shifted
target, so its unused vocabulary projection is skipped unless `--full-logits`
is passed. The benchmark uses random tokens, synchronizes the GPU around each
sample, discards warmups, and checks gradients for finite values. Gradient
clearing, optimizer updates, and data loading are excluded. No optimizer choice
is needed for this measurement; optimizer state and updates affect full
training-step speed and memory.
BF16 results use BF16 parameters and gradients, without FP32 master parameters
or optimizer state; they do not establish long-run training stability.

Measured on the local 16-GPU-core M4 Pro with PyTorch 2.10.0, batch 512, 96
source tokens per sequence, `bag_size=4`, BF16 parameters/gradients,
TorchInductor compilation, and no checkpointing:

| Backend | Median forward + backward | Source tokens/sec | Model positions/sec |
| --- | ---: | ---: | ---: |
| `training_mps` | 2.660 s | **18,477.1** | 4,619.3 |

These are five synchronized timed samples after three warmups. All five losses
were identical, and the 18,000 source-token/sec gate passed. The model processes
four source tokens per model position in this TST phase, so both throughput
figures are reported explicitly. Raw samples are saved in
`benchmark_training_results.json`.

The active-parameter lower bound is about 577 million floating-point operations
per model position for forward and backward (`6 * 96,217,344`), before attention
and non-matrix work. At four source tokens per position, 80,000 source tokens/s
would therefore require at least 11.546 TFLOP/s sustained across the complete
training graph. The synchronized local MPS roofline reaches 5.052 TFLOP/s across
the forward, input-gradient, and weight-gradient GEMMs. That places the measured
software roof at 35,004 source tokens/s even if the model's attention, routing,
normalization, activations, loss, and memory copies took no time. The full raw
sweep is saved in `benchmark_training_roofline_results.json`.

The standalone Swift harness in `native-metal/` removes PyTorch and submits all
three training GEMMs in one command buffer. The older native MPS FP16 path
reaches 5.178 TFLOP/s over 15 timed samples, an optimistic 35,879
source-token/s math-only roof. The
macOS 26 cooperative-tensor path reaches 4.845 TFLOP/s in BF16 and 4.580 TFLOP/s
in FP16 with its documented 64-by-32 tile. Native submission therefore removes
little of the measured gap. A faster backend would need better shape-specific
GEMMs and fusion across projections and activations; replacing Python dispatch
alone is insufficient. Raw samples are saved in the corresponding
`benchmark_native_metal_*_results.json` files.

## H200 CUDA training kernel

`ModelConfig(moe_backend="training_cuda")` stores routed expert weights in
grouped-GEMM layout and uses exact, differentiable top-2 routing without token
dropping. `benchmark_cuda_training.py` compiles the complete model with
TorchInductor and times the forward pass, shifted cross-entropy, router
auxiliary loss, and backward pass with CUDA events. Gradient clearing, optimizer
updates, and data loading remain outside the timed region.

On a Runpod Secure Cloud NVIDIA H200 with PyTorch 2.8.0 and CUDA 12.8, batch 960,
512 source tokens, `bag_size=4`, BF16 parameters and gradients, and default
TorchInductor compilation, the median of seven runs after two warmups was
**0.49049 seconds**, or **1,002,107 source tokens/second** and **250,527 model
positions/second**. Peak allocated memory was 141.52 GB. The result is saved in
`benchmark_cuda_h200_results.json`.

`runpod-h200/` also packages the benchmark as a queue worker. Build the image for
`linux/amd64` from the repository root and deploy it on an H200 pool. See
`runpod-h200/README.md` for the image and request format.
