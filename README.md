# vLLM-ERNIE for NVIDIA Blackwell `sm_120` (Dual RTX PRO 6000)

Run **Baidu's unmodified [`baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)** (280B total / 14B active MoE, Apache-2.0, MXFP4 routed experts + 128×128 block FP8 dense/attention, 37 HCA + 6 MLA-DSA layers with attention sinks, bundled 3-layer DSpark draft head) on **2× NVIDIA RTX PRO 6000 Blackwell (`sm_120`, 2× 96 GB VRAM, PCIe Gen5, no NVLink)** using vLLM `0.32.0` + a surgical 164-line `sm_120` dispatch patch for [`baidu/vLLM-ERNIE`](https://github.com/baidu/vLLM-ERNIE).

> **Unmodified weights pattern:** This repository does **not** re-upload or modify the 155 GiB model checkpoint. Both Docker Compose recipes pull the official Apache-2.0 weights directly from `baidu/ERNIE-5.5-Flash-Preview` on Hugging Face.

---

## Two Tested Docker Compose Profiles

| Profile | Compose File | Context | CPU KV Offload | DSpark | Concurrency | `c=1` Decode | `c=4` Aggregate | Best For |
|---|---|---|---|---|---|---|---|---|
| **Profile A — Fast In-VRAM** | [`compose.in-vram.yml`](compose.in-vram.yml) | **131,072** (201K KV pool) | Off | **On** (`7` draft tok, `2.71` tok/step) | `8` seqs | **93.2 tok/s** (`135 ms` TTFT) | **211.6 tok/s** (`58.0`/stream) | Fast daily coding, multi-agent bursts, low host RAM usage |
| **Profile B — 1M Long Context** | [`compose.long-context.yml`](compose.long-context.yml) | **1,048,576** (1.06M KV pool) | **On** (`16 GiB/rank` default, configurable) | Off (`+5.9 GiB/GPU` KV) | `2` seqs | **82.3 tok/s** (`118 ms` TTFT) | — | Full-repo / 500K–1M contexts, multi-session prefix caching in host RAM (`59.6×` reload speedup) |

---

## Hardware & Software Requirements

- **GPUs:** 2× NVIDIA RTX PRO 6000 Blackwell (Max-Q 300 W or Workstation 600 W, `sm_120`, 96 GB VRAM each, PCIe Gen5)
- **NVIDIA Driver:** Open kernel module driver `>= 580` (tested on `595.99.02` with CUDA `13.0` container runtime)
- **Host Kernel:** `iommu=pt` recommended for multi-GPU PCIe P2P / NCCL
- **Host RAM:**
  - Profile A (`compose.in-vram.yml`): **64 GB+** (~20 GB used)
  - Profile B (`compose.long-context.yml`): **128 GB+** (defaults to `16 GiB/rank` = `32 GiB` CPU KV cache; scale up to `64–128 GiB/rank` on 256–512 GB hosts via `CPU_KV_BYTES_PER_RANK`)
- **Disk:** ~165 GB free for the Hugging Face checkpoint (`155 GiB`) + Triton/FlashInfer/DeepGEMM JIT kernel cache
- **Software:** Docker Engine + NVIDIA Container Toolkit (`docker compose` v2)

---

## Quick Start

### 1. Clone and Pick a Profile

```bash
git clone https://github.com/milesangelo/vllm-ernie55-sm120.git
cd vllm-ernie55-sm120
mkdir -p models cache
```

*(Optional)* Pre-download the weights if you want progress visibility before starting vLLM:

```bash
pip install -U huggingface_hub
hf download baidu/ERNIE-5.5-Flash-Preview --local-dir ./models/hub/models--baidu--ERNIE-5.5-Flash-Preview
```

### 2. Start Profile A (131K Fast In-VRAM + DSpark)

```bash
# Pull prebuilt image from GHCR (or add --build to build locally from Dockerfile):
docker compose -f compose.in-vram.yml up -d
docker compose -f compose.in-vram.yml logs -f
```

### 3. Or Start Profile B (1M Context + `SimpleCPUOffloadConnector`)

```bash
# 128 GB host RAM (safe default: 16 GiB/rank = 32 GiB total CPU KV cache):
docker compose -f compose.long-context.yml up -d

# 256 GB host RAM (64 GiB/rank = 128 GiB total CPU KV cache):
CPU_KV_BYTES_PER_RANK=68719476736 docker compose -f compose.long-context.yml up -d

# 512 GB host RAM (128 GiB/rank = 256 GiB total CPU KV cache):
CPU_KV_BYTES_PER_RANK=137438953472 docker compose -f compose.long-context.yml up -d
```

> **First boot note:** Cold start compiles Triton, FlashInfer, and DeepGEMM `sm_120` kernels (10–25 min depending on CPU). Subsequent warm boots with the `./cache` volume mounted take **~3 minutes**.

### 4. Verify & Benchmark

```bash
# Profile A (131K + DSpark):
python3 scripts/verify_and_bench.py --base-url http://127.0.0.1:8000 --niah-k 115 --concurrency 4

# Profile B (1M + CPU offload eviction & reload):
python3 scripts/verify_and_bench.py --base-url http://127.0.0.1:8000 --concurrency 1 --cpu-offload-test --niah-k 950
```

---

## Measured Benchmarks (2× RTX PRO 6000 Blackwell Max-Q, 300 W Cap)

Tested on Threadripper 7960X, 128 GB RAM (`123 GiB` usable), PCIe Gen5 (no NVLink, `iommu=pt`), NVIDIA Driver `595.99.02`, vLLM `0.32.0`, `baidu/vLLM-ERNIE@ba322a56` + [`patches/vllm-ernie-sm120.patch`](patches/vllm-ernie-sm120.patch).

### Profile A — `compose.in-vram.yml` (`131,072` ctx, FP8 KV, DSpark `k=7`, `FULL_DECODE_ONLY`, `util=0.95`)

| Metric | Measured Result |
|---|---|
| Model weights + DSpark | `80.0 GiB / GPU` (`81.3 GiB` incl. non-torch), `DEEPGEMM_MXFP4` + `DeepGemmFp8BlockScaledMMKernel` |
| GPU FP8 KV cache | `5.66 GiB / GPU` → **`201,115` tokens** (`1.53×` @ 131K) |
| Warm startup | `3m 04s` (`170.8s` load + `13.4s` init/CUDA graphs) |
| `c=1` decode (5× 512 tok, thinking off) | **93.2 tok/s median** (`91.1–100.0`), **135 ms** TTFT |
| `c=4` concurrent decode (4× 256 tok) | **211.6 tok/s aggregate** (`58.0 tok/s` median/stream) |
| DSpark speculative acceptance | **2.71 tok/step** (`24.5%` across 1,392 drafts) |
| 115K NIAH (`115,040` tok) cold prefill | **PASS** in `21.03s` (**5,471 tok/s** prefill) |
| 115K NIAH 2nd-turn prefix-cache hit | **PASS** in **0.48s** (**44× faster**, `114,688` `cached_tokens`) |
| `reasoning_effort` (`low`/`medium`/`high`) & `ernie5` tool calling | **PASS** (`0.30–0.36s`; invalid effort → HTTP `400`) |
| Peak VRAM / Host RAM / Stability | `95.5 / 97.9 GiB` VRAM, `20 / 123 GiB` host RAM, `restarts=0`, zero Xids |

### Profile B — `compose.long-context.yml` (`1,048,576` ctx, FP8 KV, `SimpleCPUOffloadConnector 16 GiB/rank`, DSpark off, `util=0.98`)

| Metric | Measured Result |
|---|---|
| Model weights (DSpark off) | `74.08 GiB / GPU` (`+5.9 GiB/GPU` reclaimed for KV) |
| GPU FP8 KV cache | `14.17 GiB / GPU` → **`1,059,602` tokens** (`1.01×` @ 1M) |
| CPU KV offload pool | `16.00 GiB / rank` (`45,466` blocks/rank, `32 GiB` total) |
| Warm startup | `3m 15s` (`174.6s` load + `20.9s` init/CUDA graphs) |
| `c=1` decode (3× 512 tok, thinking off) | **82.3 tok/s median** (`[82.3, 82.3, 82.3]`), **118 ms** TTFT |
| 462K NIAH (Session A, `462,263` tok) | **PASS** in `126.29s` (**3,660 tok/s** prefill) |
| 600K NIAH (Session B evictor, `600,052` tok) | **PASS** in `181.49s` (**3,306 tok/s** prefill, evicts Session A to CPU RAM) |
| **Session A 462K CPU-Offload Reload** | **PASS** in **2.12s** (**59.6× faster** than cold prefill; `462,080` `cached_tokens` reloaded from host RAM over PCIe!) |
| 833K NIAH (`833,384` tok) | **PASS** in `299.33s` (**2,784 tok/s** prefill) |
| 878K NIAH (`877,559` tok) | **PASS** in `323.97s` (**2,709 tok/s** prefill) |
| `reasoning_effort` (`low`/`medium`/`high`) & `ernie5` tool calling | **PASS** (`0.28–0.70s`; invalid effort → HTTP `400`) |
| Peak VRAM / Host RAM / Stability | `96.8 / 97.9 GiB` VRAM, `52 / 123 GiB` host RAM used (`70 GiB` avail), `restarts=0`, zero Xids |

---

## What `patches/vllm-ernie-sm120.patch` Fixes & Why Each Flag Matters

1. **`sm_120` HCA & Sparse-MLA Dispatch (`patches/vllm-ernie-sm120.patch`):**
   Upstream `baidu/vLLM-ERNIE@ba322a56` only dispatched Hopper (`sm_90`) and Blackwell Datacenter (`sm_100`). vLLM `0.32.0` already ships `DeepseekV4FlashInferSM120Attention` and `FlashInferMLASparseSM120Backend` (whose FlashInfer `SparseMLASm120Wrapper` natively accepts `sinks=[self.sinks]`). Our 164-line patch wires all 37 HCA layers (+ DSpark draft layers) and all 6 DSA layers to these native `sm_120` kernels.
2. **`--disable-custom-all-reduce`:**
   Dual PCIe Blackwell cards without NVLink hit `cudaErrorIllegalAddress` in `custom_all_reduce.cuh:164` during CUDA graph capture. Disabling custom all-reduce routes TP2 collectives through PyNCCL cleanly.
3. **`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`:**
   Breakable piecewise CUDA graph replay faults on short prefill prompts (`<= 64` tokens, e.g. `"hi"`). `FULL_DECODE_ONLY` keeps prefills eager and captures FULL CUDA graphs for decode (`--max-cudagraph-capture-size 64` for `c=1..8` with DSpark, or `4` for `c=1..2` without DSpark).
4. **No `--attention-config '{"indexer_kv_dtype":"mxfp4"}'`:**
   MXFP4 indexer KV is `sm_100`-only in vLLM `0.32.0`; omitting it keeps the native FP8 Lightning Indexer.
5. **No `--moe-backend humming`:**
   `humming` is intended for FP8 checkpoints; omitting `--moe-backend` on the official MXFP4 checkpoint lets vLLM auto-select native Blackwell `DEEPGEMM_MXFP4`.
6. **`HF_MODULES_CACHE=/cache/hf-modules`:**
   Allows mounting your Hugging Face model cache read-only (`:ro`) while `--trust-remote-code` caches ERNIE's tokenizer modules in `/cache`.
7. **`PYTORCH_CUDA_ALLOC_CONF=""`:**
   Avoids `expandable_segments` warnings near high GPU memory utilization and stays compatible with `SimpleCPUOffloadConnector`.
8. **How `SimpleCPUOffloadConnector` works in Profile B:**
   Active request KV still lives in GPU VRAM (possible up to `1,048,576` tokens on 2× 96 GB cards thanks to ERNIE-5.5's 37 HCA 128:1 compressed layers + only 6 full MLA-DSA layers, totaling ~`14.17 GiB/GPU` for `1.06M` FP8 tokens). `SimpleCPUOffloadConnector` transparently offloads evicted/inactive session prefixes into pinned host RAM so switching between multiple massive sessions reloads in ~2s instead of minutes.

---

## LiteLLM & API Notes

- See [`configs/litellm_example.yaml`](configs/litellm_example.yaml) for a drop-in LiteLLM Proxy entry.
- ERNIE-5.5 supports exact `reasoning_effort` values: `"low"`, `"medium"`, `"high"` (defaults to `"high"`; any other value raises HTTP `400`).
- Disable chain-of-thought per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- `--enable-prompt-tokens-details` enables standard OpenAI `usage.prompt_tokens_details.cached_tokens` reporting.

---

## Community & Upstream

- Copy-pasteable Hugging Face Community Discussion guide: [`docs/hf_community_post.md`](docs/hf_community_post.md)
- Upstream PR description for `baidu/vLLM-ERNIE`: [`docs/upstream_pr.md`](docs/upstream_pr.md)

## License

Apache-2.0 (matching both `baidu/ERNIE-5.5-Flash-Preview` and `baidu/vLLM-ERNIE`).
