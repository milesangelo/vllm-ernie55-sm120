# Running `baidu/ERNIE-5.5-Flash-Preview` on Dual RTX PRO 6000 Blackwell (`sm_120`) — Up to 1M Context + CPU Prefix Offloading (vLLM 0.32.0)

*Ready-to-post discussion for the [Hugging Face Community tab on `baidu/ERNIE-5.5-Flash-Preview`](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview/discussions).*

---

For anyone trying to run the unmodified official **`baidu/ERNIE-5.5-Flash-Preview`** weights on **2× NVIDIA RTX PRO 6000 Blackwell (`sm_120`, 2× 96 GB VRAM, PCIe Gen5, no NVLink)**, here is a tested Docker setup, a minimal 164-line `sm_120` patch for `baidu/vLLM-ERNIE`, and measured benchmarks for **two profiles**:

- **Profile A (Fast In-VRAM + DSpark, 131K context):** **93.2 tok/s** single-stream (`135 ms` TTFT), **211.6 tok/s** aggregate at `c=4`, `5,471 tok/s` 115K NIAH prefill, `0.48s` prefix-cache hit (`44×` faster).
- **Profile B (1M Context + `SimpleCPUOffloadConnector`, DSpark off):** **`1,048,576` max context** (`1,059,602` GPU FP8 KV tokens in `14.17 GiB/GPU`), **82.3 tok/s** single-stream (`118 ms` TTFT), **462K → 600K GPU eviction → 462K CPU-RAM reload in `2.12s` (`59.6×` faster than cold prefill!)**, and **833K / 878K NIAH 100% passkey recall** with zero restarts and zero Xid errors.

**Repository & Prebuilt Image (Apache-2.0, pulls official unmodified weights directly from this HF repo):**
- GitHub: `https://github.com/milesangelo/vllm-ernie55-sm120`
- Prebuilt GHCR image: `ghcr.io/milesangelo/vllm-ernie55-sm120:v0.32.0`

---

### Why `baidu/vLLM-ERNIE` Didn't Work Out of the Box on `sm_120`

1. Upstream `baidu/vLLM-ERNIE@ba322a56` only dispatched Hopper (`sm_90`) and Blackwell Datacenter (`sm_100`).
2. `vllm/vllm-openai:v0.32.0` **already ships** `DeepseekV4FlashInferSM120Attention` and `FlashInferMLASparseSM120Backend` (whose FlashInfer `SparseMLASm120Wrapper` natively supports `attn_sink`!).
3. Our 164-line patch (`patches/vllm-ernie-sm120.patch` in the repo) simply wires:
   - all 37 HCA layers (+ DSpark draft layers) to `Ernie5_5HCAFlashInferSM120Attention`, and
   - all 6 DSA layers to `Ernie5_5FlashInferMLASparseSM120Backend` passing `sinks=[self.sinks]`.

### Key PCIe Dual-Blackwell (`sm_120`) Gotchas Solved

- **`--disable-custom-all-reduce`:** Prevents `cudaErrorIllegalAddress` in `custom_all_reduce.cuh:164` during CUDA graph capture on non-NVLink PCIe cards.
- **`--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`:** Fixes a breakable piecewise CUDA graph replay fault on short prompts (`<= 64` tokens, e.g. `"hi"`) while keeping decode in FULL CUDA graphs (`capture_size=64` for `c=1..8` + DSpark).
- **Omit `--attention-config '{"indexer_kv_dtype":"mxfp4"}'`:** MXFP4 indexer KV is `sm_100`-only in vLLM `0.32.0`; omitting it uses the FP8 Lightning Indexer.
- **Omit `--moe-backend humming`:** Letting vLLM auto-select on the official MXFP4 checkpoint enables native Blackwell `DEEPGEMM_MXFP4`.
- **`PYTORCH_CUDA_ALLOC_CONF=""`:** Required for `SimpleCPUOffloadConnector` compatibility and clean high-utilization startup.

### Measured Results (2× RTX PRO 6000 Max-Q, 300 W Cap, 128 GB Host RAM)

| Profile | Context | KV Pool | `c=1` Decode | `c=4` Aggregate | Long-Context NIAH | Prefix Reload |
|---|---|---|---|---|---|---|
| **Profile A** (`compose.in-vram.yml`, DSpark `k=7`) | `131,072` | `201,115` GPU tok | **93.2 tok/s** (`135 ms` TTFT) | **211.6 tok/s** | 115K: **PASS** (`21.0s`, `5,471 tok/s`) | GPU hit: **0.48s** (`44×`) |
| **Profile B** (`compose.long-context.yml`, `SimpleCPUOffloadConnector 16 GiB/rank`) | `1,048,576` | `1,059,602` GPU tok + `32 GiB` CPU RAM | **82.3 tok/s** (`118 ms` TTFT) | — (`max_num_seqs=2`) | 462K / 600K / 833K / 878K: **ALL PASS** (`2,709–3,660 tok/s`) | **CPU RAM reload (462K after GPU eviction): `2.12s` (`59.6×`)** |

Both profiles also pass `reasoning_effort` (`low` / `medium` / `high` + HTTP `400` on invalid values) and `ernie5` function/tool calling.

### Quick Start

```bash
git clone https://github.com/milesangelo/vllm-ernie55-sm120.git
cd vllm-ernie55-sm120

# Option 1: 131K fast in-VRAM + DSpark (93 tok/s c=1, 212 tok/s c=4)
docker compose -f compose.in-vram.yml up -d

# Option 2: 1M context + CPU KV prefix offloading (tune CPU_KV_BYTES_PER_RANK for your RAM)
docker compose -f compose.long-context.yml up -d
```

Hope this saves other RTX PRO 6000 / Blackwell workstation owners some debugging time! Feedback and upstream review welcome.
