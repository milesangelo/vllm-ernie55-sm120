# Add NVIDIA Blackwell `sm_120` (RTX PRO 6000 / RTX 50-series) Support for ERNIE-5.5 HCA & Sparse-MLA Attention

*Ready-to-open Pull Request description for [`baidu/vLLM-ERNIE`](https://github.com/baidu/vLLM-ERNIE) using [`patches/vllm-ernie-sm120.patch`](../patches/vllm-ernie-sm120.patch).*

---

## Summary

Adds native NVIDIA Blackwell `sm_120` (RTX PRO 6000 Blackwell / GeForce RTX 50-series) attention dispatch for **ERNIE-5.5** (`Ernie5_5HCA` and `Ernie5_5DSA`) on vLLM `0.32.0`, without changing any existing `sm_90` or `sm_100` code paths.

## Motivation

Currently (`ba322a56`), `vllm_ernie/models/ernie5_5/nvidia/attention.py` only selects Hopper (`sm_90`) and Blackwell Datacenter (`sm_100`) HCA classes, and `Ernie5_5DSAAttention` assumes `FLASHMLA_SPARSE` (which rejects `sm_120`).

However, `vllm==0.32.0` already ships both `sm_120` primitives needed by ERNIE-5.5:
1. `DeepseekV4FlashInferSM120Attention` (for the 37 HCA layers and bundled DSpark draft layers)
2. `FlashInferMLASparseSM120Backend` / `FlashInferMLASparseSM120Impl` wrapping FlashInfer's `SparseMLASm120Wrapper`, which natively supports `attn_sink` (`sinks=[self.sinks]`).

## Changes

- **`vllm_ernie/models/ernie5_5/nvidia/attention.py`** (`patches/vllm-ernie-sm120.patch`):
  - Adds `Ernie5_5HCAFlashInferSM120Attention(Ernie5_5HCAAttentionMixin, DeepseekV4FlashInferSM120Attention)` and dispatches to it in `select_ernie5_5_hca_cls()` when `current_platform.is_device_capability_family(120)` is true.
  - Adds `Ernie5_5FlashInferMLASparseSM120Impl` and `Ernie5_5FlashInferMLASparseSM120Backend`, passing `sinks=[self.sinks]` directly into FlashInfer's SM120 sparse-MLA kernel.
  - Dispatches `Ernie5_5DSAAttention` to `Ernie5_5FlashInferMLASparseSM120Backend` on `sm_120` while preserving `FLASHMLA_SPARSE` on `sm_90`/`sm_100`.
  - Guards `sm_120` imports gracefully so older vLLM versions without SM120 classes continue importing cleanly.

## Verification

Tested end-to-end on **2× NVIDIA RTX PRO 6000 Blackwell Max-Q (`sm_120`, 96 GB each, PCIe Gen5, no NVLink)** with `vllm/vllm-openai:v0.32.0-x86_64`, CUDA `13.0`, PyTorch `2.13.0+cu130`, Driver `595.99.02`, TP2, and unmodified `baidu/ERNIE-5.5-Flash-Preview` (`ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe`):

1. **Profile A (`131,072` context, FP8 KV, bundled 7-token DSpark, `FULL_DECODE_ONLY`, `--disable-custom-all-reduce`, `util=0.95`):**
   - Native `DEEPGEMM_MXFP4` + `DeepGemmFp8BlockScaledMMKernel` selected automatically.
   - `80.0 GiB/GPU` weights+DSpark, `5.66 GiB/GPU` KV (`201,115` tokens).
   - Short prompts, `reasoning_effort` (`low`/`medium`/`high` + HTTP `400` on invalid), and `ernie5` tool calling: **ALL PASS**.
   - `c=1` decode: **93.2 tok/s median** (`135 ms` TTFT); `c=4` aggregate: **211.6 tok/s**; DSpark acceptance: **2.71 tok/step**.
   - 115K NIAH (`115,040` tok): **PASS** (`21.03s`, `5,471 tok/s`), 2nd-turn prefix-cache hit: `0.48s` (`44×`).
2. **Profile B (`1,048,576` context, FP8 KV, `SimpleCPUOffloadConnector 16 GiB/rank`, DSpark off, `util=0.98`):**
   - `74.08 GiB/GPU` weights, `14.17 GiB/GPU` GPU KV (`1,059,602` tokens), `16 GiB/rank` CPU offload.
   - `c=1` decode: **82.3 tok/s median** (`118 ms` TTFT).
   - 462K NIAH (`126.3s`), 600K evictor NIAH (`181.5s`), 462K CPU-offload reload (**`2.12s`**, `59.6×` speedup, `462,080` `cached_tokens`), 833K NIAH (`299.3s`), and 878K NIAH (`324.0s`): **ALL PASS**, `restarts=0`, zero `NVRM: Xid`.

## Notes for Reviewers

- `--attention-config '{"indexer_kv_dtype":"mxfp4"}'` remains `sm_100`-only in vLLM `0.32.0`; `sm_120` uses the default FP8 Lightning Indexer.
- Non-NVLink dual PCIe `sm_120` setups require `--disable-custom-all-reduce` and `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` for clean CUDA graph capture and short-prompt replay.
