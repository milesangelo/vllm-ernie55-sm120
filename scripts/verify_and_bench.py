#!/usr/bin/env python3
"""Verification and benchmark suite for baidu/ERNIE-5.5-Flash-Preview on vLLM.

Covers:
  1. Short-prompt stability (regression test for piecewise CUDA graph faults)
  2. reasoning_effort ("low", "medium", "high") + invalid-effort HTTP 400 check
  3. ernie5 tool-calling parser verification
  4. Single-stream (c=1) decode tok/s & TTFT (streaming)
  5. Optional concurrent (c=4) aggregate decode tok/s
  6. Optional Needle-in-a-Haystack (NIAH) + prefix-cache hit speedup
  7. Optional CPU-offload eviction + reload benchmark (for compose.long-context.yml)

Example usage:
  # Profile A (131K in-VRAM + DSpark):
  python3 scripts/verify_and_bench.py --base-url http://127.0.0.1:8000 --niah-k 115 --concurrency 4

  # Profile B (Long-context + SimpleCPUOffloadConnector):
  python3 scripts/verify_and_bench.py --base-url http://127.0.0.1:8000 --concurrency 1 --cpu-offload-test --niah-k 950
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def post_json(base_url: str, path: str, payload: dict, timeout: int = 900) -> tuple[dict, float]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp), time.perf_counter() - t0


def stream_bench(base_url: str, model: str, prompt: str, max_tokens: int = 512) -> tuple[float, float, int, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and (choices[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    comp = int(usage.get("completion_tokens", 0))
    dec = max(total - (ttft or 0.0), 1e-6)
    return (ttft or total), dec, comp, comp / dec


def make_haystack(sentence: str, target_k: int, code: str, depth: float) -> str:
    # ~18 tokens per filler sentence for ERNIE-5.5 tokenizer
    n_sent = max(1, int(target_k * 1000 / 18))
    idx = int(n_sent * depth)
    needle = f"IMPORTANT ARCHIVE PASSKEY: The secret code is {code}. Remember it verbatim. "
    return (sentence * idx) + needle + (sentence * (n_sent - idx))


def run_niah(base_url: str, model: str, sentence: str, target_k: int, code: str, depth: float, label: str) -> tuple[str, dict, float, bool]:
    hay = make_haystack(sentence, target_k, code, depth)
    resp, dt = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": hay + "\n\nWhat is the exact secret code? Reply with the code only.",
                }
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 48,
        },
        timeout=900,
    )
    ans = (resp["choices"][0]["message"].get("content") or "").strip()
    ptok = resp["usage"]["prompt_tokens"]
    ok = code in ans
    print(f"[{label}] prompt_tokens={ptok:,} dt={dt:.2f}s ({ptok / dt:,.0f} tok/s prefill) PASS={ok} answer={ans!r}")
    return hay, resp["usage"], dt, ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify and benchmark ERNIE-5.5-Flash-Preview on vLLM")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="ernie-5.5-flash-preview")
    ap.add_argument("--runs", type=int, default=5, help="Single-stream decode runs")
    ap.add_argument("--concurrency", type=int, default=4, help="Concurrent decode test streams (set 1 to skip)")
    ap.add_argument("--niah-k", type=int, default=115, help="Target NIAH prompt size in thousands of tokens (0 to skip)")
    ap.add_argument("--cpu-offload-test", action="store_true", help="Run 520K -> 600K eviction -> 520K CPU reload test")
    args = ap.parse_args()

    # 1. Short prompt
    resp, dt = post_json(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 32,
        },
    )
    print(f"[ShortPrompt] dt={dt:.2f}s content={resp['choices'][0]['message'].get('content')!r}")

    # 2. Reasoning effort levels + negative test
    for effort in ("low", "medium", "high"):
        resp, dt = post_json(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
                "reasoning_effort": effort,
                "temperature": 0,
                "max_tokens": 256,
            },
        )
        msg = resp["choices"][0]["message"]
        has_reasoning = bool(msg.get("reasoning") or msg.get("reasoning_content"))
        print(f"[Effort={effort}] dt={dt:.2f}s has_reasoning={has_reasoning} content={msg.get('content')!r}")

    try:
        post_json(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "invalid_level",
                "max_tokens": 16,
            },
        )
        print("[Effort=invalid] UNEXPECTED 200 OK")
        return 1
    except urllib.error.HTTPError as exc:
        print(f"[Effort=invalid] expected HTTP {exc.code} PASS={exc.code == 400}")

    # 3. Tool calling
    resp, dt = post_json(
        args.base_url,
        "/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Read /etc/hostname."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "reasoning_effort": "low",
            "temperature": 0,
            "max_tokens": 256,
        },
    )
    tool_calls = resp["choices"][0]["message"].get("tool_calls") or []
    print(f"[ToolCall] dt={dt:.2f}s PASS={len(tool_calls) > 0} tool_calls={tool_calls}")

    # 4. Single-stream decode (c=1)
    stream_bench(args.base_url, args.model, "Warmup: explain PCIe Gen5 briefly.", 128)
    r1 = [
        stream_bench(
            args.base_url,
            args.model,
            f"Run {i}: Explain hybrid compressed attention, sparse MLA indexing, and speculative decoding in detail.",
            512,
        )
        for i in range(args.runs)
    ]
    print(
        f"[c=1 decode] median={statistics.median(x[3] for x in r1):.1f} tok/s "
        f"(runs={[round(x[3], 1) for x in r1]}) median_ttft={statistics.median(x[0] for x in r1) * 1000:.0f}ms"
    )

    # 5. Concurrent decode (optional)
    if args.concurrency > 1:
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [
                ex.submit(
                    stream_bench,
                    args.base_url,
                    args.model,
                    f"Stream {i}: Write a detailed explanation of distributed KV caching and tensor parallelism.",
                    256,
                )
                for i in range(args.concurrency)
            ]
            rc = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        total_tok = sum(x[2] for x in rc)
        print(
            f"[c={args.concurrency} decode] aggregate={total_tok / wall:.1f} tok/s "
            f"per_stream_median={statistics.median(x[3] for x in rc):.1f} tok/s wall={wall:.2f}s"
        )

    # 6. Optional CPU-offload eviction + reload test (520K -> 600K evictor -> 520K reload)
    if args.cpu_offload_test:
        hay_a, _, _, ok_a = run_niah(
            args.base_url,
            args.model,
            "Alpha sector telemetry log recorded nominal thermal readings across compute rack modules during overnight inspection. ",
            520,
            "ALPHA-520K-7731",
            0.37,
            "NIAH 520K Session A",
        )
        _, _, _, ok_b = run_niah(
            args.base_url,
            args.model,
            "Bravo datacenter storage audit verified clean NVMe checksums across archival snapshots during scheduled verification. ",
            600,
            "BRAVO-600K-9142",
            0.63,
            "NIAH 600K Session B (evictor)",
        )
        resp_a2, dt_a2 = post_json(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": hay_a + "\n\nRepeat the exact secret code once more.",
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_tokens": 48,
            },
            timeout=600,
        )
        ans_a2 = (resp_a2["choices"][0]["message"].get("content") or "").strip()
        ok_a2 = "ALPHA-520K-7731" in ans_a2
        print(
            f"[Session A 520K CPU-Offload Reload] dt={dt_a2:.2f}s PASS={ok_a2} "
            f"usage={resp_a2['usage']} answer={ans_a2!r}"
        )
        if not (ok_a and ok_b and ok_a2):
            return 1

    # 7. Primary NIAH + prefix-cache hit check
    if args.niah_k > 0:
        hay, _, dt_cold, ok_cold = run_niah(
            args.base_url,
            args.model,
            "Observatory instrument log recorded steady atmospheric calibration readings across every optical array channel. ",
            args.niah_k,
            f"PASSKEY-{args.niah_k}K-2026",
            0.51,
            f"NIAH {args.niah_k}K Cold",
        )
        resp_hit, dt_hit = post_json(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": hay + "\n\nRepeat the exact secret code once more.",
                    }
                ],
                "chat_template_kwargs": {"enable_thinking": False},
                "temperature": 0,
                "max_tokens": 48,
            },
            timeout=600,
        )
        ans_hit = (resp_hit["choices"][0]["message"].get("content") or "").strip()
        ok_hit = f"PASSKEY-{args.niah_k}K-2026" in ans_hit
        print(
            f"[NIAH {args.niah_k}K Prefix Hit] dt={dt_hit:.2f}s ({dt_cold / max(dt_hit, 1e-6):.1f}x faster) "
            f"PASS={ok_hit} usage={resp_hit['usage']} answer={ans_hit!r}"
        )
        if not (ok_cold and ok_hit):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
