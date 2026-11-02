#!/usr/bin/env python3
"""Benchmark a Whisper transcription endpoint (OpenAI-compatible /v1/audio/transcriptions).

Metrics follow the Open ASR Leaderboard (arXiv:2510.06961):
  * WER  - corpus word error rate after Whisper's EnglishTextNormalizer
  * RTFx - audio seconds transcribed / wall-clock seconds (higher = faster)
plus serving metrics per concurrency level: req/s, latency and RTF p50/p95, failed requests.

Example:
  export ASR_API_KEY=<token>
  python asr_bench.py --endpoint https://whisper.example.com/v1 --model openai/whisper-large-v3-turbo \
      --num-samples 100 --concurrency 1,2,4,8
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jiwer
import numpy as np
import requests
import soundfile as sf
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

# ----------------------------------------------------------------------------- data


def load_hf(dataset: str, config: str, split: str, text_col: str, n: int) -> list[dict]:
    """Stream n samples from a Hugging Face dataset (default: LibriSpeech test-clean)."""
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))  # keep the original encoded bytes
    samples = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        raw = item["audio"]
        data = raw["bytes"] if raw.get("bytes") else Path(raw["path"]).read_bytes()
        name = Path(raw.get("path") or f"{i}.flac").name
        samples.append(
            {"name": name, "data": data, "duration": sf.info(io.BytesIO(data)).duration, "text": item[text_col]}
        )
    return samples


def load_manifest(path: str, n: int) -> list[dict]:
    """JSONL lines: {"audio": "path/to/file.wav", "text": "reference"}; paths relative to the manifest."""
    base = Path(path).parent
    samples = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if len(samples) >= n:
            break
        row = json.loads(line)
        audio_path = base / row["audio"]
        samples.append(
            {
                "name": audio_path.name,
                "data": audio_path.read_bytes(),
                "duration": sf.info(str(audio_path)).duration,
                "text": row.get("text"),
            }
        )
    return samples


# ----------------------------------------------------------------------------- endpoint


class EndpointClient:
    """POST {endpoint}/v1/audio/transcriptions with a Bearer token."""

    def __init__(self, args) -> None:
        url = args.endpoint.rstrip("/")
        if not url.endswith("/audio/transcriptions"):
            url += "/audio/transcriptions" if url.endswith("/v1") else "/v1/audio/transcriptions"
        self.url = url
        self.args = args
        self.session = requests.Session()
        self.session.verify = not args.insecure
        self.session.headers["Authorization"] = f"Bearer {args.api_key}"

    def transcribe(self, sample: dict) -> str:
        data = {"model": self.args.model, "response_format": "json", "temperature": "0"}
        if self.args.language:
            data["language"] = self.args.language
        resp = self.session.post(
            self.url,
            files={"file": (sample["name"], sample["data"])},
            data=data,
            timeout=self.args.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()["text"].strip()


# ----------------------------------------------------------------------------- bench


def pct(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else float("nan")


def run_level(client: EndpointClient, samples: list[dict], concurrency: int, args, normalize) -> dict:
    """Send every sample once with `concurrency` requests in flight."""
    for _ in range(args.warmup):
        client.transcribe(samples[0])

    def timed(s: dict) -> tuple[float, str | None, str | None]:
        t0 = time.perf_counter()
        try:
            hyp = client.transcribe(s)
            return time.perf_counter() - t0, hyp, None
        except Exception as exc:  # count failures instead of aborting the sweep
            return time.perf_counter() - t0, None, f"{type(exc).__name__}: {exc}"[:200]

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out = list(pool.map(timed, samples))
    wall = time.perf_counter() - t_start

    ok = [i for i, (_, hyp, _) in enumerate(out) if hyp is not None]
    errors = [err for _, _, err in out if err]
    latencies = [out[i][0] for i in ok]
    rtf = [out[i][0] / samples[i]["duration"] for i in ok]
    audio_ok = sum(samples[i]["duration"] for i in ok)
    pairs = [(normalize(samples[i]["text"]), normalize(out[i][1])) for i in ok if samples[i]["text"]]
    pairs = [(r, h) for r, h in pairs if r]  # skip empty references
    wer = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]) * 100 if pairs else None
    return {
        "concurrency": concurrency,
        "requests": len(samples),
        "failed": len(errors),
        "errors": sorted(set(errors))[:5],
        "wall_s": round(wall, 2),
        "throughput_rps": round(len(ok) / wall, 3),
        "rtfx": round(audio_ok / wall, 3),
        "latency_p50_s": round(pct(latencies, 50), 3),
        "latency_p95_s": round(pct(latencies, 95), 3),
        "rtf_p50": round(pct(rtf, 50), 4),
        "rtf_p95": round(pct(rtf, 95), 4),
        "wer_pct": round(wer, 2) if wer is not None else None,
        "scored_files": len(pairs),
        "examples": [{"ref": samples[i]["text"], "hyp": out[i][1]} for i in ok[:3]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark a Whisper transcription endpoint (WER + RTFx).")
    ap.add_argument("--endpoint", required=True, help="base URL, e.g. https://host/v1")
    ap.add_argument("--model", required=True, help="model name served by the endpoint")
    ap.add_argument("--api-key", default=os.environ.get("ASR_API_KEY"), help="Bearer token (default: $ASR_API_KEY)")
    ap.add_argument("--concurrency", default="1", help="comma-separated concurrent requests, e.g. 1,2,4,8")
    ap.add_argument("--manifest", help="JSONL manifest of local audio; default is a Hugging Face dataset")
    ap.add_argument("--hf-dataset", default="hf-audio/esb-datasets-test-only-sorted")
    ap.add_argument("--hf-config", default="librispeech")
    ap.add_argument("--hf-split", default="test.clean")
    ap.add_argument("--text-column", default="text")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--language", default="en", help="language code, or 'auto' for detection")
    ap.add_argument("--warmup", type=int, default=1, help="untimed requests before each concurrency level")
    ap.add_argument("--timeout", type=float, default=600, help="per-request timeout (s)")
    ap.add_argument("--insecure", action="store_true", help="skip TLS verification (self-signed certs)")
    ap.add_argument("--output", default="results.json")
    args = ap.parse_args()
    if not args.api_key:
        ap.error("no token: pass --api-key or set ASR_API_KEY")
    args.language = None if args.language == "auto" else args.language
    levels = [int(c) for c in args.concurrency.split(",")]

    print("loading data ...")
    if args.manifest:
        samples = load_manifest(args.manifest, args.num_samples)
    else:
        samples = load_hf(args.hf_dataset, args.hf_config, args.hf_split, args.text_column, args.num_samples)
    total_audio = sum(s["duration"] for s in samples)
    print(f"{len(samples)} files, {total_audio:.1f} s audio")

    client = EndpointClient(args)
    t0 = time.perf_counter()
    client.transcribe(samples[0])  # fails fast on a wrong URL / token / model name
    print(f"endpoint {client.url} (model {args.model}): first request {time.perf_counter() - t0:.2f} s")

    normalize = EnglishTextNormalizer() if args.language == "en" else BasicTextNormalizer()
    results = []
    for c in levels:
        print(f"running concurrency={c} ...")
        r = run_level(client, samples, c, args, normalize)
        results.append(r)
        print(f"  RTFx {r['rtfx']:.2f}  p50 {r['latency_p50_s']:.2f}s  WER {r['wer_pct']}%  failed {r['failed']}")

    report = {
        "endpoint": client.url,
        "model": args.model,
        "dataset": args.manifest or f"{args.hf_dataset}/{args.hf_config}/{args.hf_split}",
        "files": len(samples),
        "audio_s": round(total_audio, 1),
        "language": args.language,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2))

    print(f"\n{args.model} @ {client.url} ({len(samples)} files, {total_audio:.0f} s audio)\n")
    print("| concurrency | req/s | RTFx | latency p50 (s) | latency p95 (s) | RTF p50 | RTF p95 | WER % | failed |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['concurrency']} | {r['throughput_rps']:.2f} | {r['rtfx']:.2f} | {r['latency_p50_s']:.2f} | "
            f"{r['latency_p95_s']:.2f} | {r['rtf_p50']:.3f} | {r['rtf_p95']:.3f} | {r['wer_pct']} | {r['failed']} |"
        )
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
