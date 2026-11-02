# Whisper endpoint benchmark

This script benchmarks a deployed Whisper model, such as `whisper-large-v3-turbo` on CPU. It works with any OpenAI-compatible `/v1/audio/transcriptions` endpoint, for example MLIS/KServe, vLLM or speaches.

Each run measures:

- **WER**: corpus word error rate after Whisper's `EnglishTextNormalizer`. This is the [Open ASR Leaderboard](https://arxiv.org/abs/2510.06961) method.
- **RTFx**: audio seconds transcribed divided by wall-clock seconds. Higher is faster, and the value roughly equals how many real-time audio streams the endpoint can keep up with.
- **Per concurrency level**: requests per second, latency p50/p95, RTF p50/p95 and the number of failed requests. RTF below 1 means a request finished faster than real time.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
export ASR_API_KEY="<your token>"
python asr_bench.py \
    --endpoint https://<your-endpoint-host>/v1 \
    --model <served-model-name> \
    --num-samples 100 --concurrency 1,2,4,8
```

```bash
export ASR_API_KEY="<your token>"
python asr_bench.py --endpoint <url>/v1 --model whisper-large-v3-turbo --insecure \
  --hf-config voxpopuli --hf-split test --num-samples 100 --concurrency 1 \
  --output voxpopuli_results.json
```

- `--endpoint` can be the base host, the `/v1` URL, or the full `.../v1/audio/transcriptions` URL.
- `--model` must be the exact name the endpoint serves. `GET /v1/models` lists it.
- If the URL, token or model name is wrong, the first request fails with the server's error.

By default the script streams LibriSpeech test-clean from Hugging Face. To use your own audio instead, pass a JSONL manifest with one line per file; audio paths are relative to the manifest:

```json
{"audio": "clips/call_001.wav", "text": "reference transcript"}
```

```bash
python asr_bench.py --endpoint ... --model ... --manifest data.jsonl --concurrency 1,4
```

To check the endpoint before benchmarking:

```bash
curl -s https://<your-endpoint-host>/v1/audio/transcriptions \
    -H "Authorization: Bearer $ASR_API_KEY" \
    -F file=@sample.wav -F model=<served-model-name> -F response_format=json
```

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--endpoint` | required | endpoint URL |
| `--model` | required | model name served by the endpoint |
| `--api-key` | `$ASR_API_KEY` | Bearer token |
| `--concurrency` | `1` | comma-separated numbers of requests sent at once |
| `--num-samples` | 100 | files to send at each level |
| `--manifest` | none | JSONL of local audio instead of LibriSpeech |
| `--language` | `en` | language code, or `auto` to let the model detect it |
| `--warmup` | 1 | untimed requests before each level |
| `--timeout` | 600 | per-request timeout in seconds |
| `--insecure` | off | skip TLS verification (for self-signed certificates) |
| `--output` | `results.json` | full results, including errors and example transcripts |

## Output

The script prints a Markdown table and writes `results.json`:

```
| concurrency | req/s | RTFx | latency p50 (s) | latency p95 (s) | RTF p50 | RTF p95 | WER % | failed |
```

As concurrency grows, RTFx should rise until the server's CPUs are saturated, then flatten while latency keeps climbing. The level just before that point is the endpoint's practical capacity.

Reference:
https://localaimaster.com/blog/best-local-speech-to-text-models
https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
https://arxiv.org/html/2510.06961v4
https://hug-proxy.pages.dev/TheStageAI/thewhisper-large-v3-turbo
