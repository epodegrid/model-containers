# model-containers

Two container images for serving LLMs on the Azure LLM platform — both built
on [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (with
[llama-swap](https://github.com/mostlygeek/llama-swap) pre-integrated via the
official `cpu-swap` upstream image), then wrapped to ship the sharded GGUF
weights inside the image so it pulls as a single OCI artifact.

This sidesteps two of the org constraints called out in the architecture
brief:

1. **No GPU driver path** — ik_llama.cpp's `cpu-swap` image is pure
   userspace. No NVIDIA driver DaemonSet, no privileged pod, nothing for the
   privileged-namespace policy to block.
2. **GHCR layer limit (10 GB per layer)** — `Ornith-1.0-35B` Q4_K_M is 21 GB,
   so the GGUF is sharded (~5 GB each, one image layer per shard) before
   `docker build`. Each layer stays well under the limit and uploads within
   GHCR's per-layer timeout.

## Images

| Image | Model | Quant | Weight | Shards | Hardware target |
|---|---|---|---|---|---|
| `ghcr.io/epodegrid/ik-llama-ornith:q4km`      | `deepreinforce-ai/Ornith-1.0-35B`        | Q4_K_M | 21.2 GB | 5 × ~5 GB | EPYC Genoa CPU (NCC40) |
| `ghcr.io/epodegrid/ik-llama-qwen3.5-0.8b:q4km` | `Qwen/Qwen3.5-0.8B` (mradermacher GGUF) | Q4_K_M | 528 MB  | 1         | NCasT4_v3 (T4)         |

Both images expose the same surface: llama-swap on port `8080`, OpenAI- and
Anthropic-compatible APIs at `/v1` and `/v1/messages`, llama.cpp web UI at
`/ui`, Prometheus metrics at `/metrics`.

> **Why the same engine everywhere?** Per the brief's "single-engine default"
> decision: one image pattern, one metrics format, one ops surface. vLLM is
> only added to the small tier if measured concurrency actually saturates
> ik_llama.cpp on T4 — measured, not assumed.

### Why not stock llama.cpp / LocalAI for Ornith?

Ornith is a hybrid GatedDeltaNet + attention MoE (Qwen3.5 family). Stock
llama.cpp on that arch hits
[a known CPU perf bug](https://github.com/ggerganov/llama.cpp/issues/19480) —
Qwen3-Coder-Next (80B MoE, 3B active) gets ~7.7 tok/s where bandwidth math
predicts 35-60 tok/s. ik_llama.cpp's MoE optimizations are the cited fix.
LocalAI's default backend wraps stock llama.cpp and inherits the bug.

### Why mradermacher for Qwen3.5-0.8B?

There is no `Qwen/Qwen3.5-0.8B-GGUF`. mradermacher's quant is text-only (the
mmproj files are listed separately) and uses standard k-quants that ik_llama
loads cleanly. Unsloth's `_XL` quants are explicitly unsafe for ik_llama —
this one isn't Unsloth so that's moot, but worth flagging if we add more
small models later.

## Local quick-start

```bash
# Pull an image (or via Nexus — see below)
docker pull ghcr.io/epodegrid/ik-llama-ornith:q4km

# Run it. llama-swap listens on 8080 inside the container.
# First request for a model triggers load (~30-60s for Ornith).
docker run --rm -p 9292:8080 ghcr.io/epodegrid/ik-llama-ornith:q4km

# Smoke test
curl -s http://localhost:9292/v1/models
curl -s http://localhost:9292/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ornith",
    "messages": [{"role":"user","content":"hi"}],
    "max_tokens": 32
  }'
```

Available model IDs (defined per-image in `models/*/config.yaml`):

- `ornith`, `Ornith-1.0-35B`, `deepreinforce-ai/Ornith-1.0-35B`
  - `ornith:thinking-coding` — explicit thinking mode
  - `ornith:instruct`       — non-thinking mode
- `qwen3.5-0.8b`, `Qwen3.5-0.8B`, `Qwen/Qwen3.5-0.8B`
  - `qwen3.5-0.8b:thinking` — enable thinking
  - `qwen3.5-0.8b:vl`       — VL-style sampling (works for text-only too)

## AKS pull-through (Nexus)

The brief's Nexus-as-pull-through-proxy path applies as-is. Replace the image
reference at the AKS deployment level:

```
# Direct (only works if AKS nodes can reach ghcr.io)
ghcr.io/epodegrid/ik-llama-ornith:q4km

# Via Nexus (the org-default path — AKS nodes pull from Nexus, Nexus caches from GHCR)
<nexus-host>:8081/<docker-proxy>/epodegrid/ik-llama-ornith:q4km
```

Nexus 3.92 handles Docker-image-format proxying fine; the GGUFs are baked
into image layers (not pushed as bare ORAS artifacts) precisely because Nexus
3.92 doesn't reliably proxy OCI artifact manifests until 3.94.

## Build pipeline

`/.github/workflows/build.yml` builds both images from a single workflow with
a matrix strategy. Each matrix entry:

1. Downloads the GGUF from Hugging Face (`huggingface_hub`).
2. Builds `llama-gguf-split` from a shallow clone of `ggerganov/llama.cpp`.
3. Shards the GGUF to ~5 GB chunks (`--split-max-size 5G`).
4. Generates a `Dockerfile` from `Dockerfile.in` plus one `COPY` line per
   shard appended at the end. This keeps the per-model bits (`config.yaml`)
   in one place while letting the shard count vary per model.
5. `docker buildx build` with `--build-arg MODEL_CONFIG=<per-model path>`,
   pushes to `ghcr.io/epodegrid/<image_name>:<tags>`.

Triggers:

- Push to `main` that touches `Dockerfile.in`, `models/**`, or the workflow
  file itself.
- Manual `workflow_dispatch`.

Image tags emitted per build:

- `ghcr.io/.../<name>:q4km`         (per-quant tag, on default-branch only)
- `ghcr.io/.../<name>:latest`       (alias for `q4km`, on default-branch only)
- `ghcr.io/.../<name>:<sha-prefix>` (every build)

## Updating to a newer model / quant

1. Edit the matrix entry in `.github/workflows/build.yml` (change `hf_repo`,
   `hf_filename`, `image_name`, `image_tag`).
2. Update `models/<id>/config.yaml` so the `--model` path references the
   new first-shard filename (and so the chat template / sampler settings
   match the new model's recommended defaults).
3. Push. Workflow builds & pushes; AKS image reference points at the new
   tag.

For the 5-shard Ornith image the first-shard filename is
`Ornith-1.0-35B-Q4_K_M-00001-of-00005.gguf` — if you bump to Q5_K_M (24.7
GB → still 5 shards at 5 GB max) it becomes
`Ornith-1.0-35B-Q5_K_M-00001-of-00005.gguf`, etc.

## Repo layout

```
.
├── .github/workflows/build.yml      # CI: download → shard → build → push
├── Dockerfile.in                    # Template; CI appends one COPY per shard
├── models/
│   ├── ornith-1.0-35b/config.yaml   # llama-swap config for the CPU image
│   └── qwen3.5-0.8b/config.yaml     # llama-swap config for the T4 image
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

## Caveats / open items (carried over from the architecture brief)

- **No real EPYC Genoa benchmark yet** for Ornith on ik_llama.cpp. The
  brief's table estimates ~20-40 tok/s single-stream at Q4 on Genoa with
  ik_llama, possibly 30-60 tok/s with MTP. We need an actual run on the
  NCC40 to commit to this as the flagship path. The container image is
  independent of that answer — swap quants or engines later without
  re-platforming.
- **No published Ornith MTP GGUF found.** Multi-Token Prediction (1.4-2.2x
  throughput lever in llama.cpp) would roughly double the headroom toward
  the 40-50 tok/s commercial median. If deepreinforce-ai publishes an MTP
  GGUF, just swap the `hf_filename` in the matrix entry and re-tag.
- **No Nexus Docker-proxy auth validation yet.** Whether Nexus can pull a
  public GHCR repo directly, or whether we need a private upstream + creds,
  is still an open org-policy question. Doesn't affect building/pushing to
  GHCR — only affects how AKS pulls it.
- **GGML_NATIVE in the upstream `cpu-swap` image.** ik_llama's CI builds
  with `-DGGML_NATIVE=ON`. Whether that picks up Zen4 IQK kernels depends
  on the build host's CPU. If benchmarks show poor tok/s on EPYC Genoa we
  may need to fork the build step with `--cpu-moe` flags or rebuild from
  source with explicit Zen4 flags.
- **Single-engine assumption.** ik_llama.cpp everywhere is the default. If
  the small tier is measured to saturate (i.e. Qwen3.5-0.8B with `--parallel
  4` actually fills the T4), add a third image using vLLM as a separate
  engine. Don't preemptively add it.