# model-containers

Three container images for serving LLMs and embeddings on the Azure LLM
platform. **All CPU** (E48as_v5 today, E48as_v7 when it lands); no GPU
required (the T4 pool is air-gapped and the privileged-namespace policy
blocks GPU driver DaemonSets anyway).

| Image | Engine | Model | Format | Hardware target |
|---|---|---|---|---|
| `ghcr.io/epodegrid/eve:q4km`     | ik_llama.cpp (Zen4 build) | `deepreinforce-ai/Ornith-1.0-35B`        | GGUF Q4_K_M (5 shards × ~5 GB) | E48as_v5 (or v7) |
| `ghcr.io/epodegrid/wall-e:q4_0`  | ik_llama.cpp (Zen4 build) | `google/gemma-4-12b-it-qat-q4_0`        | GGUF Q4_0   (2 shards × ~5 GB) | E48as_v5 (or v7) |
| `ghcr.io/epodegrid/go-4:latest`  | Custom Python (FastAPI + transformers) | `nomic-embed-text-v1.5` + `nomic-embed-vision-v1.5` | safetensors | E48as_v5 (or v7) |

> **Why a custom build of ik_llama.cpp?** The upstream `cpu-swap` image uses
> `GGML_NATIVE=ON`, which only enables Zen4 features if its CI build host
> happens to be Zen4. To guarantee the IQK GEMM kernels (`HAVE_FANCY_SIMD`)
> run on our EPYC Genoa nodes, `Dockerfile.ik-llama.in` is a multi-stage
> build that compiles ik_llama.cpp from source with explicit
> `-DGGML_AVX512{,_VBMI,_VNNI,_BF16}=ON` flags (per upstream
> `docs/build.md`). Without these flags, you silently fall back to AVX2 and
> leave ~30-50% prompt-processing perf on the table.

## Why these three

- **eve** — the flagship agentic-coding model. The brief originally picked
  ik_llama.cpp on the CPU side of an H100 confidential VM; with H100
  blocked, the same model + engine now runs on E48as_v5 (or v7).
  Hybrid Qwen3.5 GatedDeltaNet arch; ik_llama's MoE fixes are exactly what
  we need (stock llama.cpp on this arch hits the
  [issue #19480 perf bug](https://github.com/ggerganov/llama.cpp/issues/19480)).
- **wall-e** — 12B dense QAT-quantized Gemma 4. Smaller than the flagship
  but useful for the small chat tier on the same pool. ik_llama.cpp has
  Gemma 4 support (PR #1581). QAT gives better quality at Q4_0 than naive Q4.
- **go-4** — multimodal embedding service (text + image in the same latent
  space). Built directly on `python:3.11-slim` because **ik_llama.cpp can't
  serve image embeddings standalone** — its vision support is for multimodal
  *chat* (vision feeds into an LLM decoder), not standalone image-embedding
  endpoints. So we wrap HF transformers + FastAPI for this one.

## Local quick-start

```bash
# eve — Ornith (CPU; no GPU needed)
docker pull ghcr.io/epodegrid/eve:q4km
docker run --rm -p 9292:8080 ghcr.io/epodegrid/eve:q4km

# wall-e — Gemma 4 12B
docker pull ghcr.io/epodegrid/wall-e:q4_0
docker run --rm -p 9293:8080 ghcr.io/epodegrid/wall-e:q4_0

# go-4 — embedding service
docker pull ghcr.io/epodegrid/go-4:latest
docker run --rm -p 9294:8000 ghcr.io/epodegrid/go-4:latest

# Smoke test eve
curl -s http://localhost:9292/v1/models
curl -s http://localhost:9292/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"eve","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'

# Smoke test go-4
curl -s http://localhost:9294/health
curl -s http://localhost:9294/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-vision-v1.5","input":"hello world"}'
```

### Model IDs / aliases

**eve** (llama-swap routing):
- `eve`, `Ornith-1.0-35B`, `deepreinforce-ai/Ornith-1.0-35B`
  - `eve:thinking-coding` — explicit thinking-mode sampling
  - `eve:instruct`        — non-thinking mode (enable_thinking=false)

**wall-e** (llama-swap routing):
- `wall-e`, `gemma-4-12b`, `google/gemma-4-12b-it-qat-q4_0`
  - `wall-e:coding`   — coding-mode sampler (temp 0.6)
  - `wall-e:instruct` — non-thinking mode

**go-4** (single FastAPI app, no swap):
- `nomic-embed-vision-v1.5` — accepts mixed text + image inputs in `/v1/embeddings`

## AKS pull-through (Nexus)

Replace the image reference at the AKS deployment level:

```
# Direct (only works if AKS nodes can reach ghcr.io)
ghcr.io/epodegrid/eve:q4km
ghcr.io/epodegrid/wall-e:q4_0
ghcr.io/epodegrid/go-4:latest

# Via Nexus (the org-default path — AKS nodes pull from Nexus, Nexus caches from GHCR)
<nexus-host>:8081/<docker-proxy>/epodegrid/eve:q4km
<nexus-host>:8081/<docker-proxy>/epodegrid/wall-e:q4_0
<nexus-host>:8081/<docker-proxy>/epodegrid/go-4:latest
```

Nexus 3.92 handles Docker-image-format proxying fine; the GGUFs are baked
into image layers (not pushed as bare ORAS artifacts) precisely because
Nexus 3.92 doesn't reliably proxy OCI artifact manifests until 3.94.

## Build pipeline

`/.github/workflows/build.yml` builds all three images from one workflow with
a matrix strategy. Per matrix entry:

1. **Downloads** model artifacts from Hugging Face:
   - `single` mode: one `.gguf` file via `hf_hub_download`
   - `snapshot` mode: full repo snapshots via `snapshot_download` (one or
     many repos in parallel — `go-4` uses this for two repos)
2. **(ik_llama only)** Builds `llama-gguf-split` from a shallow clone of
   `ggerganov/llama.cpp`, shards the GGUF to ~5 GB chunks, deletes the
   llama.cpp source to free disk.
3. **(ik_llama only)** Generates the `Dockerfile` from
   `Dockerfile.ik-llama.in` plus one `COPY` line per shard appended.
4. `docker buildx build` pushes to `ghcr.io/epodegrid/<image_name>:<tags>`.

### The ik_llama multi-stage build

```
FROM ubuntu:24.04 AS ik-llama-builder
  … build-essential, cmake, git, libcurl4-openssl-dev …
  git clone https://github.com/ikawrakow/ik_llama.cpp.git /src
  cmake -B build \
    -DGGML_NATIVE=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_OPENMP=ON \
    -DGGML_IQK_FA_ALL_QUANTS=ON
  cmake --build build --target llama-server -j$(nproc)

FROM ghcr.io/ikawrakow/ik-llama-cpp:cpu-swap
  COPY --from=ik-llama-builder /src/build/bin/llama-server /app/llama-server
  COPY config.yaml /app/config.yaml
  COPY <shards…> /models/
```

Triggers:

- Push to `main` that touches `Dockerfile.ik-llama.in`, `go-4/**`,
  `models/**`, or the workflow file itself.
- Manual `workflow_dispatch`.

Image tags emitted per build (per matrix entry):

- `ghcr.io/.../<name>:q4km` / `:q4_0` / `:latest` (per-image tag, default-branch only)
- `ghcr.io/.../<name>:latest`                       (default-branch only)
- `ghcr.io/.../<name>:<sha-prefix>`                 (every build)

## Updating to a newer model / quant

For **eve / wall-e**: edit the matrix entry's `hf_repo` / `hf_filename` /
`image_tag`, update `--model` path in `models/<id>/config.yaml` if shard
count changes, push.

For **go-4**: edit the matrix entry's `hf_repos` and `hf_snapshot_dirs`
(one per line), push.

To swap engines, add another matrix entry with the appropriate
`dockerfile_template`, `download_mode`, `shard`, and `hf_*` fields.

## Repo layout

```
.
├── .github/workflows/build.yml      # CI: download → (shard) → build → push
├── Dockerfile.ik-llama.in           # Multi-stage: build ik_llama.cpp w/ Zen4
│                                    # IQK kernels + ship via cpu-swap base.
├── models/
│   ├── ornith-1.0-35b/config.yaml   # llama-swap config (eve)
│   └── gemma-4-12b-it-qat-q4_0/config.yaml   # llama-swap config (wall-e)
├── go-4/
│   ├── Dockerfile.go-4.in           # Python service template
│   ├── service.py                   # FastAPI embedding service
│   └── requirements.txt
├── .dockerignore
├── .gitignore
├── LICENSE
└── README.md
```

## Caveats / open items (carried over from the architecture brief)

- **No real EPYC Genoa benchmark yet.** The brief's table estimates ~20-40
  tok/s single-stream at Q4 on Genoa with ik_llama, possibly 30-60 tok/s
  with MTP. The custom Zen4 build should hit the upper end of that range.
  We need an actual `llama-bench` run on the E48as_v5 node to confirm
  `HAVE_FANCY_SIMD` is firing and not silently falling back to AVX2.
- **No published Ornith MTP GGUF found.** MTP (1.4-2.2x lever) would
  roughly double throughput headroom toward the 40-50 tok/s commercial
  median. If deepreinforce-ai publishes one, just swap the matrix entry.
- **Go-4 loads both text + vision encoders at startup.** That's ~500 MB
  of weights + ~500 MB of PyTorch CPU. The cold-start time on a 24-core
  Genoa node should be ~10-15 s. If we add image batching later, switch
  the torch wheels to a CUDA image and serve from a GPU node — the
  service code already keeps CPU inference paths clean.
- **No Nexus Docker-proxy auth validation yet.** Same as the brief —
  whether Nexus can pull public GHCR repos directly or needs a private
  upstream + creds is still an open org-policy question. Doesn't affect
  building/pushing to GHCR; only affects how AKS pulls the image.
- **GGML_NATIVE + GGML_AVX512_* on the upstream image.** Verified: the
  five flags `-DGGML_AVX512F`, `-DGGML_AVX512VNNI`, `-DGGML_AVX512VL`,
  `-DGGML_AVX512BW`, `-DGGML_AVX512DQ` together flip `HAVE_FANCY_SIMD` on
  Zen4 (per `docs/build.md`). The build container runs on a stock GitHub
  Actions Ubuntu runner (probably Zen3/Zen4 EPYC); even if it's not Zen4,
  the explicit AVX-512 flags still take effect because they're passed
  verbatim to the compiler. The compiled binary then runs unchanged on
  E48as_v5 (or v7).