# model-containers

Container images for serving LLMs and embeddings on the Azure LLM platform.
**All CPU** (E48as_v5 today, E48as_v7 when it lands); no GPU required
(the T4 pool is air-gapped and the privileged-namespace policy blocks GPU
driver DaemonSets anyway).

## Images

Each ik_llama.cpp image is built twice — once for Zen 4 (current
E48as_v5 hardware) and once for Zen 5 (E48as_v7 when it lands) — so the
binary's ISA matches the node pool. The Zen 4 build uses `-march=znver4`;
the Zen 5 build uses `-march=znver5` (which adds VNNI / BF16 / VBMI and
flips `HAVE_FANCY_SIMD` on for the IQK VNNI kernels).

| Image | Engine | Model | Format | Target ISA |
|---|---|---|---|---|
| `ghcr.io/epodegrid/eve:zen4`     | ik_llama.cpp (`-march=znver4`) | `deepreinforce-ai/Ornith-1.0-35B`        | GGUF Q4_K_M (5 shards × ~5 GB) | E48as_v5 (Zen 4) |
| `ghcr.io/epodegrid/eve:zen5`     | ik_llama.cpp (`-march=znver5`) | same                                  | same                                | E48as_v7 (Zen 5) |
| `ghcr.io/epodegrid/wall-e:zen4`  | ik_llama.cpp (`-march=znver4`) | `google/gemma-4-12b-it-qat-q4_0`        | GGUF Q4_0   (2 shards × ~5 GB) | E48as_v5 (Zen 4) |
| `ghcr.io/epodegrid/wall-e:zen5`  | ik_llama.cpp (`-march=znver5`) | same                                  | same                                | E48as_v7 (Zen 5) |
| `ghcr.io/epodegrid/go-4:latest`  | Custom Python (FastAPI + transformers) | `nomic-embed-text-v1.5` + `nomic-embed-vision-v1.5` | safetensors | Either (ISA-agnostic) |

> **Why a custom build of ik_llama.cpp?** The upstream `cpu-swap` image
> uses `GGML_NATIVE=ON`, which only enables Zen features if its CI build
> host happens to be that Zen. More importantly, upstream's `CMakeLists.txt`
> adds `-mavx512f -mavx512bw` for `GGML_AVX512=ON` but **does NOT add
> `-mavx512dq`**, which the `v_tanh(__m512)` overload requires (gated on
> `__AVX512F__ && __AVX512DQ__`). With whatever host CPU the upstream
> image happens to build on, this can silently break the build.
>
> `Dockerfile.ik-llama.in` is a multi-stage build that compiles ik_llama.cpp
> from source with explicit `-march=znver${ARCH}`. This pins the ISA to
> the target Zen regardless of build host and resolves the DQ gap.

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
# eve — Ornith (CPU; no GPU needed). Pick the ISA that matches your node.
docker pull ghcr.io/epodegrid/eve:zen4      # for E48as_v5 (Zen 4)
docker pull ghcr.io/epodegrid/eve:zen5      # for E48as_v7 (Zen 5, future)
docker run --rm -p 9292:8080 ghcr.io/epodegrid/eve:zen4

# wall-e — Gemma 4 12B
docker pull ghcr.io/epodegrid/wall-e:zen4
docker pull ghcr.io/epodegrid/wall-e:zen5
docker run --rm -p 9293:8080 ghcr.io/epodegrid/wall-e:zen4

# go-4 — embedding service (single image, ISA-agnostic)
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
ghcr.io/epodegrid/eve:zen4      # for Zen 4 nodes (E48as_v5)
ghcr.io/epodegrid/eve:zen5      # for Zen 5 nodes (E48as_v7)
ghcr.io/epodegrid/wall-e:zen4
ghcr.io/epodegrid/wall-e:zen5
ghcr.io/epodegrid/go-4:latest

# Via Nexus (the org-default path — AKS nodes pull from Nexus, Nexus caches from GHCR)
<nexus-host>:8081/<docker-proxy>/epodegrid/eve:zen4
<nexus-host>:8081/<docker-proxy>/epodegrid/eve:zen5
<nexus-host>:8081/<docker-proxy>/epodegrid/wall-e:zen4
<nexus-host>:8081/<docker-proxy>/epodegrid/wall-e:zen5
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

The ik_llama image matrix runs **4 builds per push** (eve × wall-e ×
{zen4, zen5}). The Dockerfile is parameterized via `--build-arg ARCH=4|5`:

```
FROM ubuntu:24.04 AS ik-llama-builder
  ARG ARCH=4
  … build-essential, cmake, git, libcurl4-openssl-dev …
  git clone https://github.com/ikawrakow/ik_llama.cpp.git /src
  cmake -B build \
    -DGGML_NATIVE=OFF \
    -DCMAKE_C_FLAGS="-march=znver${ARCH}" \
    -DCMAKE_CXX_FLAGS="-march=znver${ARCH}" \
    -DGGML_OPENMP=ON -DGGML_IQK_FA_ALL_QUANTS=ON
  cmake --build build --target llama-server -j$(nproc)

FROM ghcr.io/ikawrakow/ik-llama-cpp:cpu-swap
  COPY --from=ik-llama-builder /src/build/bin/llama-server /app/llama-server
  COPY config.yaml /app/config.yaml
  COPY <shards…> /models/
```

For Zen 5, swap `${ARCH}` → `5` and the build picks up VNNI/BF16/VBMI
automatically (because `znver5` defines them). For Zen 4 we leave the
explicit `znver4` so we don't get the upstream CMakeLists's `-mavx512f
-mavx512bw` partial set (which is missing the `-mavx512dq` that
`v_tanh(__m512)` requires).

Triggers:

- Push to `main` that touches `Dockerfile.ik-llama.in`, `go-4/**`,
  `models/**`, or the workflow file itself.
- Manual `workflow_dispatch`.

Image tags emitted per build (per matrix entry):

- `ghcr.io/.../<name>:zen4` / `:zen5` / `:latest`  (per-image tag, default-branch only)
- `ghcr.io/.../<name>:<sha-prefix>`                (every build)

The matrix produces 5 images per default-branch push: `eve:zen4`,
`eve:zen5`, `wall-e:zen4`, `wall-e:zen5`, `go-4:latest`.

## Updating to a newer model / quant

For **eve / wall-e**: edit the matrix entries' `hf_repo` / `hf_filename`,
update `--model` path in `models/<id>/config.yaml` if shard count changes,
push. Both Zen 4 and Zen 5 builds need updating (4 entries total: 2 models
× 2 ISAs).

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
  with MTP. We need an actual `llama-bench` run on the E48as_v5 node to
  confirm the build is on the right ISA and not silently falling back.
- **No published Ornith MTP GGUF found.** MTP (1.4-2.2x lever) would
  roughly double throughput headroom toward the 40-50 tok/s commercial
  median. If deepreinforce-ai publishes one, just swap the matrix entry.
- **Zen 4 lacks VNNI / BF16 / VBMI.** So `HAVE_FANCY_SIMD` (which is
  gated on `__AVX512VNNI__`) is **NOT** defined on the `zen4` build —
  we get the AVX-512 base path, not the IQK VNNI kernels. Zen 4 uses
  256-bit double-pumped AVX-512 (~30-50% faster than AVX2). The `zen5`
  build gets the full IQK VNNI kernels because znver5 has VNNI.
- **Go-4 loads both text + vision encoders at startup.** That's ~500 MB
  of weights + ~500 MB of PyTorch CPU. The cold-start time on a 24-core
  Genoa node should be ~10-15 s. If we add image batching later, switch
  the torch wheels to a CUDA image and serve from a GPU node — the
  service code already keeps CPU inference paths clean.
- **No Nexus Docker-proxy auth validation yet.** Same as the brief —
  whether Nexus can pull public GHCR repos directly or needs a private
  upstream + creds is still an open org-policy question. Doesn't affect
  building/pushing to GHCR; only affects how AKS pulls the image.