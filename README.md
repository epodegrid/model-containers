# model-containers

Container images for serving LLMs and embeddings on the Azure LLM platform.
**All CPU** (E48as_v5 today, E48as_v7 when it lands); no GPU required
(the T4 pool is air-gapped and the privileged-namespace policy blocks GPU
driver DaemonSets anyway).

## Images

| Image | Engine | Model(s) | Hardware |
|---|---|---|---|
| `ghcr.io/epodegrid/eve:latest`    | ik_llama.cpp (multi-ISA dispatch) | `deepreinforce-ai/Ornith-1.0-35B`        | any amd64 with AVX-512 (Zen 3/4/5) |
| `ghcr.io/epodegrid/wall-e:latest` | ik_llama.cpp (multi-ISA dispatch) | `google/gemma-4-12b-it-qat-q4_0`        | any amd64 with AVX-512 (Zen 3/4/5) |
| `ghcr.io/epodegrid/go-4:latest`   | Python (FastAPI + transformers)   | `nomic-embed-text-v1.5` + `nomic-embed-vision-v1.5` | any amd64 |

Both ik_llama images ship **three compiled llama-server binaries** inside
the image (`llama-server.zen3`, `.zen4`, `.zen5`). A small shell wrapper
at `/app/llama-server` reads `/proc/cpuinfo` and `exec`s the right binary
at startup. The dispatch is decided once per pod start (or per model
load via llama-swap), not per request — there is no runtime cost in the
hot path.

This is the answer to the "Azure assigns Milan or Genoa randomly to
E48as_v5" problem: **same image tag, whatever CPU Azure gives you**.
The wrapper picks:

| `flags` from `/proc/cpuinfo` | Binary | ISA | Works on |
|---|---|---|---|
| `avx512vnni` | `/app/llama-server.zen5` | Zen 5 (Turin) + VNNI/BF16/VBMI | Zen 5 silicon only |
| `avx512vl`   | `/app/llama-server.zen4` | Zen 4 (Genoa) + VL         | Zen 4, Zen 5 (downward) |
| `avx512f`    | `/app/llama-server.zen3` | Zen 3 (Milan)             | Zen 3, Zen 4, Zen 5 (all downward) |

A `-march=znver4` binary crashes on Milan because Milan lacks AVX-512VL;
a `-march=znver5` binary (or its GCC 13 emulation) crashes on Zen 4
because Zen 4 lacks VNNI. The wrapper picks the highest-fidelity binary
the host can actually run.

## Why a multi-ISA image vs three separate images

- **No node labels needed** — AKS deployments don't have to know which
  CPU family a given node got. Auto-scaling across a pool works
  regardless of how Azure allocates capacity.
- **One tag to pin** — production rollouts reference `eve:latest` (or
  `:sha-abc1234`) and don't have to track which CPU family is being
  served.
- **Tradeoff** — image is ~3× larger for the ik_llama images
  (~3 × 50 MB for the binaries, on top of the model weights). Per-build
  CI time is also ~3× (three parallel `cmake --build` invocations, all
  sharing a single `apt install` and `git clone` via a `base` stage).

## Local quick-start

```bash
# eve — Ornith
docker pull ghcr.io/epodegrid/eve:latest
docker run --rm -p 9292:8080 ghcr.io/epodegrid/eve:latest
# First request takes ~30-60s while llama-server loads the model.
# Watch the wrapper dispatch: `kubectl logs` will show
# `[llama-server-wrapper] CPU detected (zenN), exec'ing /app/llama-server.zenN`

# wall-e — Gemma 4 12B
docker pull ghcr.io/epodegrid/wall-e:latest
docker run --rm -p 9293:8080 ghcr.io/epodegrid/wall-e:latest

# go-4 — embedding service (Python, ISA-agnostic)
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

```
ghcr.io/epodegrid/eve:latest
ghcr.io/epodegrid/wall-e:latest
ghcr.io/epodegrid/go-4:latest
```

→ `nexus.corp:8081/<docker-proxy>/epodegrid/eve:latest` (etc.) for the
org-default Nexus-fronted pull path.

## Build pipeline

`/.github/workflows/build.yml` builds all three images from one workflow
with a matrix strategy. The matrix is now flat (3 entries — one per
image), not 5×ISA. Each entry:

1. Downloads model artifacts from Hugging Face
   (`hf_hub_download` for single .gguf, `snapshot_download` for the
   Nomic repos in go-4).
2. **(ik_llama only)** Builds `llama-gguf-split` from a shallow clone of
   `ggerganov/llama.cpp`, shards the GGUF to ~5 GB chunks, deletes the
   llama.cpp source to free disk.
3. Generates the `Dockerfile` from the engine-specific template plus
   shard COPYs appended.
4. `docker buildx build` pushes to GHCR.

### The ik_llama multi-stage multi-ISA build

```
FROM ubuntu:24.04 AS base          # shared: apt install, git clone ik_llama.cpp
FROM base AS zen3-builder           # -march=znver3
FROM base AS zen4-builder           # -march=znver4
FROM base AS zen5-builder           # -march=znver4 + vnni/bf16/vbmi/vbmi2
FROM ghcr.io/ikawrakow/ik-llama-cpp:cpu-swap
  COPY --from=zen3-builder  llama-server  /app/llama-server.zen3
  COPY --from=zen4-builder  llama-server  /app/llama-server.zen4
  COPY --from=zen5-builder  llama-server  /app/llama-server.zen5
  COPY wrapper.sh                     /app/llama-server   # the dispatcher
  COPY config.yaml                   /app/config.yaml
  COPY <shards…>                     /models/
```

For Zen 5, GCC 13 (Ubuntu 24.04 default) doesn't recognize
`-march=znver5` (added in GCC 14, May 2024). We emulate it with
`-march=znver4 -mavx512vnni -mavx512bf16 -mavx512vbmi -mavx512vbmi2`,
which produces the same instruction set. When we move to a base image
with GCC 14+ (or Clang 18+), the `zen5-builder` can collapse to a
single `-march=znver5`.

## Caveats / open items (carried over from the architecture brief)

- **No real EPYC Genoa benchmark yet.** The brief's table estimates ~20-40
  tok/s single-stream at Q4 on Genoa with ik_llama, possibly 30-60 tok/s
  with MTP. We need an actual `llama-bench` run on the E48as_v5 node to
  confirm the dispatch is picking the right binary and that HAVE_FANCY_SIMD
  is firing on Zen 5 silicon.
- **No published Ornith MTP GGUF found.** MTP (1.4-2.2x lever) would
  roughly double throughput headroom toward the 40-50 tok/s commercial
  median. If deepreinforce-ai publishes one, just swap the matrix entry.
- **Zen 3 lacks VNNI.** So `HAVE_FANCY_SIMD` is **NOT** defined on the
  `zen3` build — the AVX-512 base path, not the IQK VNNI kernels. Zen 4
  uses 256-bit double-pumped AVX-512 (~30-50% faster than AVX2); Zen 5
  gets the full IQK VNNI kernels.
- **Wrapper dispatch is opaque to llama-swap's health checks.** llama-swap
  spawns `/app/llama-server <args>`; the wrapper exec's the real binary
  with the same args. If the chosen binary immediately crashes (wrong ISA),
  the wrapper's exit code propagates and llama-swap marks the model as
  failed — same failure mode as if the upstream binary had crashed. So
  nothing new to monitor; existing health checks work.
- **Go-4 loads both text + vision encoders at startup.** ~500 MB of
  weights + ~500 MB of PyTorch CPU. Cold-start on a 24-core Genoa node
  ~10-15 s. If we add image batching later, switch to a CUDA image and
  serve from a GPU node — the service code already keeps CPU inference
  paths clean.
- **No Nexus Docker-proxy auth validation yet.** Same as the brief —
  whether Nexus can pull public GHCR repos directly or needs a private
  upstream + creds is still an open org-policy question. Doesn't affect
  building/pushing to GHCR; only affects how AKS pulls the image.