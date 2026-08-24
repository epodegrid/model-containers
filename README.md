# model-containers

Container images for serving LLMs and embeddings on the Azure LLM platform.
**All CPU** (E48as_v5 today, E48as_v7 when it lands); no GPU required
(the T4 pool is air-gapped and the privileged-namespace policy blocks GPU
driver DaemonSets anyway).

## Images

| Image | Engine | Model(s) | Hardware |
|---|---|---|---|
| `ghcr.io/epodegrid/eve:latest`    | ik_llama.cpp (multi-ISA dispatch) | `ornith-ai/Ornith-1.5-35B-A3B` (Q5_K_M) | any amd64 with AVX2 (Zen 3+) |
| `ghcr.io/epodegrid/wall-e:latest` | ik_llama.cpp (multi-ISA dispatch) | `Qwen/Qwen3.8-27B` (UD-Q6_K_L)        | any amd64 with AVX2 (Zen 3+) |
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
# eve — Ornith 1.5 35B-A3B (MoE)
docker pull ghcr.io/epodegrid/eve:latest
docker run --rm -p 9292:8080 ghcr.io/epodegrid/eve:latest
# First request takes ~30-60s while llama-server loads the model.
# Watch the wrapper dispatch: `kubectl logs` will show
# `[llama-server-wrapper] CPU detected (zenN), exec'ing /app/llama-server.zenN`

# wall-e — Qwen3.8 27B
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

### Memory per replica

llama.cpp commits the whole cache for `--ctx-size` at load time, used or not,
so the figure below is what each pod holds the moment it finishes loading — and
it multiplies by every replica KEDA adds. (Under Ornith 1.0 the cache was the
larger half of that figure; with the hybrid models below it no longer is — see
the note after the table.)

Both images run the full native 256K context with a **q8_0 KV cache**
(`-ctk q8_0 -ctv q8_0`), which halves the cache for negligible quality cost.
That is only legal alongside `-fa on`: ik_llama refuses a quantized V cache
without flash attention.

Both models are **Qwen3.5 hybrid-attention** architectures: `layer_types`
alternates three `linear_attention` layers with one `full_attention` layer
(`full_attention_interval: 4`). Only the full-attention layers hold a KV cache
that grows with context. The linear-attention layers carry a fixed-size
recurrent state instead — 63 MB for eve, 151 MB for wall-e, *independent of
`--ctx-size`* — which is why the cache figures below are so much smaller than
the dense Ornith 1.0 pair they replace.

| | full-attn layers | weights | KV @262144 (q8_0) | total |
|---|---|---|---|---|
| `eve` (Ornith-1.5-35B-A3B Q5_K_M) | 10 of 40 | 25.4 GB | 2.9 GB | **28.3 GB** |
| `wall-e` (Qwen3.8-27B UD-Q6_K_L) | 16 of 64 | 24.2 GB | 9.1 GB | **33.4 GB** |

wall-e still holds the larger cache — 16 full-attention layers at 4 KV heads
against eve's 10 at 2 — so at equal context it costs about 3.2× per token. With
an f16 cache these would be 34.6 GB and 46.2 GB.

The headline change from Ornith 1.0 is that **weights now dominate, not cache**.
The old pair spent 17.2 GB (wall-e) and 10.7 GB (eve) on KV; the hybrid layers
cut that to 9.1 and 2.9 while the weights roughly tripled on wall-e. Net per
replica: eve drops 35.5 → 28.3 GB, wall-e rises 26.7 → 33.4 GB.

Because the cache is now a small fraction of the total, **dropping `--ctx-size`
buys far less than it used to** — cutting eve to 64K reclaims ~2.1 GB against a
28.3 GB replica. If you need more replicas per node, a smaller quant is now the
lever, not a shorter context.

### Reasoning

Both ik_llama images serve reasoning models and are built for it: the config
passes `--reasoning on --reasoning-format deepseek`, so the assistant's
`<think> … </think>` trace comes back in a separate **`reasoning_content`**
field rather than inline in `content`. Nothing has to be enabled per request.

```jsonc
{"choices":[{"message":{"role":"assistant",
  "reasoning_content":"...the chain of thought...",
  "content":"...the answer..."}}]}
```

Bare ids (`eve`, `wall-e`) leave sampling to the caller. The suffixed ids are
**enforced presets**: llama-swap overwrites those parameters on the way
through, so a `temperature` sent to `eve:instruct` is replaced, not merged.
Send to the bare id when the caller needs control.

`:instruct` turns thinking off via `enable_thinking=false`, which is the one
way to get a non-reasoning reply.

### Model IDs / aliases

**eve** (llama-swap routing):
- `eve`, `ornith`, `Ornith-1.5-35B-A3B`, `ornith-ai/Ornith-1.5-35B-A3B`
  - `eve:thinking-coding` — agentic/coding sampler (temp 1.0, top_p 0.95)
  - `eve:instruct`        — non-thinking mode (enable_thinking=false)

**wall-e** (llama-swap routing):
- `wall-e`, `qwen`, `Qwen3.8-27B`, `Qwen/Qwen3.8-27B`
  - `wall-e:thinking-coding`  — agentic/coding sampler (temp 1.0, top_p 0.95)
  - `wall-e:instruct`         — non-thinking mode (enable_thinking=false,
    plus the card's instruct sampler: temp 0.7, top_p 0.8, **presence_penalty 1.5**)
  - `wall-e:thinking-medium`  — `reasoning_effort=medium`
  - `wall-e:thinking-low`     — `reasoning_effort=low`

Only Qwen3.8 has `reasoning_effort`, so the two depth presets exist on wall-e
alone. Note the ids are unchanged (`eve`, `wall-e`) — callers pinned to the
image name keep working across this swap; only `ornith-9b` / `Ornith-1.0-*`
went away with the weights they named.

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
  is firing on Zen 5 silicon. This matters more after the model swap, not
  less: both images changed architecture family *and* roughly tripled
  (wall-e) or held (eve) their weight footprint.
- **MTP weights now exist but are NOT wired up.** This was previously blocked
  on nothing being published. Both new models ship MTP: `Ornith-1.5-35B-A3B`
  sets `mtp_num_hidden_layers: 1`, and the Unsloth Qwen3.8 repo ships a
  separate `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB). Neither is downloaded or
  passed to llama-server today, so the 1.4-2.2x lever is still unclaimed —
  it now needs a draft-model wiring decision, not an upstream release.
- **eve is now an MoE; the throughput estimates above are stale.** The
  ~20-40 tok/s figure was measured against a *dense* 35B. Ornith 1.5 activates
  8 of 256 experts (~3B of ~35B params), so per-token memory traffic is a
  fraction of the 29.2 GB file. That should be a large speedup on a
  bandwidth-bound CPU path, but it is unmeasured — see the benchmark item above.
- **Both new models are vision-capable; both are served text-only.** Each repo
  publishes an `mmproj` file (~0.9 GB) that we do not download or mount, so
  `/v1/chat/completions` accepts text only. Adding it is a deliberate scope
  decision, not an oversight.
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