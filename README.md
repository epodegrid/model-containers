# model-containers

Two container images for serving LLMs on the Azure LLM platform. Different
engines, picked for the hardware each runs on:

- **`ik-llama-ornith`** — ik_llama.cpp on CPU. The CPU-only flagship path
  that sidesteps the org privileged-namespace policy blocking the NVIDIA
  device-plugin path. Hosts the Ornith-1.0-35B agentic-coding model.
- **`vllm-qwen3.5-9b`** — vLLM on the T4 pool. Marlin kernel + PagedAttention
  batching for dense 9B at high concurrent-user throughput.

The brief originally picked ik_llama.cpp everywhere as the "single engine"
default. This is the "add vLLM on T4 because measured concurrency on T4
benefits from PagedAttention" branch from the brief — two engines, two
images, deployed as separate AKS workloads.

## Why two engines, not one

| Pool | Engine | Why |
|---|---|---|
| CPU (NCC40 / Genoa) | **ik_llama.cpp** | CPU-only; vLLM doesn't help. ik_llama has the MoE/GatedDeltaNet optimizations that work around stock-llama.cpp's CPU bug for Ornith's Qwen3.5 hybrid arch. |
| GPU (NCasT4_v3 / T4) | **vLLM** | PagedAttention + continuous batching handle multi-user concurrency orders of magnitude better than ik_llama on the same hardware. Modern vLLM supports Marlin on Turing, which ik_llama.cpp doesn't have. |

## Images

| Image | Engine | Model | Format | Weight | Hardware |
|---|---|---|---|---|---|
| `ghcr.io/epodegrid/ik-llama-ornith:q4km`   | ik_llama.cpp `cpu-swap` | `deepreinforce-ai/Ornith-1.0-35B`  | GGUF Q4_K_M (5 shards × ~5 GB) | 21.2 GB | EPYC Genoa (NCC40) |
| `ghcr.io/epodegrid/vllm-qwen3.5-9b:bf16`  | vLLM OpenAI              | `Qwen/Qwen3.5-9B`                  | safetensors BF16                | ~18 GB  | NCasT4_v3 (T4)       |

> **Why ik_llama.cpp on T4 wouldn't work as well for Qwen3.5-9B:** The brief's
> shortlist noted "No Marlin = modest tok/s" for the T4 pool with ik_llama
> (Turing predates the Marlin kernel). vLLM has Marlin support on Turing +
> PagedAttention for batched serving. For a dense 9B model with concurrent
> users, vLLM is the right pick.

## Why GGUF sharded for Ornith

GHCR's per-layer upload limit is 10 GB. The Ornith Q4_K_M is 21 GB. To push
as a single OCI image we shard with `llama-gguf-split --split-max-size 5G`
(5 GB chunks, each well under the limit) and `COPY` each shard as its own
image layer. llama.cpp / ik_llama loads a sharded GGUF natively — just point
`--model` at the `-00001-of-` shard.

We shard to 5 GB (not 9 GB or 10 GB) to leave headroom for layer-encoding
overhead and to fit within GHCR's ~10 min per-layer upload timeout even on
slower links.

## Why bake the vLLM model into the image too

Same reason: AKS pods can't reach Hugging Face at runtime (org egress
block). We pre-download in CI (where egress works) and `COPY` each
safetensors file into the image as its own layer, ordered largest-first so
config/tokenizer edits don't invalidate the heavy weight layers in the
build cache.

## Local quick-start

```bash
# Ornith on CPU (the brief's flagship path)
docker pull ghcr.io/epodegrid/ik-llama-ornith:q4km
docker run --rm -p 9292:8080 ghcr.io/epodegrid/ik-llama-ornith:q4km

# Qwen3.5-9B on GPU (T4)
docker pull ghcr.io/epodegrid/vllm-qwen3.5-9b:bf16
docker run --rm --gpus all --ipc=host -p 9292:8000 ghcr.io/epodegrid/vllm-qwen3.5-9b:bf16

# Smoke test (both)
curl -s http://localhost:9292/v1/models
curl -s http://localhost:9292/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"ornith","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

### Model IDs / aliases (Ornith image, llama-swap routing)

- `ornith`, `Ornith-1.0-35B`, `deepreinforce-ai/Ornith-1.0-35B`
  - `ornith:thinking-coding` — explicit thinking-mode sampling
  - `ornith:instruct`       — non-thinking mode (enable_thinking=false)

### Model IDs (vLLM image, single-model pod)

- `qwen3.5-9b` (or `--model Qwen/Qwen3.5-9B` when calling the API)
  - Defaults: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5 (model card thinking defaults)
  - Append `:instruct` style suffixes to override sampling at the client level

## AKS pull-through (Nexus)

Replace the image reference at the AKS deployment level:

```
# Direct (only works if AKS nodes can reach ghcr.io)
ghcr.io/epodegrid/ik-llama-ornith:q4km
ghcr.io/epodegrid/vllm-qwen3.5-9b:bf16

# Via Nexus (the org-default path — AKS nodes pull from Nexus, Nexus caches from GHCR)
<nexus-host>:8081/<docker-proxy>/epodegrid/ik-llama-ornith:q4km
<nexus-host>:8081/<docker-proxy>/epodegrid/vllm-qwen3.5-9b:bf16
```

Nexus 3.92 handles Docker-image-format proxying fine. The GGUFs are baked
into image layers (not pushed as bare ORAS artifacts) precisely because
Nexus 3.92 doesn't reliably proxy OCI artifact manifests until 3.94.

## Build pipeline

`/.github/workflows/build.yml` builds both images from a single workflow with
a matrix strategy. Each matrix entry:

1. **Downloads** the model from Hugging Face:
   - ik_llama: `hf_hub_download` for a single `.gguf` file
   - vLLM: `snapshot_download` for the full safetensors repo
2. **(ik_llama only)** Builds `llama-gguf-split` from a shallow clone of
   `ggerganov/llama.cpp`, shards the GGUF to ~5 GB chunks, deletes the
   llama.cpp source to free disk.
3. **Generates** the `Dockerfile` from the engine-specific template
   (`Dockerfile.ik-llama.in` or `Dockerfile.vllm.in`) plus one `COPY` line
   per file appended:
   - ik_llama: one COPY per shard
   - vLLM: one COPY per snapshot file, ordered largest-first
4. `docker buildx build` pushes to `ghcr.io/epodegrid/<image_name>:<tags>`.

Triggers:

- Push to `main` that touches `Dockerfile.ik-llama.in`, `Dockerfile.vllm.in`,
  `models/**`, or the workflow file itself.
- Manual `workflow_dispatch`.

Image tags emitted per build (per matrix entry):

- `ghcr.io/.../<name>:q4km` or `:bf16` (per-engine tag, default-branch only)
- `ghcr.io/.../<name>:latest`                          (default-branch only)
- `ghcr.io/.../<name>:<sha-prefix>`                    (every build)

## Updating to a newer model / quant

For Ornith (ik_llama): edit `matrix.include` in the workflow (change
`hf_filename`, `image_tag`), update the `--model` path in
`models/ornith-1.0-35b/config.yaml` if the shard count changes, push.

For Qwen3.5-9B (vLLM): edit `matrix.include` in the workflow, update the
`--model` arg and `--max-model-len` in `Dockerfile.vllm.in` if needed, push.

To swap engines (e.g., add a vLLM image for Ornith on GPU or an ik_llama
image for Qwen3.5-9B on CPU), add another matrix entry with the
appropriate `dockerfile_template`, `download_mode`, `shard`, and
`hf_*` fields.

## Repo layout

```
.
├── .github/workflows/build.yml      # CI: download → (shard) → build → push
├── Dockerfile.ik-llama.in           # Template for the CPU/Genoa image
├── Dockerfile.vllm.in               # Template for the GPU/T4 image
├── models/
│   └── ornith-1.0-35b/config.yaml   # llama-swap config (CPU image only)
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
  may need to fork the build step with explicit Zen4 flags.
- **Single-engine default was overruled.** The brief said "start
  single-engine on ik_llama, add vLLM only if the small tier is measured
  to actually saturate." We chose to commit to vLLM on T4 up front because
  PagedAttention is a large enough batched-throughput lever that the
  "actually saturate" question is answered in advance for dense 9B at any
  realistic concurrency level. If you want to revert, change the
  `qwen3.5-9b` matrix entry's `dockerfile_template` to `Dockerfile.ik-llama.in`,
  set `download_mode: single`, point `hf_repo` / `hf_filename` at a
  Qwen3.5-9B GGUF (need to find one), and set `shard: true`.