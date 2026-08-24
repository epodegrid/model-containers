#!/usr/bin/env bash
# Smoke test for a built model image: start it, wait for the HTTP server to
# come up, then probe /health, /v1/models, and the heavy (inference) endpoint.
#
# Lives in a script rather than inline in build.yml so it can be run and
# tested locally — the inline version accumulated shell bugs that only
# ever surfaced by burning a full multi-GB CI build.
#
# Required env:
#   IMAGE          image ref to run
#   HEALTH_PATH    e.g. /health
#   HEAVY_PATH     e.g. /v1/chat/completions
#   HEAVY_BODY     JSON POST body for HEAVY_PATH
#   HEAVY_TIMEOUT  seconds to allow for model load + inference
# Optional env:
#   MODELS_PATH    e.g. /v1/models; skipped when empty
#   NAME           label used in the container name (default: smoke)
#   PORT           host port to probe (default: 8080)
#   HEALTH_WAIT    seconds to wait for the server to bind (default: 300)
#   SKIP_RUN       set to 1 to probe an already-running server (for local tests)
#   SMOKE_CTX      rewrite --ctx-size in the image's llama-swap config to this
#                  before running. Production context sizes (131072 for wall-e,
#                  262144 for eve) need a KV cache far larger than a 16 GB CI
#                  runner has, so llama-server dies during model load and every
#                  inference request 502s. Loading the real model at a small
#                  context still exercises the ISA dispatch, the binary, and the
#                  GGUF shards — it just doesn't demand prod-sized memory.
#   UPSTREAM_LOGS  set to 1 to pull llama-swap's /logs on failure (see below)

set -euo pipefail

: "${IMAGE:?IMAGE is required}"
: "${HEALTH_PATH:?HEALTH_PATH is required}"
NAME="${NAME:-smoke}"
PORT="${PORT:-8080}"
HEALTH_WAIT="${HEALTH_WAIT:-300}"
MODELS_PATH="${MODELS_PATH:-}"
SKIP_RUN="${SKIP_RUN:-0}"
SMOKE_CTX="${SMOKE_CTX:-}"
UPSTREAM_LOGS="${UPSTREAM_LOGS:-0}"

base="http://localhost:${PORT}"
log_file="$(mktemp -t smoke_container.XXXXXX)"

echo "=== Smoke test: $IMAGE ==="
echo "Config: health=$HEALTH_PATH models=${MODELS_PATH:-<skip>} heavy=${HEAVY_PATH:-<skip>} port=$PORT"

if [ "$SKIP_RUN" != "1" ]; then
  # The build step loads the image into the local Docker daemon when it can, so
  # there may be nothing to pull. Both ik_llama images are now ~29 GB of shards,
  # where a third local copy fills the runner ("No space left on device"), so
  # they build with load disabled and we fetch them here — after the caller has
  # freed the buildkit cache and the shards, which is what makes room. go-4 is
  # small enough to still load directly, hence the inspect-then-pull below.
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image already in the local daemon; skipping pull."
  else
    echo "Image not loaded locally; pulling..."
    docker pull "$IMAGE"
  fi

  #
  # --network=host so curl on the runner's localhost reaches the container's
  # port directly. No --rm, so a crashed container stays inspectable.
  # Shrink the context if asked, by rewriting the config baked into the image
  # and mounting the result over it. Done here rather than in the Dockerfile so
  # the shipped image keeps its production context size.
  mount_args=""
  if [ -n "$SMOKE_CTX" ]; then
    smoke_cfg="$(mktemp -t smoke_config.XXXXXX)"
    docker run --rm --entrypoint cat "$IMAGE" /app/config.yaml \
      | sed -E "s/--ctx-size[[:space:]]+[0-9]+/--ctx-size ${SMOKE_CTX}/" > "$smoke_cfg"
    if ! grep -q -- "--ctx-size ${SMOKE_CTX}" "$smoke_cfg"; then
      echo "FAIL: could not rewrite --ctx-size to ${SMOKE_CTX} in /app/config.yaml"
      cat "$smoke_cfg"
      exit 1
    fi
    echo "Context rewritten to ${SMOKE_CTX} for the smoke run: $(grep -- '--ctx-size' "$smoke_cfg" | tr -s ' ')"
    mount_args="-v $smoke_cfg:/app/config.yaml:ro"
  fi

  # shellcheck disable=SC2086  # mount_args is intentionally word-split
  container_id="$(docker run -d --name "smoke-${NAME}-$$" --network=host --shm-size=2g $mount_args "$IMAGE")"
  docker logs -f "$container_id" > "$log_file" 2>&1 &
  logs_pid=$!
  # shellcheck disable=SC2064  # expand ids now, not at trap time
  trap "kill $logs_pid 2>/dev/null || true; docker stop $container_id >/dev/null 2>&1 || true; docker rm $container_id >/dev/null 2>&1 || true" EXIT
  echo "Container: $container_id (logs: $log_file)"
fi

dump_logs() {
  echo "--- container logs (last 80 lines) ---"
  tail -80 "$log_file" 2>/dev/null || echo "(no logs captured)"

  # llama-swap does NOT put the spawned model's stdout/stderr on the container's
  # stdout — it keeps it in a per-model log monitor exposed at /logs. Without
  # this, a model that fails to load shows up only as an opaque
  # "exit status N" with no reason, which is exactly what stalled the
  # earlier 127 and exit-1 investigations.
  if [ "$UPSTREAM_LOGS" = "1" ]; then
    echo "--- llama-swap upstream /logs (last 80 lines) ---"
    curl -s --max-time 10 "${base}/logs" 2>/dev/null | tail -80 || echo "(could not fetch /logs)"
  fi
}

# Print an endpoint's HTTP status, or 000 if the connection failed.
#
# curl already prints 000 to stdout on connection-refused, but it ALSO exits
# non-zero — and under `set -e` a bare `code=$(curl ...)` assignment inherits
# that status and kills the script. `|| true` inside the substitution is what
# keeps the failure local. This bit the CI twice; leave the `|| true` alone.
http_code() {
  local url="$1" out="${2:-/dev/null}"
  local code
  code="$(curl -s -o "$out" -w '%{http_code}' "$url" 2>/dev/null || true)"
  echo "${code:-000}"
}

# ---------- wait for the server to bind ----------
attempts=$(( HEALTH_WAIT / 2 ))
up=0
for i in $(seq 1 "$attempts"); do
  code="$(http_code "${base}${HEALTH_PATH}")"
  if [ "$code" != "000" ]; then
    echo "  $HEALTH_PATH responded with $code after $(( i * 2 ))s"
    up=1
    break
  fi
  sleep 2
done

if [ "$up" = "0" ]; then
  echo "FAIL: $HEALTH_PATH never responded within ${HEALTH_WAIT}s (connection refused)"
  dump_logs
  exit 1
fi

# ---------- /health must be a real 200, not a redirect ----------
health_body="$(mktemp -t smoke_health.XXXXXX)"
code="$(http_code "${base}${HEALTH_PATH}" "$health_body")"
echo "$HEALTH_PATH -> HTTP $code, body: $(cat "$health_body")"
if [ "$code" != "200" ]; then
  echo "FAIL: $HEALTH_PATH returned $code, expected 200"
  dump_logs
  exit 1
fi

# ---------- Zen dispatch sanity check (ik_llama images only) ----------
if grep -q llama-server-wrapper "$log_file" 2>/dev/null; then
  echo "Zen dispatch: $(grep -m1 llama-server-wrapper "$log_file")"
fi

# ---------- /v1/models (optional) ----------
if [ -n "$MODELS_PATH" ]; then
  models_body="$(mktemp -t smoke_models.XXXXXX)"
  code="$(http_code "${base}${MODELS_PATH}" "$models_body")"
  echo "$MODELS_PATH -> HTTP $code"
  if [ "$code" != "200" ]; then
    echo "FAIL: $MODELS_PATH returned $code, expected 200"
    cat "$models_body" 2>/dev/null || true
    dump_logs
    exit 1
  fi
  echo "$MODELS_PATH body: $(cat "$models_body")"
fi

# ---------- heavy endpoint (model load + tiny inference) ----------
if [ -n "${HEAVY_PATH:-}" ]; then
  echo "Hitting $HEAVY_PATH (timeout ${HEAVY_TIMEOUT}s)..."
  heavy_body="$(mktemp -t smoke_heavy.XXXXXX)"
  code="$(curl -s -o "$heavy_body" -w '%{http_code}' \
      -X POST "${base}${HEAVY_PATH}" \
      -H 'Content-Type: application/json' \
      -d "$HEAVY_BODY" \
      --max-time "$HEAVY_TIMEOUT" 2>/dev/null || true)"
  code="${code:-000}"
  echo "$HEAVY_PATH -> HTTP $code"
  if [ "$code" != "200" ]; then
    echo "FAIL: $HEAVY_PATH returned $code, expected 200"
    echo "Response body:"
    head -c 500 "$heavy_body" 2>/dev/null || true
    echo
    dump_logs
    exit 1
  fi
  echo "Heavy response (first 500 bytes):"
  head -c 500 "$heavy_body"
  echo
fi

echo "=== Smoke test PASSED ==="
