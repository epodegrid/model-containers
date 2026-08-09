#!/bin/sh
# /app/llama-server wrapper for the ik_llama multi-ISA images.
#
# Detects the host CPU features via /proc/cpuinfo and exec's the matching
# pre-built binary bundled alongside this script:
#
#   avx512vnni  ->  Zen 5 (Turin)   ->  /app/llama-server.zen5
#   avx512vl    ->  Zen 4 (Genoa)   ->  /app/llama-server.zen4
#   avx2        ->  Zen 3 (Milan)   ->  /app/llama-server.zen3
#
# NOTE: Zen 3 / EPYC Milan has NO AVX-512 — AMD introduced it with Zen 4
# (Genoa). The znver3 binary is therefore an AVX2 build, and gating it on
# avx512f (as this script used to) meant Milan fell through to the FATAL
# branch and the container refused to start. That is precisely our target
# SKU: Azure E48as_v5 is the EPYC 7763v (Milan). Match on avx2 instead.
#
# Falls back to a clear error only on CPUs without AVX2, which no Zen-era
# EPYC lacks.
#
# The wrapper is invoked by llama-swap per model spawn, so the dispatch
# cost is one-time per pod start (or per model load), not per request.

set -e

# /proc/cpuinfo's `flags` line lists all CPU feature flags for the
# first logical core (sufficient for ISA detection — features are
# uniform across cores in any sane VM SKU).
flags=$(awk '/^flags[[:space:]]*:/{print; exit}' /proc/cpuinfo)

case "$flags" in
    *avx512vnni*) bin=/app/llama-server.zen5 ;;
    *avx512vl*)   bin=/app/llama-server.zen4 ;;
    *avx2*)       bin=/app/llama-server.zen3 ;;
    *)
        echo "[llama-server-wrapper] FATAL: CPU lacks AVX2." >&2
        echo "[llama-server-wrapper] ik_llama.cpp requires at least Zen 3 (Milan)." >&2
        echo "[llama-server-wrapper] Detected flags: $flags" >&2
        exit 1
        ;;
esac

# Emit the dispatch decision to stderr so it lands in `kubectl logs`
# without polluting llama-server's stdout (which llama-swap parses).
echo "[llama-server-wrapper] CPU detected (${bin##*.}), exec'ing $bin" >&2
exec "$bin" "$@"