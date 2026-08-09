#!/bin/sh
# /app/llama-server wrapper for the ik_llama multi-ISA images.
#
# Detects the host CPU features via /proc/cpuinfo and exec's the matching
# pre-built binary bundled alongside this script:
#
#   avx512vnni  ->  Zen 5 (Turin)   ->  /app/llama-server.zen5
#   avx512vl    ->  Zen 4 (Genoa)   ->  /app/llama-server.zen4
#   avx512f     ->  Zen 3 (Milan)   ->  /app/llama-server.zen3
#
# Falls back to a clear error on CPUs without AVX-512F (anything older
# than Zen 3 won't run ik_llama.cpp with the Zen3+ paths we built).
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
    *avx512f*)    bin=/app/llama-server.zen3 ;;
    *)
        echo "[llama-server-wrapper] FATAL: CPU lacks AVX-512F." >&2
        echo "[llama-server-wrapper] ik_llama.cpp requires at least Zen 3 (Milan)." >&2
        echo "[llama-server-wrapper] Detected flags: $flags" >&2
        exit 1
        ;;
esac

# Emit the dispatch decision to stderr so it lands in `kubectl logs`
# without polluting llama-server's stdout (which llama-swap parses).
echo "[llama-server-wrapper] CPU detected (${bin##*.}), exec'ing $bin" >&2
exec "$bin" "$@"