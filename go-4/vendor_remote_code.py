"""Vendor Hugging Face `trust_remote_code` modules into the model directories.

Both nomic-embed-text-v1.5 and nomic-embed-vision-v1.5 declare their model
code in a *different* repo, via the `auto_map` cross-repo syntax:

    "AutoModel": "nomic-ai/nomic-bert-2048--modeling_hf_nomic_bert.NomicBertModel"
                  ^^^^^^^^^^^^^^^^^^^^^^^^--

The `<repo>--` prefix tells transformers to fetch that repo from the Hub at
load time. Baking only the model snapshots is therefore not enough: the first
`from_pretrained(..., trust_remote_code=True)` still reaches out to
huggingface.co and, in an air-gapped cluster, crashloops with

    OSError: We couldn't connect to 'https://huggingface.co' to load the
    files, and couldn't find them in the cached files.

HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE do not help — they only turn the
download into a cache miss, because the code was never in the image.

This script copies the code modules into each model directory and rewrites
`auto_map` to drop the `<repo>--` prefix, so transformers resolves the classes
from files sitting next to config.json. No network, no HF cache layout to keep
in sync. It is idempotent: entries without a `--` are left alone.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def vendor(code_dir: Path, model_dirs: list[Path]) -> None:
    modules = sorted(code_dir.glob("*.py"))
    if not modules:
        raise SystemExit(f"no *.py modules found in {code_dir}")

    for model_dir in model_dirs:
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise SystemExit(f"missing config.json in {model_dir}")

        for module in modules:
            shutil.copy2(module, model_dir / module.name)
        print(f"{model_dir}: vendored {', '.join(m.name for m in modules)}")

        config = json.loads(config_path.read_text())
        auto_map = config.get("auto_map") or {}
        rewritten = {}
        for key, target in auto_map.items():
            # "repo/name--file.Class" -> "file.Class"; leave local refs as-is.
            local = target.split("--", 1)[1] if "--" in target else target
            if local != target:
                print(f"  auto_map {key}: {target} -> {local}")
            rewritten[key] = local

        if rewritten:
            config["auto_map"] = rewritten
            config_path.write_text(json.dumps(config, indent=2) + "\n")

        # A stale reference anywhere means we would still hit the network.
        remaining = [v for v in rewritten.values() if "--" in v]
        if remaining:
            raise SystemExit(f"{model_dir}: cross-repo refs survived: {remaining}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <code_dir> <model_dir> [model_dir...]")
    vendor(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
