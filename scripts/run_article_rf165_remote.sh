#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
experiment_root="${project_root}/outputs/experiments/article_rf165"
remote_log_root="${experiment_root}/remote_logs"

cd "${project_root}"
mkdir -p "${remote_log_root}"

export PYTHONPATH="${project_root}/src"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

date -Is > "${remote_log_root}/started_at.txt"

python - <<'PY'
import json
import platform
import subprocess
from pathlib import Path

import numpy
import scipy
import torch
import yaml

output_path = Path("outputs/experiments/article_rf165/remote_logs/environment.json")
payload = {
    "schema_version": 1,
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pyyaml": yaml.__version__,
    "torch": torch.__version__,
    "cuda_compiled": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
}
if not payload["cuda_available"]:
    raise RuntimeError("CUDA is unavailable in the remote environment.")
output_path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

python -m compileall -q src

if [[ ! -f "${experiment_root}/audits/context_safety.json" ]]; then
    python -m bee_counter_bench.cli audit-article-context \
        --plan configs/article_rf165.yaml
fi

python -m bee_counter_bench.cli smoke-test \
    --device cuda \
    --model-config configs/model_rf165.yaml

python -m bee_counter_bench.cli run-article-suite \
    --plan configs/article_rf165.yaml \
    --parallel 4 \
    --device cuda

python -m bee_counter_bench.cli summarize-article-experiment \
    --plan configs/article_rf165.yaml

date -Is > "${remote_log_root}/experiment_completed_at.txt"

archive_path="${project_root}/article_rf165_results.tar.gz"
tar \
    --exclude="outputs/experiments/article_rf165/remote_logs/run_job.log" \
    -czf "${archive_path}" \
    outputs/experiments/article_rf165
sha256sum "${archive_path}" > "${archive_path}.sha256"

date -Is > "${remote_log_root}/finished_at.txt"
