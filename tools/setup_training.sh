#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UV_HTTP_RETRIES=1
export UV_HTTP_TIMEOUT=25
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.runtime/python"
export UV_CACHE_DIR="$PROJECT_ROOT/.runtime/uv-cache"
export UV_PYTHON_INSTALL_BIN=0
"$PROJECT_ROOT/.venv-motion/bin/python" "$PROJECT_ROOT/tools/use_local_torch.py"
"$PROJECT_ROOT/.venv-motion/bin/python" "$PROJECT_ROOT/tools/use_local_warp.py"
exec "$PROJECT_ROOT/.venv-motion/bin/uv" sync --project "$PROJECT_ROOT/envs/training" --python 3.12 "$@"
