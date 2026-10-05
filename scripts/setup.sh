#!/usr/bin/env bash
set -euo pipefail
export PALMCITY_MIN_FREE_GIB="${PALMCITY_MIN_FREE_GIB:-20}"
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$PALMCITY_CODE_ROOT"
python_path="${PALMCITY_PYTHON:-python3.12}"
if ! command -v "$python_path" >/dev/null; then
    printf '%s\n' 'Install Python 3.12 separately, or set PALMCITY_PYTHON to its executable.' >&2
    exit 1
fi
"$python_path" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
command -v uv >/dev/null || { printf '%s\n' 'Install uv separately before setup.' >&2; exit 1; }
uv sync --locked --python "$python_path" --group dev
