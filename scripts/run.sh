#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    printf '%s\n' 'Project environment is missing. Run bash scripts/setup.sh first.' >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    printf '%s\n' 'Usage: bash scripts/run.sh python -m palmcity.data --help' >&2
    exit 2
fi
exec "$@"
