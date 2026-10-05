#!/usr/bin/env bash
# Source this file for a project-local environment. Shell startup files are unchanged.
_palmcity_environment() {
    local code_root bootstrap_python worf_config temp_directory
    code_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" || return 1
    export PALMCITY_CODE_ROOT="$code_root"
    if [[ -z "${PALMCITY_WORKSPACE:-}" || "$PALMCITY_WORKSPACE" != /* ]]; then
        printf '%s\n' 'Set PALMCITY_WORKSPACE to an absolute, dedicated, user-owned directory.' >&2
        return 1
    fi
    export PYTHONPATH="$code_root/src"
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONPYCACHEPREFIX="$PALMCITY_WORKSPACE/pycache"
    worf_config="$HOME/.config/worf/huggingface-cache.sh"
    if [[ -r "$worf_config" ]]; then
        export PALMCITY_WORF_PROFILE=1
        source "$worf_config" || return 1
        if [[ -z "${HF_HUB_CACHE:-}" || ! -d "$HF_HUB_CACHE" || "$HF_HUB_CACHE" != /scratch/* ]]; then
            printf '%s\n' 'Existing Worf Hugging Face cache is unavailable; refusing fallback.' >&2
            return 1
        fi
    else
        export PALMCITY_WORF_PROFILE=0
        export HF_HOME="${HF_HOME:-$PALMCITY_WORKSPACE/cache/huggingface}"
        export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
    fi
    bootstrap_python="${PALMCITY_BOOTSTRAP_PYTHON:-python3}"
    "$bootstrap_python" -m palmcity.storage --prepare --min-free-gib "${PALMCITY_MIN_FREE_GIB:-1}" >/dev/null || return 1
    export UV_PROJECT_ENVIRONMENT="${PALMCITY_ENVIRONMENT:-$PALMCITY_WORKSPACE/.venv}"
    export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$PALMCITY_WORKSPACE/cache/uv}"
    export PIP_CACHE_DIR="$PALMCITY_WORKSPACE/cache/pip"
    export TORCH_HOME="$PALMCITY_WORKSPACE/cache/torch"
    export XDG_CACHE_HOME="$PALMCITY_WORKSPACE/cache/xdg"
    temp_directory="$("$bootstrap_python" -m palmcity.storage --temporary-directory)" || return 1
    export TMPDIR="$temp_directory"
    export UV_PYTHON_DOWNLOADS=never
    export UV_LINK_MODE=copy
    export UV_NO_PROGRESS=1
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    export WANDB_MODE=offline
    export WANDB_DIR="$PALMCITY_WORKSPACE/outputs"
    export MPLCONFIGDIR="$PALMCITY_WORKSPACE/tmp/matplotlib"
    export RUFF_CACHE_DIR="$PALMCITY_WORKSPACE/tmp/ruff-cache"
    export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -o cache_dir=$PALMCITY_WORKSPACE/verification/pytest-cache"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
    export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
}
if _palmcity_environment; then
    unset -f _palmcity_environment
else
    unset -f _palmcity_environment
    return 1 2>/dev/null || exit 1
fi
