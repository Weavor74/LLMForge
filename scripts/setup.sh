#!/usr/bin/env bash
# Install LLMForge on this machine, whatever it is.
#
# The project pins a CUDA 13 build of torch, which needs an NVIDIA driver of 580 or
# newer. Plenty of perfectly good machines run older drivers, so this checks what is
# actually here and installs a matching build instead of failing at the first CUDA
# call with a message about no device being found.
#
# Nothing here needs root.

set -euo pipefail

cd "$(dirname "$0")/.."
say() { printf '  %s\n' "$*"; }

echo
echo "LLMForge setup"
echo

# --- what machine is this ----------------------------------------------------
ARCH="$(uname -m)"
OS="$(uname -s)"
say "platform     $OS $ARCH"

DRIVER=""
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
fi

# --- pick a CUDA build the driver can actually run ---------------------------
# torch 2.13 ships for cu126, cu129 and cu130, so the code never changes — only
# which index the wheel comes from.
CUDA_INDEX=""
if [ -z "$DRIVER" ]; then
    say "gpu          none detected — installing the CPU build"
    say "             (planning and the interface work; training will not)"
else
    MAJOR="${DRIVER%%.*}"
    say "driver       $DRIVER"
    if   [ "$MAJOR" -ge 580 ]; then CUDA_INDEX="cu130"
    elif [ "$MAJOR" -ge 525 ]; then CUDA_INDEX="cu126"
    else
        say "driver       too old for any supported torch build (need 525+)"
        say "             update the NVIDIA driver, then re-run this script"
        exit 1
    fi
    say "cuda build   $CUDA_INDEX"
fi

# --- uv ----------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    say "uv           installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi
say "uv           $(uv --version)"

# --- install -----------------------------------------------------------------
echo
say "installing dependencies (this pulls several GB of CUDA wheels)"
uv sync --extra finetune

# The lockfile pins cu130. On an older driver, swap torch for a build it can run.
# Everything else in the lock is unaffected.
if [ -n "$CUDA_INDEX" ] && [ "$CUDA_INDEX" != "cu130" ]; then
    echo
    say "replacing torch with the $CUDA_INDEX build for driver $DRIVER"
    uv pip install --reinstall-package torch \
        "torch>=2.13.0" --index-url "https://download.pytorch.org/whl/$CUDA_INDEX"
fi

# --- the GUI -----------------------------------------------------------------
# Committed prebuilt, so Node is only needed to change the frontend.
if [ ! -f web/dist/index.html ]; then
    if command -v npm >/dev/null 2>&1; then
        echo
        say "building the interface"
        (cd web && npm install --silent && npm run build >/dev/null)
    else
        say "interface    not built and npm is absent — the CLI still works"
    fi
fi

# --- verify ------------------------------------------------------------------
echo
uv run llmforge doctor --skip-compile || true

echo
say "next:  uv run llmforge serve        # then open http://127.0.0.1:8000"
say "       uv run llmforge --help       # everything from a terminal"
echo
