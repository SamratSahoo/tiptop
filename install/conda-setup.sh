#!/bin/bash
#
# Conda-based setup for TiPToP with a LOCAL, EDITABLE cuTAMP checkout.
#
# This mirrors the pixi workflow (`pixi install` + `pixi run setup-planners`) but
# uses conda + pip, and installs cuTAMP from a local path in editable mode instead
# of cloning the pinned release tag. Tailored for the Princeton della cluster
# (CUDA via `module load`, GPUs are A100/H100/H200/GH200).
#
# Usage:
#   bash install/conda-setup.sh
#
# Override any default via env vars, e.g.:
#   ENV_NAME=myenv CUTAMP_DIR=/path/to/cuTAMP bash install/conda-setup.sh
#
# Notes:
#   * The local cuTAMP checkout must be on the tiptop-robot/cuTAMP v0.0.5 lineage
#     (it derives __version__ from package metadata == tiptop's REQUIRED_CUTAMP_VERSION,
#     and provides the FR3/Robotiq embodiment tiptop imports). The public NVlabs
#     cuTAMP is an older lineage and is missing those modules. To re-base a fork:
#       git -C "$CUTAMP_DIR" remote add upstream https://github.com/tiptop-robot/cuTAMP.git
#       git -C "$CUTAMP_DIR" fetch upstream --tags && git -C "$CUTAMP_DIR" checkout -b tiptop-v0.0.5 v0.0.5
#   * cuRobo CUDA kernels are built for sm_80 (A100) and sm_90 (H100/H200/GH200).
#     Building does not require a live GPU as long as TORCH_CUDA_ARCH_LIST is set.
#   * The ZED Python API is only installed if the ZED SDK is present at
#     /usr/local/zed (not the case on the cluster login/compute nodes).

set -eo pipefail

# ---- Configuration -------------------------------------------------------------
ENV_NAME="${ENV_NAME:-tiptop}"
PY_VERSION="${PY_VERSION:-3.12}"
CUDA_MODULE="${CUDA_MODULE:-cudatoolkit/12.6}"
TORCH_PKGS="${TORCH_PKGS:-torch==2.5.1 torchvision==0.20.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export MAX_JOBS="${MAX_JOBS:-4}"

TIPTOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTAMP_DIR="${CUTAMP_DIR:-/scratch/gpfs/TSILVER/ss1824/cuTAMP}"
CUROBO_DIR="${CUROBO_DIR:-$TIPTOP_DIR/curobo}"
CUROBO_REPO="${CUROBO_REPO:-https://github.com/williamshen-nz/curobo.git}"
TIPTOP_EXTRAS="${TIPTOP_EXTRAS:-sam-server,test,dev}"   # add 'ur5' if you use a UR5 arm

# Run a command inside the conda env, streaming output.
run() { conda run --no-capture-output -n "$ENV_NAME" "$@"; }

# ---- 1. Conda environment ------------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "==> Conda env '$ENV_NAME' already exists, reusing it"
else
    echo "==> Creating conda env '$ENV_NAME' (Python $PY_VERSION)"
    conda create -y -n "$ENV_NAME" "python=$PY_VERSION" pip setuptools wheel
fi

# ---- 2. PyTorch (CUDA wheels) --------------------------------------------------
echo "==> Installing PyTorch: $TORCH_PKGS (from $TORCH_INDEX)"
run python -m pip install $TORCH_PKGS --index-url "$TORCH_INDEX"

# ---- 3. CUDA toolkit (needed to build cuRobo) ----------------------------------
source /usr/share/Modules/init/bash 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null || true
module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
echo "==> Using CUDA_HOME=$CUDA_HOME ($(command -v nvcc))"

# ---- 4. cuTAMP (local, editable) -----------------------------------------------
echo "==> Installing local cuTAMP (editable) from $CUTAMP_DIR"
run python -m pip install -e "$CUTAMP_DIR"

# ---- 5. cuRobo (build from source) ---------------------------------------------
if [ ! -d "$CUROBO_DIR" ]; then
    echo "==> Cloning cuRobo into $CUROBO_DIR"
    git clone --depth 1 "$CUROBO_REPO" "$CUROBO_DIR"
fi
echo "==> Installing cuRobo build/runtime deps (installed --no-deps below)"
run python -m pip install ninja pybind11 numpy-quaternion setuptools_scm importlib_resources scikit-image
echo "==> Building cuRobo (CUDA kernels; can take 10-20 min)"
( cd "$CUROBO_DIR" && run python -m pip install -e . --no-build-isolation --no-deps )

# ---- 6. SAM-2, then TiPToP (editable) ------------------------------------------
# SAM2_BUILD_CUDA=0 skips the optional CUDA postprocessing extension (not needed
# for image segmentation; no GPU on the login node to build it).
echo "==> Installing SAM-2 (CUDA ext disabled)"
SAM2_BUILD_CUDA=0 run python -m pip install --no-build-isolation \
    "SAM-2 @ git+https://github.com/facebookresearch/segment-anything-2.git"

echo "==> Installing TiPToP (editable) with extras [$TIPTOP_EXTRAS]"
SAM2_BUILD_CUDA=0 run python -m pip install -e "$TIPTOP_DIR[$TIPTOP_EXTRAS]" --no-build-isolation

# ---- 7. ZED Python API (optional) ----------------------------------------------
if [ -f /usr/local/zed/get_python_api.py ]; then
    echo "==> Installing ZED Python API"
    run python /usr/local/zed/get_python_api.py
else
    echo "==> ZED SDK not found at /usr/local/zed -- skipping (only needed for ZED cameras)"
fi

# ---- 8. Verify -----------------------------------------------------------------
echo "==> Verifying installation"
run python - <<'PY'
import torch, cutamp, curobo, tiptop
from tiptop.utils import check_cutamp_version
check_cutamp_version()
print("torch     :", torch.__version__)
print("cutamp    :", cutamp.__version__)
print("curobo    :", getattr(curobo, "__version__", "?"))
print("tiptop    :", getattr(tiptop, "__version__", "?"))
print("OK: imports + cuTAMP version check passed")
PY

echo
echo "==> Done. Activate with:  conda activate $ENV_NAME"
echo "    (For commands that build/run CUDA, also: module load $CUDA_MODULE)"
