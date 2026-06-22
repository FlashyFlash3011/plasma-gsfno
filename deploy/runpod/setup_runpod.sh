#!/usr/bin/env bash
# Set up a RunPod (CUDA) box to train/eval plasma-gsfno.
#
# Expected layout (clone BOTH repos side by side; Solaris MUST be named "Solaris"):
#   workdir/
#   ├── Solaris/          git clone https://github.com/FlashyFlash3011/Solaris.git Solaris
#   └── plasma-gsfno/     git clone <your plasma-gsfno remote> plasma-gsfno
#
# Run from anywhere inside plasma-gsfno:  bash deploy/runpod/setup_runpod.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")/../.." && pwd)"     # plasma-gsfno root
PARENT="$(dirname "$HERE")"
SOLARIS="$PARENT/Solaris"

echo "==> plasma-gsfno: $HERE"
echo "==> expecting Solaris at: $SOLARIS"

if [ ! -d "$SOLARIS" ]; then
  echo "==> Solaris not found; cloning..."
  git clone https://github.com/FlashyFlash3011/Solaris.git "$SOLARIS"
fi

python -m pip install --upgrade pip

# CUDA build of torch. Change cu124 to match the pod's CUDA toolkit if needed.
echo "==> installing CUDA torch (cu124)"
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "==> installing runtime deps"
python -m pip install numpy h5py hydra-core omegaconf tqdm scipy matplotlib

# Install Solaris (provides solaris.models.FNO etc.) and plasma-gsfno.
# torch is already present, so these won't pull a CPU wheel over it.
echo "==> installing Solaris + plasma-gsfno (editable)"
python -m pip install -e "$SOLARIS"
python -m pip install -e "$HERE"

echo "==> verifying CUDA"
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("CUDA available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA not available — check the pod image / torch build")
PY

echo "==> done. Upload your dataset to $HERE/data/equilibria.h5, then run:"
echo "    bash deploy/runpod/train_and_eval.sh"
