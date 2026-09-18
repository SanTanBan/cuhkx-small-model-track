#!/usr/bin/env bash
# Reproduce a submission CSV from the packaged checkpoint.
#
#   ./inference.sh <test_data_dir> [out_csv] [checkpoint]
#
# <test_data_dir> is the extracted small_model_track_test folder, i.e. the one
# holding SM_test_0001/, SM_test_0002/, ... each with its modality subfolders.
# The checkpoint defaults to checkpoints/model.pth; download it from the Kaggle
# dataset linked in README.md. Everything runs offline; no weights are fetched.
set -euo pipefail

DATA=${1:?usage: ./inference.sh <test_data_dir> [out_csv] [checkpoint]}
OUT=${2:-final_submission.csv}
CKPT=${3:-checkpoints/model.pth}
HERE=$(cd "$(dirname "$0")" && pwd)

python "$HERE/kaggle/infer.py" --data "$DATA" --out "$OUT" --ckpt "$CKPT"
echo "wrote $OUT"
