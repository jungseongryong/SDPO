#!/bin/bash
set -euo pipefail

# =============================================================================
# Prepare DeepMath-103K dataset for AntiSD training
#
# Downloads from HuggingFace: zwhe99/DeepMath-103K
# Output: datasets/math/deepmath/train.parquet
#
# Usage:
#   bash data/prepare_deepmath.sh
#   bash data/prepare_deepmath.sh --output_dir /custom/path
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/datasets/math"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "Preprocessing DeepMath-103K -> ${OUTPUT_DIR}/deepmath/"
python3 "${SCRIPT_DIR}/preprocess_math_datasets.py" \
    --output_dir "$OUTPUT_DIR" \
    --datasets deepmath

echo ""
echo "Done! Use in training scripts:"
echo "  data.train_files=[${OUTPUT_DIR}/deepmath/train.parquet]"
