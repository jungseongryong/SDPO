#!/bin/bash
set -euo pipefail

# =============================================================================
# Prepare DAPO-Math (cleaned) for AntiSD training
#
# Source: BytedTsinghua-SIA/DAPO-Math-17k (HuggingFace)
# Output: datasets/math/dapo-math/train.parquet
#
# Usage:
#   bash data/prepare_dapomath.sh
#   bash data/prepare_dapomath.sh --output_dir /path/to/datasets/math
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

echo "Preprocessing DAPO-Math-17k -> ${OUTPUT_DIR}/dapo-math/"
python3 "${SCRIPT_DIR}/preprocess_math_datasets.py" \
    --output_dir "$OUTPUT_DIR" \
    --datasets dapomath

echo ""
echo "Done! Use in training scripts:"
echo "  data.train_files=[${OUTPUT_DIR}/dapo-math/train.parquet]"
