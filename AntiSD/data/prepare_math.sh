#!/bin/bash
set -euo pipefail

# =============================================================================
# Copy pre-processed math parquets from a verl is_shape recipe directory.
#
# This is a legacy convenience script for users who already have a directory
# of pre-processed parquets (e.g. the dapo_math 1.79M-row train set bundled
# with verl/recipe/is_shape/data on shared infra). For a from-scratch HF
# download instead, use data/prepare_antisd.sh.
#
# Sources: <SOURCE_DIR>/{dapo_math,amc23,aime25,aime_2024,math500}/{train,test}.parquet
# Output:  datasets/math/  (train.parquet + per-dataset test parquets)
#
# Usage:
#   bash data/prepare_math.sh --source /path/to/verl/recipe/is_shape/data
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/datasets/math"

# Source must be provided explicitly — no default mount path baked in.
SOURCE_DIR="${SOURCE_DIR:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source) SOURCE_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$SOURCE_DIR" ]]; then
    echo "ERROR: --source <dir> is required (e.g. /path/to/verl/recipe/is_shape/data)"
    echo "       For a from-scratch HuggingFace download, use data/prepare_antisd.sh instead."
    exit 1
fi

step() { echo -e "\n\033[1;32m===> $1\033[0m"; }

# -----------------------------------------------------------------------------
# Validate source
# -----------------------------------------------------------------------------
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "ERROR: Source directory not found: $SOURCE_DIR"
    echo "       Specify --source /path/to/verl/recipe/is_shape/data"
    exit 1
fi

for f in dapo_math/train.parquet amc23/test.parquet aime25/test.parquet aime_2024/test.parquet; do
    if [[ ! -f "$SOURCE_DIR/$f" ]]; then
        echo "ERROR: Missing $SOURCE_DIR/$f"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Copy data
# -----------------------------------------------------------------------------
mkdir -p "$OUTPUT_DIR"

step "Copy training data (dapo_math, ~1.79M samples)"
cp -v "$SOURCE_DIR/dapo_math/train.parquet" "$OUTPUT_DIR/train.parquet"

step "Copy test data"
for dataset in amc23 aime25 aime_2024 math500; do
    src="$SOURCE_DIR/$dataset/test.parquet"
    if [[ -f "$src" ]]; then
        mkdir -p "$OUTPUT_DIR/$dataset"
        cp -v "$src" "$OUTPUT_DIR/$dataset/test.parquet"
    else
        echo "  SKIP: $src (not found)"
    fi
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
step "Done!"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Train: $OUTPUT_DIR/train.parquet"
echo "Test:"
for dataset in amc23 aime25 aime_2024 math500; do
    f="$OUTPUT_DIR/$dataset/test.parquet"
    [[ -f "$f" ]] && echo "  $f"
done
echo ""
echo "Use in training scripts:"
echo "  data.train_files=[${OUTPUT_DIR}/train.parquet]"
echo "  data.val_files=[${OUTPUT_DIR}/amc23/test.parquet,${OUTPUT_DIR}/aime25/test.parquet,${OUTPUT_DIR}/aime_2024/test.parquet]"
