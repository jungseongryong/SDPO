#!/bin/bash
# =============================================================================
# Prepare math datasets for AntiSD recipe (self-contained, HuggingFace-only).
#
# Downloads + preprocesses:
#   training:    BytedTsinghua-SIA/DAPO-Math-17k  → datasets/math/train.parquet
#   evaluation:  math-ai/aime25                    → datasets/math/aime25/test.parquet
#                HuggingFaceH4/aime_2024           → datasets/math/aime_2024/test.parquet
#                MathArena/hmmt_feb_2025           → datasets/math/hmmt25/test.parquet
#
# These are exactly the parquets the recipe scripts under recipe/antisd/ expect
# (`datasets/math/train.parquet`, `datasets/math/aime25/test.parquet`).
#
# Usage:
#   bash data/prepare_antisd.sh                         # all four datasets
#   bash data/prepare_antisd.sh --train-only            # just DAPO-Math
#   bash data/prepare_antisd.sh --output_dir /custom    # custom output root
#   bash data/prepare_antisd.sh --extra-eval beyondaime # add other test sets
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/datasets/math"

TRAIN_ONLY=false
EXTRA_EVAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)  OUTPUT_DIR="$2"; shift 2 ;;
        --train-only)  TRAIN_ONLY=true; shift ;;
        --extra-eval)  EXTRA_EVAL+=("$2"); shift 2 ;;
        -h|--help)
            sed -n '3,20p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

step() { echo -e "\n\033[1;32m===> $1\033[0m"; }
info() { echo "     $1"; }

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ── Training set: DAPO-Math-17k ───────────────────────────────
step "Downloading + preprocessing DAPO-Math-17k (train)"
python3 "${SCRIPT_DIR}/preprocess_math_datasets.py" \
    --output_dir "$OUTPUT_DIR" --datasets dapomath

# Recipe scripts expect $OUTPUT_DIR/train.parquet (no subdir).
# preprocess_math_datasets.py writes to $OUTPUT_DIR/dapo-math/train.parquet.
# Link/copy to the canonical recipe path so launchers work as-is.
TRAIN_SRC="${OUTPUT_DIR}/dapo-math/train.parquet"
TRAIN_DST="${OUTPUT_DIR}/train.parquet"
if [[ -f "$TRAIN_SRC" ]]; then
    if [[ -e "$TRAIN_DST" || -L "$TRAIN_DST" ]]; then
        rm -f "$TRAIN_DST"
    fi
    # Symlink keeps a single source of truth; cp also works on Windows-mounted FS
    ln -s "dapo-math/train.parquet" "$TRAIN_DST"
    info "Linked: $TRAIN_DST → dapo-math/train.parquet"
fi

# ── Evaluation sets ─────────────────────────────────────────────────────
if [[ "$TRAIN_ONLY" == "false" ]]; then
    step "Downloading + preprocessing eval sets (aime25, aime_2024, hmmt25)"
    python3 "${SCRIPT_DIR}/preprocess_math_datasets.py" \
        --output_dir "$OUTPUT_DIR" \
        --datasets aime25 aime_2024 hmmt25

    if [[ ${#EXTRA_EVAL[@]} -gt 0 ]]; then
        step "Extra eval sets: ${EXTRA_EVAL[*]}"
        python3 "${SCRIPT_DIR}/preprocess_math_datasets.py" \
            --output_dir "$OUTPUT_DIR" \
            --datasets "${EXTRA_EVAL[@]}"
    fi
fi

# ── Summary ─────────────────────────────────────────────────────────────
step "Done!"
echo "Train: $OUTPUT_DIR/train.parquet  →  $(readlink "$TRAIN_DST" 2>/dev/null || echo '(direct)')"
echo "Test sets:"
for d in aime25 aime_2024 hmmt25 amc23 math500 olympiadbench minervamath beyondaime aime26 "${EXTRA_EVAL[@]}"; do
    f="$OUTPUT_DIR/$d/test.parquet"
    [[ -f "$f" ]] && echo "  $f"
done
echo ""
echo "Use in recipe/antisd/* — already wired to:"
echo "  data.train_files=[$OUTPUT_DIR/train.parquet]"
echo "  data.val_files=[$OUTPUT_DIR/aime25/test.parquet]"
