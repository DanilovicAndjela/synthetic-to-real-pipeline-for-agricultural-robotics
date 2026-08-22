#!/usr/bin/env bash
# The run matrix. Execute INSIDE the TAO container.
#
#   docker run -it --rm --gpus all -v $WORKSPACE:/workspace --shm-size=16g \
#     $(cat $WORKSPACE/.tao_image) /bin/bash
#   bash /workspace/scripts/04_run_matrix.sh smoke
set -euo pipefail

W=/workspace
SPECS=$W/specs
RES=$W/results
MODE="${1:-all}"
NS="${NS:-50 100 200 400}"

ckpt() {  
  find "$1/train" -name '*.pth' -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

# smoke
if [ "$MODE" = "smoke" ]; then
  echo "MOKE TEST - 3 epochs. Ignore the mAP."
  rtdetr train -e "$SPECS/rtdetr_A_synth.yaml" \
    results_dir="$RES/smoke" \
    train.num_epochs=3 train.checkpoint_interval=1 train.validation_interval=1
  exit 0
fi

# A
if [[ "$MODE" =~ ^(A|all)$ ]]; then
  echo "RUN A - synthetic only (~120 epochs)"
  rtdetr train -e "$SPECS/rtdetr_A_synth.yaml"

  A_CKPT=$(ckpt "$RES/A_synth")
  echo "A checkpoint: $A_CKPT"
  echo "$A_CKPT" > "$RES/A_synth/BEST_CKPT"

  echo "A evaluated on REAL test - this is raw sim-to-real gap"
  rtdetr evaluate -e "$SPECS/rtdetr_A_synth.yaml" \
    evaluate.checkpoint="$A_CKPT" \
    evaluate.results_dir="$RES/A_synth/eval_real" \
    dataset.test_data_sources.image_dir=$W/data/real/images \
    dataset.test_data_sources.json_file=$W/data/real/annotations/real_test.json
fi

A_CKPT=$(cat "$RES/A_synth/BEST_CKPT" 2>/dev/null || echo "")

# B, C
for N in $NS; do
  TRAIN_JSON=$W/data/real/annotations/real_train_n${N}.json
  [ -f "$TRAIN_JSON" ] || { echo "skip N=$N (no $TRAIN_JSON)"; continue; }

  if [[ "$MODE" =~ ^(B|all)$ ]]; then
    echo "RUN B  N=$N  (synthetic -> real)"
    [ -n "$A_CKPT" ] || { echo "ERROR: run A first"; exit 1; }
    rtdetr train -e "$SPECS/rtdetr_B_finetune.yaml" \
      results_dir="$RES/B_finetune_n${N}" \
      train.pretrained_model_path="$A_CKPT" \
      dataset.train_data_sources.json_file="$TRAIN_JSON"

    rtdetr evaluate -e "$SPECS/rtdetr_B_finetune.yaml" \
      evaluate.checkpoint="$(ckpt "$RES/B_finetune_n${N}")" \
      evaluate.results_dir="$RES/B_finetune_n${N}/eval_real" \
      dataset.test_data_sources.image_dir=$W/data/real/images \
      dataset.test_data_sources.json_file=$W/data/real/annotations/real_test.json
  fi

  if [[ "$MODE" =~ ^(C|all)$ ]]; then
    echo "### RUN C  N=$N  (real only baseline)"
    rtdetr train -e "$SPECS/rtdetr_C_real_only.yaml" \
      results_dir="$RES/C_real_only_n${N}" \
      dataset.train_data_sources.json_file="$TRAIN_JSON"

    rtdetr evaluate -e "$SPECS/rtdetr_C_real_only.yaml" \
      evaluate.checkpoint="$(ckpt "$RES/C_real_only_n${N}")" \
      evaluate.results_dir="$RES/C_real_only_n${N}/eval_real" \
      dataset.test_data_sources.image_dir=$W/data/real/images \
      dataset.test_data_sources.json_file=$W/data/real/annotations/real_test.json
  fi
done
