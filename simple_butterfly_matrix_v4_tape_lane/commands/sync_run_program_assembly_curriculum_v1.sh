#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[program curriculum] sync branch=$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/program_assembly_curriculum_v1_${TS}}"
mkdir -p "$REPORT_DIR"

MAIN="simple_butterfly_matrix_v4_tape_lane/program_assembly_curriculum_v1.py"

DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-fp16}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TRAIN_STEPS="${TRAIN_STEPS:-250}"
EVAL_BATCHES="${EVAL_BATCHES:-30}"
DIM="${DIM:-32}"
HIDDEN="${HIDDEN:-192}"
STEPS="${STEPS:-6}"
LR="${LR:-8e-4}"
SEED="${SEED:-42}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
SIGNAL_STRUCTURE="${SIGNAL_STRUCTURE:-1}"
INACTIVE_TOKEN_WEIGHT="${INACTIVE_TOKEN_WEIGHT:-0.08}"
BOUNDARY_MAX_PEAKS="${BOUNDARY_MAX_PEAKS:-4}"
ROUTE_ENTROPY_MAX="${ROUTE_ENTROPY_MAX:-1.25}"

SAVE_ARGS=()
if [ "$SAVE_CHECKPOINT" != "0" ] && [ "$SAVE_CHECKPOINT" != "false" ] && [ "$SAVE_CHECKPOINT" != "False" ]; then
  SAVE_ARGS+=(--save-checkpoint)
fi
if [ "$SIGNAL_STRUCTURE" = "0" ] || [ "$SIGNAL_STRUCTURE" = "false" ] || [ "$SIGNAL_STRUCTURE" = "False" ]; then
  SAVE_ARGS+=(--no-signal-structure)
fi

echo "[program curriculum] py_compile"
python -m py_compile "$MAIN"
python "$MAIN" --help >/tmp/program_assembly_curriculum_v1_help.txt

echo "[program curriculum] run -> $REPORT_DIR" | tee "$REPORT_DIR/run_status.txt"
cat > "$REPORT_DIR/config.txt" <<EOF_CFG
BRANCH=$BRANCH
DEVICE=$DEVICE
AMP=$AMP
EPOCHS=$EPOCHS
BATCH_SIZE=$BATCH_SIZE
TRAIN_STEPS=$TRAIN_STEPS
EVAL_BATCHES=$EVAL_BATCHES
DIM=$DIM
HIDDEN=$HIDDEN
STEPS=$STEPS
LR=$LR
SEED=$SEED
SAVE_CHECKPOINT=$SAVE_CHECKPOINT
SIGNAL_STRUCTURE=$SIGNAL_STRUCTURE
INACTIVE_TOKEN_WEIGHT=$INACTIVE_TOKEN_WEIGHT
BOUNDARY_MAX_PEAKS=$BOUNDARY_MAX_PEAKS
ROUTE_ENTROPY_MAX=$ROUTE_ENTROPY_MAX
EOF_CFG

set +e
python "$MAIN" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --train-steps-per-epoch "$TRAIN_STEPS" \
  --eval-batches "$EVAL_BATCHES" \
  --dim "$DIM" \
  --hidden "$HIDDEN" \
  --steps "$STEPS" \
  --lr "$LR" \
  --seed "$SEED" \
  --inactive-token-weight "$INACTIVE_TOKEN_WEIGHT" \
  --boundary-max-peaks "$BOUNDARY_MAX_PEAKS" \
  --route-entropy-max "$ROUTE_ENTROPY_MAX" \
  --out-dir "$REPORT_DIR" \
  "${SAVE_ARGS[@]}" 2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "[program curriculum] run_status=$RUN_STATUS" | tee -a "$REPORT_DIR/run_status.txt"

STATUS="simple_butterfly_matrix_v4_tape_lane/AGENT_STATUS.md"
cat > "$STATUS" <<EOF_STATUS
# Agent Status

STATUS: PROGRAM_ASSEMBLY_CURRICULUM_DONE
VERSION: program_assembly_curriculum_v1
RUN_STATUS: $RUN_STATUS
REPORT_DIR: $REPORT_DIR

CHANGED:
- program_assembly_curriculum_v1.py: synthetic program assembly school with repeat/repair/denoise/mixed modes
- sync_run_program_assembly_curriculum_v1.sh: one-command train+logs+push

RUN_NOW:
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v1.sh

QUICK_SMOKE:
EPOCHS=2 TRAIN_STEPS=50 EVAL_BATCHES=10 BATCH_SIZE=256 bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v1.sh

CHECK_LOGS:
- REPORT_TO_CHATGPT.txt
- metrics.csv
- examples_epoch_XXX.json
- final_report.json
- train.log

BLOCKERS:
- none if run_status=0; inspect train.log otherwise

NEXT:
- if primitive_acc/route_acc/boundary_f1/full_program_exact are good, add v4.4 adapter for primitive/read/route/boundary/write controller weights
EOF_STATUS

find "$REPORT_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) -print -exec git add {} +
git add "$STATUS" "$MAIN" "simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v1.sh"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo "[program curriculum] ERROR: checkpoint staged"
  exit 1
fi
if ! git diff --cached --quiet; then
  git commit -m "Add program assembly curriculum v1 run logs $TS"
  git push origin "$BRANCH" || {
    echo "[program curriculum] push failed. Run: git pull --rebase --autostash origin $BRANCH && git push origin $BRANCH" | tee -a "$REPORT_DIR/run_status.txt"
    exit 3
  }
fi

exit "$RUN_STATUS"
