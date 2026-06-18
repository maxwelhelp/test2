#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "[program curriculum v2] sync branch=$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

TS="$(date +%Y%m%d_%H%M%S)"
MAIN="simple_butterfly_matrix_v4_tape_lane/program_assembly_curriculum_v2_structured.py"
MODE="${MODE:-train_online}"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/program_assembly_curriculum_v2_structured_${TS}}"
DATASET_OUT="${DATASET_OUT:-simple_butterfly_matrix_v4_tape_lane/program_datasets/program_assembly_v2_${TS}/examples.jsonl}"
STRUCTURED_JSONL="${STRUCTURED_JSONL:-}"

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
TASK_SOURCE="${TASK_SOURCE:-synthetic}"
STRUCTURED_MIX_PROB="${STRUCTURED_MIX_PROB:-0.5}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"

mkdir -p "$REPORT_DIR" "$(dirname "$DATASET_OUT")"

echo "[program curriculum v2] py_compile/help"
python -m py_compile "$MAIN"
python "$MAIN" --help >/tmp/program_assembly_curriculum_v2_help.txt

SAVE_ARGS=()
if [ "$SAVE_CHECKPOINT" != "0" ] && [ "$SAVE_CHECKPOINT" != "false" ] && [ "$SAVE_CHECKPOINT" != "False" ]; then
  SAVE_ARGS+=(--save-checkpoint)
fi
if [ -n "$STRUCTURED_JSONL" ]; then
  SAVE_ARGS+=(--structured-jsonl "$STRUCTURED_JSONL")
fi

cat > "$REPORT_DIR/config.txt" <<EOF_CFG
BRANCH=$BRANCH
MODE=$MODE
REPORT_DIR=$REPORT_DIR
DATASET_OUT=$DATASET_OUT
STRUCTURED_JSONL=$STRUCTURED_JSONL
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
TASK_SOURCE=$TASK_SOURCE
STRUCTURED_MIX_PROB=$STRUCTURED_MIX_PROB
NUM_EXAMPLES=$NUM_EXAMPLES
SAVE_CHECKPOINT=$SAVE_CHECKPOINT
EOF_CFG

set +e
if [ "$MODE" = "generate_jsonl" ]; then
  python "$MAIN" --mode generate_jsonl --dataset-out "$DATASET_OUT" --num-examples "$NUM_EXAMPLES" --dim "$DIM" --steps "$STEPS" --seed "$SEED" 2>&1 | tee "$REPORT_DIR/train.log"
elif [ "$MODE" = "inspect_jsonl" ]; then
  python "$MAIN" --mode inspect_jsonl --structured-jsonl "${STRUCTURED_JSONL:-$DATASET_OUT}" --dim "$DIM" --steps "$STEPS" 2>&1 | tee "$REPORT_DIR/train.log"
else
  RUN_MODE="$MODE"
  if [ "$MODE" = "train_jsonl" ]; then
    RUN_MODE="train_jsonl"
    TASK_SOURCE="structured"
  fi
  python "$MAIN" \
    --mode "$RUN_MODE" \
    --task-source "$TASK_SOURCE" \
    --structured-mix-prob "$STRUCTURED_MIX_PROB" \
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
    --out-dir "$REPORT_DIR" \
    "${SAVE_ARGS[@]}" 2>&1 | tee "$REPORT_DIR/train.log"
fi
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "[program curriculum v2] run_status=$RUN_STATUS" | tee -a "$REPORT_DIR/run_status.txt"

STATUS="simple_butterfly_matrix_v4_tape_lane/AGENT_STATUS.md"
cat > "$STATUS" <<EOF_STATUS
# Agent Status

STATUS: PROGRAM_ASSEMBLY_CURRICULUM_V2_DONE
VERSION: program_assembly_curriculum_v2_structured
RUN_STATUS: $RUN_STATUS
MODE: $MODE
REPORT_DIR: $REPORT_DIR
DATASET_OUT: $DATASET_OUT
STRUCTURED_JSONL: $STRUCTURED_JSONL

RUN_NOW:
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v2_structured.sh

QUICK_SMOKE:
EPOCHS=2 TRAIN_STEPS=50 EVAL_BATCHES=10 BATCH_SIZE=256 bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v2_structured.sh

GENERATE_DATASET:
MODE=generate_jsonl NUM_EXAMPLES=1000 bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v2_structured.sh

TRAIN_STRUCTURED:
MODE=train_jsonl STRUCTURED_JSONL=path/to/head_programs.jsonl bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v2_structured.sh

CHECK:
- REPORT_TO_CHATGPT.txt / final_report.json / metrics.csv for train modes
- train.log / config.txt always

NOTE:
- This is builder pretraining only. Do not load the full checkpoint into the main task model.
- Transfer controller/adapters only after metrics pass.
EOF_STATUS

find "$REPORT_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) -print -exec git add {} +
if [ "$MODE" = "generate_jsonl" ]; then
  git add "$DATASET_OUT" "$(dirname "$DATASET_OUT")/manifest.json" || true
fi
git add "$STATUS" "$MAIN" "simple_butterfly_matrix_v4_tape_lane/commands/sync_run_program_assembly_curriculum_v2_structured.sh"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo "[program curriculum v2] ERROR: checkpoint staged"
  exit 1
fi
if ! git diff --cached --quiet; then
  git commit -m "Add program assembly curriculum v2 structured logs $TS"
  git push origin "$BRANCH" || {
    echo "[program curriculum v2] push failed. Run: git pull --rebase --autostash origin $BRANCH && git push origin $BRANCH" | tee -a "$REPORT_DIR/run_status.txt"
    exit 3
  }
fi

exit "$RUN_STATUS"
