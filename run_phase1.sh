#!/usr/bin/env bash
set -euo pipefail

# Example run script for Phase1 (50 steps)
MODEL_LOCAL=${MODEL_LOCAL_PATH:-./downloaded_models/llama-3.2-3b}
LOGDIR=./logs
mkdir -p "$LOGDIR"
env FORCE_FP32=1 TORCHDYNAMO_DISABLE=1 MODEL_LOCAL_PATH="$MODEL_LOCAL" "$PWD/venv/bin/python" GRPO-format-phase1.py --use_gsm8k --max_steps 50 --lora_r 8 > "$LOGDIR/format_phase1_50_steps.out" 2>&1 &
echo $!
