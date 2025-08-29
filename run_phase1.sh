#!/usr/bin/env bash
set -euo pipefail

# If pyenv is installed in the user's home, load it so the venv/python selection works.
# This does not modify any home files; it only attempts to source pyenv at runtime when available.
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
if [[ -d "$PYENV_ROOT" && -x "$PYENV_ROOT/bin/pyenv" ]]; then
	export PATH="$PYENV_ROOT/bin:$PATH"
	# initialize pyenv (bash) and pyenv-virtualenv if available
	eval "$(pyenv init - bash)" || true
	if command -v pyenv-virtualenv-init >/dev/null 2>&1; then
		eval "$(pyenv virtualenv-init -)" || true
	fi
fi

# Example run script for Phase1 (50 steps)
MODEL_LOCAL=${MODEL_LOCAL_PATH:-./downloaded_models/llama-3.2-3b}
LOGDIR=./logs
mkdir -p "$LOGDIR"
env FORCE_FP32=1 TORCHDYNAMO_DISABLE=1 MODEL_LOCAL_PATH="$MODEL_LOCAL" "$PWD/venv/bin/python" GRPO-format-phase1.py --use_gsm8k --max_steps 50 --lora_r 8 > "$LOGDIR/format_phase1_50_steps.out" 2>&1 &
echo $!
