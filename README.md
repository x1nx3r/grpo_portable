GRPO Portable

This folder contains a portable subset of the GRPO experiments (Phase 1 format-first training and helpers).

Included scripts:
- GRPO-format-phase1.py: Phase 1 script to add tokenizer special tokens, build synthetic SFT data (optional GSM8K), optional CE finetune, and GRPO format-only training to create Adapter A (LoRA).
- GRPO-gsm8k-nounsloth.py: An Unsloth-free GRPO pipeline (Hugging Face + PEFT + TRL) used during development.
- unsloth_download_and_save.py: Helper to download gated HF models via Unsloth and save locally for offline use.
- r1_data_processor.py: Dataset skimmer and data-quality report generator (dq_report.json).

Quickstart (assumes a Python 3.11 venv with GPU tooling available):

1. Create venv and install requirements

   python -m venv venv
   . venv/bin/activate
   pip install -r requirements.txt

2. (Optional) Download gated model using Unsloth (if you have access):

   python unsloth_download_and_save.py --model meta-llama/Llama-3.2-3B-Instruct --out ./downloaded_models/llama-3.2-3b

3. Run Phase1 (50 steps example):

   env FORCE_FP32=1 TORCHDYNAMO_DISABLE=1 MODEL_LOCAL_PATH=./downloaded_models/llama-3.2-3b venv/bin/python GRPO-format-phase1.py --use_gsm8k --max_steps 50 --lora_r 8 > ./logs/format_phase1_50_steps.out 2>&1 &

Notes:
- The repository is a minimal portable snapshot for replication. Adjust hyperparameters and flags for larger machines.
- For debugging, set DEBUG_ANOMALY=1 to enable torch anomaly detection (slower).
