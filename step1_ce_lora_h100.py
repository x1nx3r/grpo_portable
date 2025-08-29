#!/usr/bin/env python3
"""H100-optimized Step 1 (LoRA): CE finetune tuned for powerful GPUs (bf16)

This script mirrors `step1_ce_lora.py` but changes sensible defaults for an
H100-class GPU: bf16, larger LoRA rank/alpha, bigger batch sizes, optional
gradient checkpointing and a few additional hyperparameters exposed.
"""
import argparse
import logging
import os
import sys
import json

from phase1_utils import add_special_tokens_and_resize, build_synthetic_dataset

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    Trainer = None
    TrainingArguments = None

try:
    from peft import LoraConfig, get_peft_model
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

logger = logging.getLogger('step1_ce_lora_h100')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./format_sft_lora_h100')
    p.add_argument('--sft_samples', type=int, default=8000)
    # default to repo-root JSONL (previous default pointed to ./grpo_portable/...,
    # which caused a duplicate path when running from the repo root)
    p.add_argument('--sft_file', default='./sft_from_deepseek_full.canonical.jsonl',
                   help='Path to JSONL SFT file with {"prompt","completion"} per line (optional)')
    p.add_argument('--ce_epochs', type=int, default=3)
    # larger per-device batch assumed on H100
    p.add_argument('--ce_batch', type=int, default=32)
    # larger LoRA rank for richer adapters
    p.add_argument('--lora_r', type=int, default=64)
    p.add_argument('--lora_alpha', type=int, default=128)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    p.add_argument('--save_adapter_name', default='adapter_lora_h100')
    # enable gradient checkpointing by default on H100; keep flag to allow future explicit control
    p.add_argument('--use_grad_checkpoint', action='store_true', default=True,
                   help='Enable gradient checkpointing if supported (default: True)')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        logger.error('transformers not available in this environment; aborting')
        return 1

    if get_peft_model is None or LoraConfig is None:
        logger.error('peft not available; install `peft` to run LoRA CE')
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model_kwargs = dict(trust_remote_code=True)
    # On H100 prefer bf16
    if args.device == 'cpu':
        model_kwargs['device_map'] = 'cpu'
        model_kwargs['torch_dtype'] = torch.float32
        use_bf16 = False
    else:
        model_kwargs['device_map'] = 'auto'
        # use bf16 where available
        model_kwargs['torch_dtype'] = getattr(torch, 'bfloat16', torch.float32)
        use_bf16 = True

    logger.info('Loading base model (H100 tune: bf16 recommended)')
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    os.makedirs(args.output, exist_ok=True)
    tokenizer, model = add_special_tokens_and_resize(tokenizer, model, save_dir=args.output)

    # optionally enable gradient checkpointing early (before PEFT wrapping) to save activation memory
    if args.use_grad_checkpoint:
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                # disable use_cache to be compatible with gradient checkpointing
                if hasattr(model, 'config'):
                    try:
                        model.config.use_cache = False
                    except Exception:
                        logger.debug('Failed to set model.config.use_cache=False')
                logger.info('Enabled model.gradient_checkpointing and set use_cache=False')
        except Exception:
            logger.exception('Failed to enable gradient checkpointing (continuing)')

    # apply LoRA adapter
    lora_conf = LoraConfig(
        r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM'
    )
    try:
        model = get_peft_model(model, lora_conf)
    except Exception:
        logger.exception('Failed to apply LoRA')
        return 1

    # ensure only LoRA params will be trained
    for n, p in model.named_parameters():
        if 'lora' not in n and 'adapter' not in n:
            p.requires_grad = False

    # load SFT dataset: prefer JSONL file when present, else build synthetic dataset
    sft = None
    if args.sft_file:
        try:
            # prefer the path provided, but if it doesn't exist try a repo-root basename
            sft_path = args.sft_file
            if not os.path.exists(sft_path):
                alt = os.path.join('.', os.path.basename(sft_path))
                if os.path.exists(alt):
                    logger.info('SFT file %s not found, using fallback %s', sft_path, alt)
                    sft_path = alt

            if os.path.exists(sft_path):
                logger.info('Loading SFT examples from %s', sft_path)
                sft = []
                with open(sft_path, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            logger.debug('Skipping malformed JSONL line %d in %s', i, args.sft_file)
                            continue
                        if isinstance(rec, dict) and 'prompt' in rec and 'completion' in rec:
                            sft.append({'prompt': rec['prompt'], 'completion': rec['completion']})
                logger.info('Loaded %d SFT examples from %s', len(sft), args.sft_file)
                if len(sft) == 0:
                    logger.warning('No examples found in %s; falling back to synthetic SFT', args.sft_file)
                    sft = None
            else:
                logger.info('SFT file %s not found; using synthetic SFT', args.sft_file)
        except Exception:
            logger.exception('Failed to read sft_file %s; falling back to synthetic', args.sft_file)

    if sft is None:
        logger.info('Building synthetic SFT dataset: %d samples', args.sft_samples)
        sft = build_synthetic_dataset(n_samples=args.sft_samples)

    # simple tokenization helper
    def tokenize_fn(x):
        inp = x['prompt']
        tgt = x['completion']
        enc = tokenizer(inp + '\n' + tgt, truncation=True, padding='max_length', max_length=1024)
        enc['labels'] = enc['input_ids'].copy()
        return enc

    try:
        tok_list = [tokenize_fn(x) for x in sft]
        try:
            from datasets import Dataset as HFDataset
            train_ds = HFDataset.from_list(tok_list)
        except Exception:
            train_ds = tok_list
    except Exception:
        logger.exception('Failed to tokenize SFT dataset')
        return 1

    if TrainingArguments is None or Trainer is None:
        logger.error('transformers.Trainer not available; cannot run CE')
        return 1

    logger.info('Using bf16=%s (device=%s, dtype=%s)', use_bf16, args.device, model_kwargs.get('torch_dtype'))

    args_tr = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.ce_epochs,
        per_device_train_batch_size=args.ce_batch,
        gradient_accumulation_steps=max(1, args.grad_accum),
        logging_steps=50,
        save_strategy='no',
        bf16=bool(use_bf16),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    # gradient checkpointing handled earlier (before PEFT wrapping)

    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds)

    logger.info('Starting H100-tuned LoRA CE finetune: %d examples, epochs=%d, batch=%d', len(sft), args.ce_epochs, args.ce_batch)
    trainer.train()

    # save adapter only
    adapter_out = os.path.join(args.output, args.save_adapter_name)
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(adapter_out)
            logger.info('Saved LoRA adapter to %s', adapter_out)
    except Exception:
        logger.exception('Failed to save LoRA adapter')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
