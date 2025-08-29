#!/usr/bin/env python3
"""Step 1 (LoRA): CE finetune that trains only LoRA adapter params to teach canonical tags

This script applies a small LoRA adapter, runs a short cross-entropy pass on
synthetic SFT data, and saves the adapter. Designed for low-memory training.
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

logger = logging.getLogger('step1_ce_lora')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./format_sft_lora')
    p.add_argument('--sft_samples', type=int, default=1000)
    p.add_argument('--sft_file', default='./grpo_portable/sft_from_deepseek.jsonl',
                   help='Path to JSONL SFT file with {"prompt","completion"} per line (optional)')
    p.add_argument('--ce_epochs', type=int, default=3)
    p.add_argument('--ce_batch', type=int, default=16)
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--grad_accum', type=int, default=2, help='Gradient accumulation steps to increase effective batch size')
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    p.add_argument('--save_adapter_name', default='adapter_lora')
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
    # Choose device_map and dtype: prefer fp16 on GPU when available to save memory
    if args.device == 'cpu':
        model_kwargs['device_map'] = 'cpu'
        model_kwargs['torch_dtype'] = torch.float32
        use_fp16 = False
    elif args.device == 'gpu':
        model_kwargs['device_map'] = 'auto'
        # If torch is available and has CUDA, use float16 for smaller memory footprint
        model_kwargs['torch_dtype'] = getattr(torch, 'float16', torch.float32)
        use_fp16 = True
    else:  # auto
        model_kwargs['device_map'] = 'auto'
        if torch is not None and torch.cuda.is_available():
            model_kwargs['torch_dtype'] = getattr(torch, 'float16', torch.float32)
            use_fp16 = True
        else:
            model_kwargs['torch_dtype'] = torch.float32
            use_fp16 = False

    logger.info('Loading base model (this may be memory-heavy)')
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    os.makedirs(args.output, exist_ok=True)
    tokenizer, model = add_special_tokens_and_resize(tokenizer, model, save_dir=args.output)

    # apply LoRA adapter
    lora_conf = LoraConfig(
        r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_alpha,
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
        # PEFT should mark adapter params as requires_grad=True and base as False,
        # but make explicit: freeze base parameters if not in adapter
        if 'lora' not in n and 'adapter' not in n:
            p.requires_grad = False

    # load SFT dataset: prefer JSONL file when present, else build synthetic dataset
    sft = None
    if args.sft_file:
        try:
            if os.path.exists(args.sft_file):
                logger.info('Loading SFT examples from %s', args.sft_file)
                sft = []
                with open(args.sft_file, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            logger.debug('Skipping malformed JSONL line %d in %s', i, args.sft_file)
                            continue
                        # prefer explicit prompt/completion fields
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

    # tokenize small dataset
    def tokenize_fn(x):
        inp = x['prompt']
        tgt = x['completion']
        enc = tokenizer(inp + '\n' + tgt, truncation=True, padding='max_length', max_length=512)
        enc['labels'] = enc['input_ids'].copy()
        return enc

    try:
        # keep simple: transform into list of tokenized dicts
        tok_list = [tokenize_fn(x) for x in sft]
        # if datasets available, convert to HF Dataset to use Trainer properly
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

    logger.info('Using fp16=%s (device=%s, dtype=%s)', use_fp16, args.device, model_kwargs.get('torch_dtype'))

    args_tr = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.ce_epochs,
        per_device_train_batch_size=args.ce_batch,
        gradient_accumulation_steps=max(1, args.grad_accum),
        logging_steps=10,
        save_strategy='no',
        fp16=bool(use_fp16),
    )

    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds)

    logger.info('Starting LoRA CE finetune: %d examples, epochs=%d, batch=%d', len(sft), args.ce_epochs, args.ce_batch)
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
