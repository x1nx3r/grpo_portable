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
import random

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
    # Use a generous max_length so we don't truncate completions (many completions end with </answer>). 
    # Packing via --pack_examples will mitigate padding overhead by concatenating short examples.
    p.add_argument('--max_length', type=int, default=10000, help='Maximum token length for prompt+completion (large by default; use packing to avoid wasted padding)')
    p.add_argument('--pack_examples', action='store_true', help='Enable packing multiple short examples into longer sequences to reduce padding waste')
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
        # Tokenize prompt and target separately so we can mask prompt tokens in labels
        prompt = x['prompt']
        tgt = x['completion']

        # encode without padding so we can control truncation strategy
        enc_prompt = tokenizer(prompt, truncation=True, padding=False, max_length=args.max_length)
        enc_tgt = tokenizer(tgt, truncation=True, padding=False, max_length=args.max_length)

        p_ids = enc_prompt.get('input_ids', [])
        t_ids = enc_tgt.get('input_ids', [])

        # concatenate prompt + target
        input_ids = p_ids + t_ids

        # ensure eos token at end if tokenizer provides one
        eos = getattr(tokenizer, 'eos_token_id', None)
        if eos is not None and (len(input_ids) == 0 or input_ids[-1] != eos):
            input_ids = input_ids + [eos]

        max_length = int(args.max_length)
        # Truncate if necessary. Prefer keeping the completion tokens intact.
        if len(input_ids) > max_length:
            tgt_len = len(t_ids) + (1 if eos is not None and (len(t_ids) == 0 or t_ids[-1] != eos) else 0)
            # if target itself is larger than max_length, keep last max_length tokens of target
            if tgt_len >= max_length:
                input_ids = (t_ids[-max_length:])
                prompt_len = 0
            else:
                keep_prompt = max_length - tgt_len
                # Prefer to preserve prompt head (instruction) not tail
                input_ids = (p_ids[:keep_prompt] + t_ids)   # <-- keep head
                prompt_len = len(p_ids[:keep_prompt])
        else:
            prompt_len = len(p_ids)

        # do NOT pad to max_length here; let the DataCollator handle dynamic padding
        attention_mask = [1] * len(input_ids)

        labels = input_ids.copy()
        # mask out prompt tokens so loss is computed only on completion tokens
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

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
        bf16=bool(use_bf16),
        learning_rate=max(1e-4, args.lr),  # default to 1e-4 if user left 2e-4
        weight_decay=args.weight_decay,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        warmup_ratio=0.03,
    )

    # gradient checkpointing handled earlier (before PEFT wrapping)

    # Use a data collator that dynamically pads batches instead of eager padding to max_length
    # Build a data collator. Prefer dynamic padding; optionally enable packing to reduce padding waste.
    collate_fn = None
    try:
        from transformers import DataCollatorWithPadding

        class DataCollatorPacked:
            """Pack multiple short examples into longer sequences up to max_length.

            This collator concatenates consecutive examples within a batch until the
            packed sequence would exceed args.max_length, producing fewer, longer
            training examples that reduce padding overhead. The collator returns
            PyTorch tensors suitable for Trainer.
            """
            def __init__(self, tokenizer, max_length, max_examples_per_pack: int = 32, shuffle: bool = True):
                self.tokenizer = tokenizer
                self.max_length = int(max_length)
                self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                # EOS token id for safe concatenation
                self.eos_token_id = tokenizer.eos_token_id
                # limit how many examples we will pack together to avoid extremely long packed sequences
                self.max_examples_per_pack = int(max_examples_per_pack)
                # whether to shuffle the input features before packing (helps diversification)
                self.shuffle = bool(shuffle)

            def __call__(self, features):
                # features: list of dicts with 'input_ids', 'attention_mask', 'labels'
                feats = list(features)
                if self.shuffle:
                    random.shuffle(feats)

                packed = []
                cur = None
                cur_count = 0
                for ex in feats:
                    ids = list(ex['input_ids'])
                    mask = list(ex.get('attention_mask', [1] * len(ids)))
                    labs = list(ex.get('labels', ids.copy()))

                    # normalize: ensure individual example ends with EOS so concatenation is well-formed
                    if self.eos_token_id is not None and len(ids) > 0 and ids[-1] != self.eos_token_id:
                        ids = ids + [self.eos_token_id]
                        mask = mask + [1]
                        labs = labs + [self.eos_token_id]

                    # start new pack if needed
                    if cur is None:
                        cur = {'input_ids': ids.copy(), 'attention_mask': mask.copy(), 'labels': labs.copy()}
                        cur_count = 1
                        continue

                    # if adding this example would overflow or exceed max_examples_per_pack, flush
                    will_len = len(cur['input_ids']) + len(ids)
                    if will_len <= self.max_length and cur_count < self.max_examples_per_pack:
                        # ensure single EOS separator between concatenated examples
                        if self.eos_token_id is not None and len(cur['input_ids']) > 0 and cur['input_ids'][-1] != self.eos_token_id:
                            cur['input_ids'].append(self.eos_token_id)
                            cur['attention_mask'].append(1)
                            cur['labels'].append(self.eos_token_id)
                        cur['input_ids'].extend(ids)
                        cur['attention_mask'].extend(mask)
                        cur['labels'].extend(labs)
                        cur_count += 1
                    else:
                        packed.append(cur)
                        cur = {'input_ids': ids.copy(), 'attention_mask': mask.copy(), 'labels': labs.copy()}
                        cur_count = 1

                if cur is not None:
                    packed.append(cur)

                # Use tokenizer.pad to convert to tensors with dynamic padding
                batch = self.tokenizer.pad(packed, padding='longest', return_tensors='pt')
                # convert pad token labels (pad_token_id) to -100
                try:
                    import torch
                    if 'labels' in batch:
                        pad_id = self.pad_token_id
                        batch['labels'][batch['labels'] == pad_id] = -100
                except Exception:
                    pass
                return batch

        if getattr(args, 'pack_examples', False):
            collate_fn = DataCollatorPacked(tokenizer, args.max_length)
        else:
            data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

            def collate_fn(batch):
                batch = data_collator(batch)
                # convert label padding (pad_token_id) to -100 so loss ignores padded positions
                if 'labels' in batch:
                    try:
                        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                        batch['labels'][batch['labels'] == pad_id] = -100
                    except Exception:
                        pass
                return batch
    except Exception:
        collate_fn = None

    trainer = Trainer(model=model, args=args_tr, train_dataset=train_ds, data_collator=collate_fn)

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
