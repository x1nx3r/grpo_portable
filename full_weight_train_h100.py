#!/usr/bin/env python3
"""
Full-weight finetune pipeline tuned for H100 (bf16, grad-checkpointing, small micro-batches).

- Uses bitsandbytes Adam8bit optimizer when available to save optimizer memory.
- Tries to enable xformers/flash attention for activation savings if present.
- Supports packed examples to reduce padding waste (pack_examples flag).

This script is intentionally conservative about heavy imports so `-h` works
without requiring GPU-specific packages to be present.
"""
import os
import sys
import argparse
import json
import logging
from typing import Any, Dict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger = logging.getLogger("full_weight_train_h100")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--base', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./full_weight_out')
    p.add_argument('--sft_file', default='./sft_from_deepseek_full.canonical.jsonl')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--per_device_batch', type=int, default=1, help='micro-batch per GPU (keep small on H100)')
    p.add_argument('--grad_accum', type=int, default=32, help='gradient accumulation to reach effective batch')
    p.add_argument('--max_length', type=int, default=10000)
    p.add_argument('--pack_examples', action='store_true')
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--device', choices=['auto','gpu','cpu'], default='auto')
    return p.parse_args(argv)


def build_sft_list(path):
    if os.path.exists(path):
        out=[]
        with open(path,'r',encoding='utf-8') as fh:
            for ln in fh:
                ln=ln.strip()
                if not ln: continue
                try:
                    r=json.loads(ln)
                except Exception:
                    continue
                if 'prompt' in r and 'completion' in r:
                    out.append({'prompt': r['prompt'], 'completion': r['completion']})
        return out
    # fallback tiny synthetic
    return [{'prompt': f'User: What is {i}?', 'completion': str(i)} for i in range(1,1001)]


def main(argv):
    args = parse_args(argv)
    # heavy imports deferred so -h works quickly
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    except Exception:
        logger.exception("transformers/torch not available; install transformers and torch to run training")
        return 1

    # load tokenizer & model (prefer bf16 on H100)
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    use_bf16 = (args.device != 'cpu') and hasattr(torch, 'bfloat16')
    model_kwargs: Dict[str, Any] = {'trust_remote_code': True}
    if args.device == 'cpu':
        model_kwargs['device_map']='cpu'
        model_kwargs['torch_dtype']=torch.float32
    else:
        model_kwargs['device_map']='auto'
        model_kwargs['torch_dtype'] = torch.bfloat16 if use_bf16 else torch.float32

    logger.info("Loading base model (device_map=%s dtype=%s)", model_kwargs.get('device_map'), model_kwargs.get('torch_dtype'))
    model = AutoModelForCausalLM.from_pretrained(args.base, **model_kwargs)

    # enable memory-efficient attention if available
    try:
        if hasattr(model, 'enable_xformers_memory_efficient_attention'):
            model.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
    except Exception:
        logger.debug("xformers/flash_attn not available; continuing without it")

    # enable gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'config'):
            try:
                model.config.use_cache = False
            except Exception:
                pass
        logger.info("Enabled gradient checkpointing and disabled use_cache")
    except Exception:
        logger.debug("Failed to enable gradient checkpointing (continuing)")

    # Token handling & special tokens
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    os.makedirs(args.output, exist_ok=True)
    tokenizer.save_pretrained(args.output)

    # prepare dataset list of tokenized examples
    sft = build_sft_list(args.sft_file)
    def tokenize_fn(x):
        enc_p = tokenizer(x['prompt'], add_special_tokens=False, truncation=True, max_length=args.max_length)
        enc_c = tokenizer(x['completion'], add_special_tokens=False, truncation=True, max_length=args.max_length)
        input_ids = enc_p['input_ids'] + enc_c['input_ids']
        eos = tokenizer.eos_token_id
        if eos is not None and (len(input_ids)==0 or input_ids[-1]!=eos):
            input_ids.append(eos)
        prompt_len = len(enc_p['input_ids'])
        labels = input_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {'input_ids': input_ids, 'attention_mask':[1]*len(input_ids), 'labels': labels, 'text': (x['prompt'] + tokenizer.eos_token + x['completion'])}
    tok_list = [tokenize_fn(x) for x in sft]

    # simple dataset wrapper if datasets not available
    try:
        from datasets import Dataset as HFDataset
        train_ds = HFDataset.from_list(tok_list)
    except Exception:
        class SimpleList:
            def __init__(self,a): self.a=a
            def __len__(self): return len(self.a)
            def __getitem__(self,i): return self.a[i]
        train_ds = SimpleList(tok_list)

    # collator: packed collator to reduce padding
    class DataCollatorPacked:
        def __init__(self, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = int(max_length)
            self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            self.eos = tokenizer.eos_token_id
        def __call__(self, features):
            packed=[]
            cur=None
            for ex in features:
                ids=ex['input_ids'][:]
                labs=ex['labels'][:]
                if self.eos is not None and ids and ids[-1]!=self.eos:
                    ids.append(self.eos); labs.append(self.eos)
                if cur is None:
                    cur={'input_ids':ids,'labels':labs}
                    continue
                if len(cur['input_ids'])+len(ids) <= self.max_length:
                    if cur['input_ids'] and cur['input_ids'][-1] != self.eos:
                        cur['input_ids'].append(self.eos); cur['labels'].append(self.eos)
                    cur['input_ids'].extend(ids); cur['labels'].extend(labs)
                else:
                    packed.append(cur); cur={'input_ids':ids,'labels':labs}
            if cur is not None: packed.append(cur)
            texts=[self.tokenizer.decode(p['input_ids'], skip_special_tokens=False) for p in packed]
            batch = self.tokenizer(texts, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=False)
            import torch
            max_len = batch['input_ids'].shape[1]
            pad_id = self.pad_token_id
            labs=[]
            for p in packed:
                l=p['labels']
                if len(l) > max_len: l=l[:max_len]
                else: l = l + [pad_id]*(max_len - len(l))
                labs.append(l)
            batch['labels'] = torch.tensor(labs, dtype=torch.long)
            batch['labels'][batch['labels']==pad_id] = -100
            return batch

    collate = DataCollatorPacked(tokenizer, args.max_length) if args.pack_examples else None

    # build TrainingArguments
    tr_args = dict(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,             # H100: prefer bf16
        fp16=False,
        logging_steps=50,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
    )
    TrainingArgumentsCls = TrainingArguments
    tr = TrainingArgumentsCls(**{k:v for k,v in tr_args.items()})

    # build optimizer: attempt bitsandbytes 8-bit AdamW to save optimizer memory
    optimizer = None
    try:
        import bitsandbytes as bnb
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logger.info("Using bitsandbytes AdamW8bit optimizer")
    except Exception:
        import torch.optim as optim
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        logger.info("bitsandbytes AdamW8bit not available, falling back to torch AdamW")

    trainer = Trainer(model=model, args=tr, train_dataset=train_ds, data_collator=collate, optimizers=(optimizer, None))

    logger.info("Starting training: per_device=%d grad_accum=%d effective_batch=%d bf16=%s", args.per_device_batch, args.grad_accum, args.per_device_batch*args.grad_accum, str(use_bf16))
    trainer.train()

    # save full model and tokenizer
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    logger.info("Saved model to %s", args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
