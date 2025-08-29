#!/usr/bin/env python3
"""Simple local inference runner for a saved LoRA adapter.

Loads base model + PEFT adapter, runs a few sample prompts (or user-provided prompt)
and prints model generations. Uses `phase1_utils` for small synthetic prompts.
"""
import argparse
import logging
import os
import sys
from typing import List

from phase1_utils import build_synthetic_dataset, make_synthetic_format_sample, add_special_tokens_and_resize

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

logger = logging.getLogger('run_adapter_inference_local')
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def parse_args(argv: List[str]):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--adapter', default='./format_sft_lora_h100/adapter_lora_h100')
    p.add_argument('--tokenizer_dir', default='./format_sft_lora_h100')
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    p.add_argument('--prompt', default=None, help='Single prompt to run (overrides synthetic samples)')
    p.add_argument('--n_samples', type=int, default=3, help='Number of synthetic prompts to run when --prompt is not given')
    p.add_argument('--max_new_tokens', type=int, default=128)
    p.add_argument('--no_prefix_think', action='store_true', help='Disable automatic prefixing of prompts with "<think>"')
    return p.parse_args(argv)


def load_tokenizer(args):
    tk_dir = args.tokenizer_dir if os.path.exists(args.tokenizer_dir) else args.model
    logger.info('Loading tokenizer from %s', tk_dir)
    return AutoTokenizer.from_pretrained(tk_dir, use_fast=True)


def load_model_with_adapter(args, tokenizer=None, dtype=None):
    model_kwargs = dict(trust_remote_code=True)
    if args.device == 'cpu':
        model_kwargs['device_map'] = 'cpu'
        if dtype is not None:
            model_kwargs['torch_dtype'] = torch.float32
    else:
        model_kwargs['device_map'] = 'auto'
        if dtype is not None:
            model_kwargs['torch_dtype'] = dtype

    logger.info('Loading base model from %s', args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    # if a tokenizer is provided, ensure base embeddings match tokenizer size
    try:
        if tokenizer is not None:
            tok_len = len(tokenizer)
            emb = base.get_input_embeddings()
            if emb is not None and emb.weight.size(0) != tok_len:
                logger.info('Resizing base model embeddings from %d to %d to match tokenizer', emb.weight.size(0), tok_len)
                base.resize_token_embeddings(tok_len)
    except Exception:
        logger.exception('Failed checking/resizing base embeddings (continuing)')

    if PeftModel is None:
        logger.error('peft.PeftModel not available; install peft to load adapter')
        raise SystemExit(1)

    logger.info('Loading PEFT adapter from %s', args.adapter)
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    return base, model


def build_prompts(args):
    if args.prompt:
        prompts = [args.prompt]
    else:
        # synthetic format prompts from phase1_utils
        samples = build_synthetic_dataset(n_samples=args.n_samples)
        prompts = [s['prompt'] for s in samples]

    # prefix with opening <think> tag unless explicitly disabled or already present
    if not getattr(args, 'no_prefix_think', False):
        prefixed = []
        for p in prompts:
            # avoid double-prefixing; check case-insensitively
            if '<think>' in p.lower():
                prefixed.append(p)
            else:
                prefixed.append(f"<think>\n{p}")
        prompts = prefixed

    return prompts


def generate_and_print(model, tokenizer, prompts, max_new_tokens=128):
    device = next(model.parameters()).device
    for i, prompt in enumerate(prompts, 1):
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        print('--- PROMPT %d ---' % i)
        print(prompt)
        print('--- OUTPUT %d ---' % i)
        print(txt)
        print()


def main(argv):
    args = parse_args(argv)

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        logger.error('transformers not available in this environment; aborting')
        return 1

    tokenizer = load_tokenizer(args)

    # prefer bf16 if available
    dtype = getattr(torch, 'bfloat16', None)
    base, model = load_model_with_adapter(args, tokenizer=tokenizer, dtype=dtype)

    # ensure tokenizer special tokens are present and resize base embeddings if needed
    try:
        tokenizer, base = add_special_tokens_and_resize(tokenizer, base, save_dir=args.tokenizer_dir)
    except Exception:
        logger.exception('Failed to add special tokens and resize (continuing)')

    prompts = build_prompts(args)
    generate_and_print(model, tokenizer, prompts, max_new_tokens=args.max_new_tokens)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
