#!/usr/bin/env python3
"""Phase 1: Format-first GRPO training (Adapter A)

- Adds tokenizer special tokens for R1 XML tags.
- Generates synthetic SFT samples that exercise formatting tags (<think>...</think>, <answer>...</answer>).
- Optionally runs a short CE finetune (1-3 epochs) on the synthetic data.
- Runs GRPO to train Adapter A (LoRA) with a format-only reward until format pass rate target.

Usage: set up your venv and run this file. Safe imports used so it can be
inspected without all dependencies.
"""
import argparse
import logging
import os
import sys
from datetime import datetime

import random
import re
import json

try:
    import torch
except Exception:
    torch = None

try:
    from datasets import Dataset, load_dataset
except Exception:
    Dataset = list
    load_dataset = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None
    Trainer = None
    TrainingArguments = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

import numpy as np

logger = logging.getLogger('GRPO_Format_Phase1')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# R1 tags we want to enforce
SPECIAL_TOKENS = ['<think>', '</think>', '<answer>', '</answer>']

# simple extraction used by rewards
XML_THINK_PAT = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
XML_ANSWER_PAT = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


def add_special_tokens_and_resize(tokenizer, model, save_dir=None):
    if tokenizer is None or model is None:
        logger.warning('Tokenizer or model not available — skipping special tokens step')
        return tokenizer, model
    added = tokenizer.add_tokens(SPECIAL_TOKENS)
    logger.info('Added %d special tokens', added)
    if added > 0:
        try:
            model.resize_token_embeddings(len(tokenizer))
            logger.info('Resized model embeddings to %d', len(tokenizer))
        except Exception:
            logger.exception('Failed to resize embeddings')
    if save_dir:
        try:
            tokenizer.save_pretrained(save_dir)
        except Exception:
            logger.exception('Failed to save tokenizer to %s', save_dir)
    return tokenizer, model


def make_synthetic_format_sample(i):
    # create a short math or reasoning prompt with XML-COT format
    q = f"What is {i} plus {i}?"
    reasoning = f"{i} + {i} = {i*2}"
    answer = str(i*2)
    prompt = (
        "System: Please answer in R1 XML-COT format.\n"
        f"User: {q}\n"
    )
    completion = f"<think>{reasoning}</think><answer>{answer}</answer>"
    return {'prompt': prompt, 'completion': completion}


def build_synthetic_dataset(n_samples=2000, seed=42):
    random.seed(seed)
    samples = [make_synthetic_format_sample(i) for i in range(1, n_samples+1)]
    try:
        return Dataset.from_list(samples)
    except Exception:
        return samples


def extract_hash_answer(text: str) -> str | None:
    # GSM8K answers often contain a '####' separator before the final answer
    try:
        if text is None:
            return None
        if "####" in text:
            return text.split('####')[-1].strip()
        # fallback: take last token sequence that looks like an integer
        m = re.search(r"(\d+)$", text.strip())
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def build_synthetic_from_gsm8k(n_samples=2000, split='train'):
    """Build synthetic R1-format SFT samples using GSM8K prompts and ground-truth answers.

    If `datasets` is not available or GSM8K cannot be loaded, falls back to
    the numeric synthetic generator.
    """
    if load_dataset is None:
        logger.warning('datasets.load_dataset not available; falling back to numeric synthetic samples')
        return build_synthetic_dataset(n_samples)

    try:
        ds = load_dataset('openai/gsm8k', 'main')[split]
    except Exception:
        logger.exception('Failed to load GSM8K; falling back to numeric synthetic samples')
        return build_synthetic_dataset(n_samples)

    samples = []
    for i, ex in enumerate(ds):
        if len(samples) >= n_samples:
            break
        try:
            q = ex.get('question') or ex.get('input') or ''
            ans_raw = ex.get('answer') or ''
            ans = extract_hash_answer(ans_raw)
            # if no explicit answer, skip
            if not ans:
                continue
            prompt = "System: Please answer in R1 XML-COT format.\nUser: " + q + "\n"
            # simple synthetic reasoning: echo a trivial reasoning line when possible
            reasoning = f"Answer is {ans}"
            completion = f"<think>{reasoning}</think><answer>{ans}</answer>"
            samples.append({'prompt': prompt, 'completion': completion})
        except Exception:
            continue

    if not samples:
        return build_synthetic_dataset(n_samples)

    try:
        return Dataset.from_list(samples)
    except Exception:
        return samples


# strict format reward used for GRPO (kept for backward compatibility)
def format_reward_fn(completions, **kwargs):
    # legacy strict reward: 1.0 only when both canonical <think> and <answer> exist
    contents = []
    for c in completions:
        try:
            if isinstance(c, str):
                contents.append(c)
            elif isinstance(c, (list, tuple)) and len(c) > 0 and isinstance(c[0], dict):
                contents.append(c[0].get('content', ''))
            elif isinstance(c, dict) and 'content' in c:
                contents.append(c.get('content', ''))
            else:
                contents.append(str(c))
        except Exception:
            contents.append('')

    def score(text: str) -> float:
        try:
            has_think = bool(XML_THINK_PAT.search(text))
            has_answer = bool(XML_ANSWER_PAT.search(text))
            return 1.0 if (has_think and has_answer) else 0.0
        except Exception:
            return 0.0

    out = [score(t) for t in contents]
    return out


# Build an enhanced/soft format reward allowing tag variants and partial credit
ALT_THINK_RE = re.compile(r"<(think|reasoning|r1response)[^>]*>.*?</(think|reasoning|r1response)>", flags=re.IGNORECASE | re.DOTALL)
ALT_ANSWER_RE = re.compile(r"<(answer|ans|solution|final_answer|solution_text)[^>]*>.*?</(answer|ans|solution|final_answer|solution_text)>", flags=re.IGNORECASE | re.DOTALL)

def build_format_reward(mode: str = 'strict'):
    """Return a reward function suitable for TRL/GRPO.

    mode: 'strict' -> only canonical tags give full credit;
          'soft'   -> partial credit for partial/variant formatting.
    """
    def reward_fn(completions, **kwargs):
        contents = []
        for c in completions:
            try:
                if isinstance(c, str):
                    contents.append(c)
                elif isinstance(c, (list, tuple)) and len(c) > 0 and isinstance(c[0], dict):
                    contents.append(c[0].get('content', ''))
                elif isinstance(c, dict) and 'content' in c:
                    contents.append(c.get('content', ''))
                else:
                    contents.append(str(c))
            except Exception:
                contents.append('')

        def score_text(text: str) -> float:
            try:
                # canonical tags
                has_think = bool(XML_THINK_PAT.search(text))
                has_answer = bool(XML_ANSWER_PAT.search(text))
                if mode == 'strict':
                    return 1.0 if (has_think and has_answer) else 0.0

                # soft mode: consider variants and partial credit
                alt_think = bool(ALT_THINK_RE.search(text))
                alt_answer = bool(ALT_ANSWER_RE.search(text))

                # full credit if canonical both
                if has_think and has_answer:
                    return 1.0
                # high credit if one canonical and one variant
                if (has_think and alt_answer) or (alt_think and has_answer):
                    return 0.9
                # medium credit if both variants
                if alt_think and alt_answer:
                    return 0.8
                # partial credit for only one side present
                if has_think or alt_think:
                    return 0.5
                if has_answer or alt_answer:
                    return 0.5
                # small credit if we detect XML-like structure (angle brackets)
                if '<' in text and '>' in text:
                    return 0.2
                return 0.0
            except Exception:
                return 0.0

        return [score_text(t) for t in contents]

    return reward_fn


def do_ce_finetune(tokenizer, model, sft_dataset, output_dir, epochs=1, per_device_batch=4):
    if Trainer is None or TrainingArguments is None:
        logger.warning('transformers.Trainer not available; skipping CE finetune')
        return False
    # prepare tokenized dataset
    def tokenize_fn(x):
        inp = x['prompt']
        tgt = x['completion']
        # concatenate and tokenize
        enc = tokenizer(inp + '\n' + tgt, truncation=True, padding='max_length', max_length=512)
        enc['labels'] = enc['input_ids'].copy()
        return enc

    try:
        ds = sft_dataset
        if not hasattr(ds, 'map'):
            ds = Dataset.from_list(ds)
        tokds = ds.map(lambda x: tokenize_fn(x), batched=False)
    except Exception:
        logger.exception('Failed to tokenize SFT dataset; skipping CE finetune')
        return False

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch,
        logging_steps=10,
        save_strategy='no',
        fp16=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokds)
    trainer.train()
    return True


def train_adapter_a_with_grpo(model, tokenizer, train_dataset, output_dir, lora_r=16, max_steps=200, target_format=0.99, reward_mode: str = 'auto'):
    # apply LoRA
    if get_peft_model is None or LoraConfig is None:
        logger.error('PEFT not available; cannot create LoRA adapter')
        return False
    try:
        lora_conf = LoraConfig(r=lora_r, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_alpha=16, bias='none')
        model = get_peft_model(model, lora_conf)
    except Exception:
        logger.exception('Failed to apply LoRA')
        return False

    # register grad sanitizers
    def _register_grad_sanitizer(model, max_abs=1e4):
        handles = []
        if model is None or torch is None:
            return handles
        def make_hook(n):
            def hook(g):
                try:
                    if g is None:
                        return g
                    if not torch.isfinite(g).all():
                        g = torch.nan_to_num(g, nan=0.0, posinf=max_abs, neginf=-max_abs)
                    return torch.clamp(g, -max_abs, max_abs)
                except Exception:
                    return g
            return hook
        for n,p in model.named_parameters():
            if p.requires_grad:
                try:
                    handles.append(p.register_hook(make_hook(n)))
                except Exception:
                    pass
        logger.info('Installed %d grad sanitizer hooks', len(handles))
        return handles

    grad_hooks = _register_grad_sanitizer(model)

    # determine and configure GRPO reward function
    if GRPOConfig is None or GRPOTrainer is None:
        logger.error('TRL GRPO not available; cannot run GRPO')
        return False
    # preflight reward sanity check when in 'auto' mode: run strict reward on a
    # few synthetic completions from train_dataset; if all zeros, switch to soft.
    chosen_mode = reward_mode
    try:
        if reward_mode == 'auto':
            # extract up to 16 completion strings from the provided dataset
            sample_comps = []
            try:
                # train_dataset may be a list of dicts or a datasets.Dataset
                iterator = train_dataset if isinstance(train_dataset, list) else list(train_dataset)
                for ex in iterator[:16]:
                    if isinstance(ex, dict):
                        sample_comps.append(ex.get('completion', ''))
                    else:
                        # fallback: string
                        sample_comps.append(str(ex))
            except Exception:
                sample_comps = []

            if sample_comps:
                strict_rewards = format_reward_fn(sample_comps)
                mean_strict = float(np.mean(strict_rewards)) if len(strict_rewards) > 0 else 0.0
                logger.info('Preflight strict-format mean reward on synthetic samples: %f', mean_strict)
                if mean_strict <= 0.0:
                    logger.info('Strict reward returned zero on synthetic completions; switching to soft reward mode')
                    chosen_mode = 'soft'
                else:
                    chosen_mode = 'strict'
            else:
                logger.info('No sample completions available for preflight; defaulting to soft reward')
                chosen_mode = 'soft'
    except Exception:
        logger.exception('Preflight reward check failed; defaulting to soft')
        chosen_mode = 'soft'

    reward_fn = build_format_reward(mode=chosen_mode)

    args = GRPOConfig(
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.0,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        optim='paged_adamw_8bit',
        logging_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        # TRL/GRPO requires at least 2 generations per prompt to compute advantages
        # TRL/GRPO requires at least 2 generations per prompt to compute advantages
        num_generations=2,
        # batch size for generation must be divisible by num_generations
        generation_batch_size=2,
        max_prompt_length=256,
        max_completion_length=128,
        max_steps=max_steps,
        save_steps=max_steps,
        max_grad_norm=0.5,
        report_to='none',
        output_dir=output_dir,
    )

    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward_fn], args=args, train_dataset=train_dataset)

    # run training and monitor format pass rate on a small val split
    try:
        trainer.train()
    except Exception:
        logger.exception('GRPO training failed')
        for h in grad_hooks:
            try:
                h.remove()
            except Exception:
                pass
        return False

    # save adapter
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(output_dir)
    except Exception:
        logger.exception('Failed to save LoRA adapter')
    finally:
        for h in grad_hooks:
            try:
                h.remove()
            except Exception:
                pass
    return True


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH', './downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./format_phase1')
    p.add_argument('--sft_samples', type=int, default=2000)
    p.add_argument('--use_gsm8k', action='store_true', help='Build synthetic formatting samples from GSM8K')
    p.add_argument('--do_ce', action='store_true')
    p.add_argument('--ce_epochs', type=int, default=1)
    p.add_argument('--ce_batch', type=int, default=4, help='Per-device batch size for CE finetune')
    p.add_argument('--ce_device', choices=['auto', 'cpu', 'gpu'], default='auto', help='Device for CE finetune (auto will use GPU if available)')
    p.add_argument('--lora_r', type=int, default=16)
    p.add_argument('--max_steps', type=int, default=200)
    p.add_argument('--reward_mode', choices=['auto', 'strict', 'soft'], default='auto', help='Format reward mode: auto (preflight), strict, or soft')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    logger.info('Starting Phase1: %s', datetime.utcnow().isoformat())

    # load tokenizer + model
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        logger.error('transformers not available in this env; aborting')
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_kwargs = dict(trust_remote_code=True)
    # allow CE to be executed on CPU to avoid GPU OOM for small synthetic finetunes
    if torch is not None:
        if getattr(args, 'ce_device', 'auto') == 'cpu' and getattr(args, 'do_ce', False):
            model_kwargs['device_map'] = 'cpu'
        else:
            # default: load with automatic device placement (GPU if available)
            model_kwargs['device_map'] = 'auto'
        model_kwargs['torch_dtype'] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # add special tokens and resize
    os.makedirs(args.output, exist_ok=True)
    tokenizer, model = add_special_tokens_and_resize(tokenizer, model, save_dir=args.output)

    # build synthetic dataset (optionally from GSM8K)
    if getattr(args, 'use_gsm8k', False):
        logger.info('Building synthetic SFT dataset from GSM8K')
        sft_ds = build_synthetic_from_gsm8k(n_samples=args.sft_samples)
    else:
        sft_ds = build_synthetic_dataset(n_samples=args.sft_samples)
    sft_path = os.path.join(args.output, 'sft_synthetic.jsonl')
    try:
        with open(sft_path, 'w') as f:
            for ex in (sft_ds if isinstance(sft_ds, list) else list(sft_ds)):
                f.write(json.dumps(ex) + '\n')
    except Exception:
        pass

    # optional CE finetune
    if args.do_ce:
        logger.info('Running CE finetune for %d epochs (per-device batch=%d)', args.ce_epochs, args.ce_batch)
        do_ce_finetune(tokenizer, model, sft_ds, output_dir=args.output, epochs=args.ce_epochs, per_device_batch=args.ce_batch)

    # run GRPO format-only to create Adapter A
    logger.info('Running GRPO format-only for Adapter A')
    ok = train_adapter_a_with_grpo(model, tokenizer, sft_ds, os.path.join(args.output, 'adapter_a'), lora_r=args.lora_r, max_steps=args.max_steps, reward_mode=args.reward_mode)
    if not ok:
        logger.error('Adapter A training failed')
        return 2

    logger.info('Phase1 complete; Adapter A saved to %s', os.path.join(args.output, 'adapter_a'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
