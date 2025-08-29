#!/usr/bin/env python3
"""Shared utilities for Phase1 scripts (small, safe helpers).

This module exposes tokenizer/model helpers, simple synthetic dataset
builders, and reward builders used by the three-step pipeline scripts.
"""
import logging
import random
import re
import json
import os
from typing import List
try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger('phase1_utils')
logger.addHandler(logging.NullHandler())

SPECIAL_TOKENS = ['<think>', '</think>', '<answer>', '</answer>']
XML_THINK_PAT = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
XML_ANSWER_PAT = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
ALT_THINK_RE = re.compile(r"<(think|reasoning|r1response)[^>]*>.*?</(think|reasoning|r1response)>", flags=re.IGNORECASE | re.DOTALL)
ALT_ANSWER_RE = re.compile(r"<(answer|ans|solution|final_answer|solution_text)[^>]*>.*?</(answer|ans|solution|final_answer|solution_text)>", flags=re.IGNORECASE | re.DOTALL)

def add_special_tokens_and_resize(tokenizer, model, save_dir=None):
    if tokenizer is None or model is None:
        logger.warning('tokenizer/model missing; skip special token step')
        return tokenizer, model
    try:
        added = tokenizer.add_tokens(SPECIAL_TOKENS)
        logger.info('Added %d special tokens', added)
        if added > 0:
            model.resize_token_embeddings(len(tokenizer))
            logger.info('Resized embeddings to %d', len(tokenizer))
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            tokenizer.save_pretrained(save_dir)
    except Exception:
        logger.exception('failed to add special tokens')
    return tokenizer, model

def make_synthetic_format_sample(i:int):
    q = f"What is {i} plus {i}?"
    reasoning = f"{i} + {i} = {i*2}"
    answer = str(i*2)
    prompt = "System: Please answer in R1 XML-COT format.\nUser: " + q + "\n"
    completion = f"<think>{reasoning}</think><answer>{answer}</answer>"
    return {'prompt': prompt, 'completion': completion}

def build_synthetic_dataset(n_samples=2000, seed=42):
    random.seed(seed)
    return [make_synthetic_format_sample(i) for i in range(1, n_samples+1)]

def build_format_reward_strict():
    def reward(completions, **kwargs):
        out = []
        for c in completions:
            try:
                text = c if isinstance(c, str) else (c[0].get('content','') if isinstance(c, (list,tuple)) and len(c)>0 and isinstance(c[0], dict) else str(c))
            except Exception:
                text = ''
            has_think = bool(XML_THINK_PAT.search(text))
            has_answer = bool(XML_ANSWER_PAT.search(text))
            out.append(1.0 if (has_think and has_answer) else -1.0)
        return out
    return reward

def build_format_reward_soft():
    def reward(completions, **kwargs):
        out = []
        for c in completions:
            try:
                text = c if isinstance(c, str) else (c[0].get('content','') if isinstance(c, (list,tuple)) and len(c)>0 and isinstance(c[0], dict) else str(c))
            except Exception:
                text = ''
            has_think = bool(XML_THINK_PAT.search(text))
            has_answer = bool(XML_ANSWER_PAT.search(text))
            alt_think = bool(ALT_THINK_RE.search(text))
            alt_answer = bool(ALT_ANSWER_RE.search(text))
            if has_think and has_answer:
                out.append(1.0)
            elif alt_think and alt_answer:
                out.append(0.8)
            elif has_think or alt_think or has_answer or alt_answer:
                out.append(0.5)
            elif '<' in text and '>' in text:
                out.append(0.2)
            else:
                out.append(-1.0)
        return out
    return reward

def do_ce_finetune(tokenizer, model, sft_dataset, output_dir, epochs=1, per_device_batch=1):
    try:
        from transformers import Trainer, TrainingArguments
    except Exception:
        logger.warning('transformers.Trainer not available; skipping CE finetune')
        return False

    def tokenize_fn(x):
        inp = x['prompt']
        tgt = x['completion']
        enc = tokenizer(inp + '\n' + tgt, truncation=True, padding='max_length', max_length=512)
        enc['labels'] = enc['input_ids'].copy()
        return enc

    try:
        ds = sft_dataset
        if not hasattr(ds, 'map'):
            # convert list to dataset-like list handled by Trainer
            tok_list = [tokenize_fn(x) for x in ds]
            # use simple dict-based dataset wrapper via Dataset.from_dict if available
            try:
                from datasets import Dataset as HFDataset
                tokds = HFDataset.from_list(tok_list)
            except Exception:
                tokds = tok_list
        else:
            tokds = ds.map(lambda x: tokenize_fn(x), batched=False)
    except Exception:
        logger.exception('Failed to tokenize SFT dataset')
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
