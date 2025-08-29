#!/usr/bin/env python3
"""Simple Unsloth downloader

Downloads a model via `unsloth.FastLanguageModel.from_pretrained` and saves the
model and tokenizer to a local directory for offline use. Includes fallbacks
and clear error messages for gated repos.

Usage examples:
  FORCE_FP32=1 python unsloth_download_and_save.py --model meta-llama/Llama-3.2-3B-Instruct --out ./llama-3.2-3b
  python unsloth_download_and_save.py --model facebook/opt-350m --out ./opt-350m

The script is intentionally defensive: it uses safe imports and prints guidance
if the model repo is gated.
"""
import argparse
import logging
import os
import sys
from datetime import datetime

try:
    import unsloth
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None

try:
    import torch
except Exception:
    torch = None

logger = logging.getLogger('unsloth_downloader')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def save_model_and_tokenizer(model, tokenizer, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    # Try tokenizer save
    try:
        if tokenizer is not None and hasattr(tokenizer, 'save_pretrained'):
            tokenizer.save_pretrained(out_dir)
            logger.info('Tokenizer saved to %s', out_dir)
            saved.append('tokenizer')
    except Exception as e:
        logger.warning('Failed to save tokenizer via save_pretrained: %s', e)

    # Try model.save_pretrained
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(out_dir)
            logger.info('Model saved to %s via save_pretrained()', out_dir)
            saved.append('model_save_pretrained')
            return True
    except Exception as e:
        logger.warning('model.save_pretrained() failed: %s', e)

    # Fallback: try to save state_dict if available
    try:
        state = None
        if hasattr(model, 'state_dict'):
            state = model.state_dict()
        elif hasattr(model, 'module') and hasattr(model.module, 'state_dict'):
            state = model.module.state_dict()
        if state is not None and torch is not None:
            torch.save(state, os.path.join(out_dir, 'pytorch_model.bin'))
            logger.info('Saved model.state_dict() to %s', os.path.join(out_dir, 'pytorch_model.bin'))
            saved.append('state_dict')
            return True
    except Exception as e:
        logger.warning('Failed to save state_dict: %s', e)

    # Last resort: try to use HuggingFace snapshot via push/pull not attempted here
    if not saved:
        logger.error('Unable to save model or tokenizer with available methods. Inspect the loaded objects.')
        return False
    return True


def download_model(model_name: str, out_dir: str, force_fp32: bool = False, load_in_4bit: bool = False):
    if FastLanguageModel is None:
        logger.error('Unsloth not available in this environment. Install unsloth to run this script.')
        return 2

    logger.info('Starting download at %s', datetime.utcnow().isoformat())
    logger.info('Model: %s  out_dir: %s  force_fp32: %s  load_in_4bit: %s', model_name, out_dir, force_fp32, load_in_4bit)

    try:
        model_kwargs = dict(max_seq_length=1024, fast_inference=True)
        # mirror earlier code's args: prefer fp32 if forced
        if force_fp32:
            model_kwargs['load_in_4bit'] = False
        else:
            model_kwargs['load_in_4bit'] = load_in_4bit

        # Attempt to load (this will perform any HF download needed)
        lm, tokenizer = FastLanguageModel.from_pretrained(model_name, **model_kwargs)
        logger.info('Model and tokenizer loaded successfully')
    except Exception as e:
        # try to detect gated repo error from huggingface_hub
        msg = str(e)
        logger.exception('Failed to load model: %s', e)
        if 'gated' in msg.lower() or 'forbidden' in msg.lower() or '403' in msg:
            logger.error('It looks like the model repo is gated (HTTP 403). Please request access on Hugging Face or use a public model.')
            return 3
        return 1

    ok = save_model_and_tokenizer(lm, tokenizer, out_dir)
    if not ok:
        return 4

    logger.info('Download and save completed successfully')
    return 0


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', '-m', default=os.environ.get('MODEL_NAME', 'meta-llama/Llama-3.2-3B-Instruct'), help='Model repo id to download')
    p.add_argument('--out', '-o', default=os.environ.get('MODEL_OUT', './downloaded_model'), help='Local directory to save model/tokenizer')
    p.add_argument('--fp32', action='store_true', help='Force FP32 load')
    p.add_argument('--no-4bit', dest='no4bit', action='store_true', help="Don't attempt 4-bit loading")
    return p.parse_args(argv)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    force_fp32 = args.fp32 or os.environ.get('FORCE_FP32', '0') == '1'
    load_in_4bit = not args.no4bit and not force_fp32
    rc = download_model(args.model, args.out, force_fp32=force_fp32, load_in_4bit=load_in_4bit)
    raise SystemExit(rc)
