#!/usr/bin/env python3
"""Step 1: CE SFT to teach canonical tags

Usage: python step1_ce_sft.py --model <model_path> --output <outdir> --sft_samples 200 --ce_epochs 1
"""
import argparse, logging, sys, os, json
from phase1_utils import add_special_tokens_and_resize, build_synthetic_dataset, do_ce_finetune
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logger = logging.getLogger('step1_ce_sft')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./format_sft')
    p.add_argument('--sft_samples', type=int, default=200)
    p.add_argument('--ce_epochs', type=int, default=1)
    p.add_argument('--ce_batch', type=int, default=1)
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_kwargs = dict(trust_remote_code=True)
    if args.device == 'cpu':
        model_kwargs['device_map'] = 'cpu'
    else:
        model_kwargs['device_map'] = 'auto'
    model_kwargs['torch_dtype'] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    os.makedirs(args.output, exist_ok=True)
    tokenizer, model = add_special_tokens_and_resize(tokenizer, model, save_dir=args.output)

    sft = build_synthetic_dataset(n_samples=args.sft_samples)
    with open(os.path.join(args.output,'sft_small.jsonl'),'w') as f:
        for ex in sft:
            f.write(json.dumps(ex)+"\n")

    do_ce_finetune(tokenizer, model, sft, output_dir=args.output, epochs=args.ce_epochs, per_device_batch=args.ce_batch)

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
