#!/usr/bin/env python3
"""Step 2: GRPO format enforcement

Usage: python step2_grpo_format.py --adapter_input <path_to_model_or_adapter> --output <outdir>
"""
import argparse, logging, sys, os
from phase1_utils import build_format_reward_strict, build_format_reward_soft
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

logger = logging.getLogger('step2_grpo_format')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./format_grpo')
    p.add_argument('--sft_samples', type=int, default=200)
    p.add_argument('--max_steps', type=int, default=100)
    p.add_argument('--reward_mode', choices=['strict','soft','auto'], default='auto')
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    if GRPOConfig is None:
        logger.error('TRL GRPO not available in this env; aborting')
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_kwargs = dict(trust_remote_code=True)
    model_kwargs['device_map'] = 'auto'
    model_kwargs['torch_dtype'] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # select reward
    if args.reward_mode == 'strict':
        reward = build_format_reward_strict()
    elif args.reward_mode == 'soft':
        reward = build_format_reward_soft()
    else:
        # default auto: try strict on synthetic sample and fall back
        from phase1_utils import build_synthetic_dataset, format
        s = build_synthetic_dataset(n_samples=8)
        strict = build_format_reward_strict()
        scores = strict([ex['completion'] for ex in s])
        if sum(scores) <= 0:
            reward = build_format_reward_soft()
        else:
            reward = build_format_reward_strict()

    args_cfg = GRPOConfig(
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        max_steps=args.max_steps,
        output_dir=args.output,
    )

    # synthetic train dataset
    from phase1_utils import build_synthetic_dataset
    train_ds = build_synthetic_dataset(n_samples=args.sft_samples)

    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[reward], args=args_cfg, train_dataset=train_ds)
    trainer.train()
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
