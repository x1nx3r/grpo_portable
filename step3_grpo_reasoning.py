#!/usr/bin/env python3
"""Step 3: GRPO for reasoning quality (gated by format)

Usage: python step3_grpo_reasoning.py --model <model> --reasoning_dataset <jsonl>
"""
import argparse, logging, sys, os, json
from phase1_utils import build_format_reward_strict, build_format_reward_soft
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

logger = logging.getLogger('step3_grpo_reasoning')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./reasoning_grpo')
    p.add_argument('--reasoning_dataset', default='deepseek-r1.jsonl')
    p.add_argument('--max_steps', type=int, default=200)
    return p.parse_args(argv)

def reasoning_reward_stub(completions, **kwargs):
    # Placeholder: encourage longer, non-empty reasoning blocks; real implementation should compare to gold traces
    out = []
    for c in completions:
        try:
            text = c if isinstance(c, str) else (c[0].get('content','') if isinstance(c, (list,tuple)) and len(c)>0 and isinstance(c[0], dict) else str(c))
        except Exception:
            text = ''
        # give modest reward for non-trivial reasoning length
        if len(text.strip()) > 80:
            out.append(1.0)
        elif len(text.strip()) > 20:
            out.append(0.5)
        else:
            out.append(-0.5)
    return out

def main(argv):
    args = parse_args(argv)
    if GRPOConfig is None:
        logger.error('TRL GRPO not available; aborting')
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model_kwargs = dict(trust_remote_code=True)
    model_kwargs['device_map'] = 'auto'
    model_kwargs['torch_dtype'] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # compose rewards: format strict check (gate) + reasoning quality
    format_reward = build_format_reward_strict()
    def composite_reward(completions, **kwargs):
        f = format_reward(completions)
        r = reasoning_reward_stub(completions)
        out = []
        for fi, ri in zip(f, r):
            # require format pass (fi>0) to allow reasoning reward, otherwise heavy penalty
            if fi > 0:
                out.append(ri)
            else:
                out.append(-1.0)
        return out

    # load reasoning dataset (expect JSONL of {prompt, completion})
    train_ds = []
    try:
        with open(args.reasoning_dataset,'r') as fh:
            for line in fh:
                train_ds.append(json.loads(line))
    except Exception:
        logger.warning('Failed to load reasoning dataset; falling back to synthetic')
        from phase1_utils import build_synthetic_dataset
        train_ds = build_synthetic_dataset(n_samples=200)

    args_cfg = GRPOConfig(
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        max_steps=args.max_steps,
        output_dir=args.output,
    )

    trainer = GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[composite_reward], args=args_cfg, train_dataset=train_ds)
    trainer.train()
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
