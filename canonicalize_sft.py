#!/usr/bin/env python3
"""Canonicalize SFT JSONL completions to strict <think>...</think><answer>...</answer> format.

Reads an input JSONL with {'prompt','completion'} per line and writes a new JSONL
where each completion is wrapped as:

  <think>...original completion...</think><answer>...original completion...</answer>

If a completion already contains both tags, it is left unchanged.
"""
import argparse
import json
import os
import sys


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--in_file', default='./grpo_portable/sft_from_deepseek.jsonl')
    p.add_argument('--out_file', default='./grpo_portable/sft_from_deepseek.canonical.jsonl')
    p.add_argument('--force', action='store_true', help='Overwrite out_file if exists')
    return p.parse_args(argv)


def canonicalize_completion(text: str) -> str:
    # If already contains both tags, return as-is
    if '<think>' in text and '</think>' in text and '<answer>' in text and '</answer>' in text:
        return text
    # Strip surrounding whitespace
    t = text.strip()
    # Wrap the whole completion in both tags (think holds reasoning, answer holds final answer)
    think = f"<think>{t}</think>"
    answer = f"<answer>{t}</answer>"
    return think + answer


def main(argv):
    args = parse_args(argv)

    if os.path.exists(args.out_file) and not args.force:
        print(f'Output file {args.out_file} exists; use --force to overwrite', file=sys.stderr)
        return 1

    if not os.path.exists(args.in_file):
        print(f'Input file {args.in_file} not found', file=sys.stderr)
        return 1

    written = 0
    with open(args.in_file, 'r', encoding='utf-8') as inf, open(args.out_file, 'w', encoding='utf-8') as outf:
        for i, line in enumerate(inf):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            prompt = rec.get('prompt')
            completion = rec.get('completion')
            if prompt is None or completion is None:
                continue
            new_comp = canonicalize_completion(completion)
            out_rec = {'prompt': prompt, 'completion': new_comp}
            outf.write(json.dumps(out_rec, ensure_ascii=False) + '\n')
            written += 1

    print(f'Wrote {written} canonicalized examples to {args.out_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
