#!/usr/bin/env python3
"""Prepare a full SFT JSONL from a large DeepSeek JSONL source.

Heuristics used (in order):
- explicit 'prompt'/'completion'
- 'input' / ('output'|'answer'|'response'|'completion')
- 'instruction' / ('output'|'response')
- 'question' / 'answer'
- choices/messages structures (common chat formats)
- fallback: pick two longest string fields

Completions are canonicalized to <think>...</think><answer>...</answer> unless they already contain both tags.
Duplicates are removed. Up to --max_examples are written.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='deepseek_classified_math_science.jsonl')
    p.add_argument('--out', default='./grpo_portable/sft_from_deepseek_full.canonical.jsonl')
    p.add_argument('--max_examples', type=int, default=8000)
    p.add_argument('--min_prompt_len', type=int, default=20)
    p.add_argument('--min_completion_len', type=int, default=10)
    p.add_argument('--force', action='store_true')
    return p.parse_args(argv)


def has_both_tags(text: str) -> bool:
    return '<think>' in text and '</think>' in text and '<answer>' in text and '</answer>' in text


def canonicalize_completion(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if has_both_tags(text):
        return text
    # Avoid double-wrapping if tags partially present
    # Put reasoning inside <think> and final inside <answer>
    return f"<think>{text}</think><answer>{text}</answer>"


def extract_from_record(rec):
    # Try many common key patterns
    if not isinstance(rec, dict):
        return None
    # 1. prompt/completion
    if 'prompt' in rec and 'completion' in rec:
        return rec['prompt'], rec['completion']

    # 2. input / output variants
    input_keys = ['input', 'prompt', 'question', 'query', 'instruction']
    output_keys = ['output', 'completion', 'response', 'answer', 'text']
    for ik in input_keys:
        if ik in rec:
            for ok in output_keys:
                if ok in rec:
                    return rec[ik], rec[ok]

    # 3. instruction / response
    if 'instruction' in rec:
        for ok in ['response', 'output', 'completion']:
            if ok in rec:
                return rec['instruction'], rec[ok]

    # 4. choices / messages (OpenAI-like)
    if 'choices' in rec and isinstance(rec['choices'], list) and rec['choices']:
        # try to extract from first choice
        ch = rec['choices'][0]
        if isinstance(ch, dict):
            # Chat-style
            if 'message' in ch and isinstance(ch['message'], dict):
                msg = ch['message']
                if 'content' in msg:
                    # if record has prompt elsewhere, pair prompt + content
                    prompt = rec.get('prompt') or rec.get('input') or rec.get('instruction')
                    if prompt:
                        return prompt, msg['content']
                    # fallback: use id or empty prompt
                    return '', msg['content']
            for k in ['text', 'output_text', 'content']:
                if k in ch:
                    prompt = rec.get('prompt') or rec.get('input') or rec.get('instruction')
                    return prompt or '', ch[k]

    if 'messages' in rec and isinstance(rec['messages'], list):
        # collect user/system as prompt, assistant as completion
        user_parts = []
        assistant_parts = []
        for m in rec['messages']:
            if not isinstance(m, dict):
                continue
            role = m.get('role','').lower()
            content = m.get('content') or m.get('text')
            if not isinstance(content, str):
                continue
            if role in ('user','system'):
                user_parts.append(content)
            elif role in ('assistant','bot'):
                assistant_parts.append(content)
        if assistant_parts:
            prompt = '\n'.join(user_parts).strip() if user_parts else ''
            return prompt, '\n'.join(assistant_parts).strip()

    # 5. nested 'data' or 'record'
    for k in ('data','record','example'):
        if k in rec and isinstance(rec[k], dict):
            out = extract_from_record(rec[k])
            if out:
                return out

    # 6. fallback: pick two longest string fields
    strs = [(k, v) for k, v in rec.items() if isinstance(v, str)]
    if len(strs) >= 2:
        strs_sorted = sorted(strs, key=lambda kv: len(kv[1] or ''), reverse=True)
        return strs_sorted[0][1], strs_sorted[1][1]

    return None


def main(argv):
    args = parse_args(argv)

    if not os.path.exists(args.source):
        print('Source file not found:', args.source, file=sys.stderr)
        return 2

    if os.path.exists(args.out) and not args.force:
        print('Output exists; use --force to overwrite:', args.out, file=sys.stderr)
        return 3

    seen = set()
    out_count = 0

    with open(args.source, 'r', encoding='utf-8', errors='ignore') as inf, open(args.out, 'w', encoding='utf-8') as outf:
        for i, line in enumerate(inf):
            if out_count >= args.max_examples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                # try to skip lines that are pure text or prefixed; attempt to heuristically split
                continue

            pair = extract_from_record(rec)
            if not pair:
                continue
            prompt, comp = pair
            if not isinstance(prompt, str) or not isinstance(comp, str):
                try:
                    prompt = str(prompt)
                    comp = str(comp)
                except Exception:
                    continue
            prompt = prompt.strip()
            comp = comp.strip()
            if len(prompt) < args.min_prompt_len or len(comp) < args.min_completion_len:
                continue

            comp_can = canonicalize_completion(comp)

            key = (prompt, comp_can)
            if key in seen:
                continue
            seen.add(key)
            out_rec = {'prompt': prompt, 'completion': comp_can}
            outf.write(json.dumps(out_rec, ensure_ascii=False) + '\n')
            out_count += 1

    print(f'Wrote {out_count} examples to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
