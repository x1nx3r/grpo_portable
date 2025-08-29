#!/usr/bin/env python3
"""Load base model + LoRA adapter, generate completions for sample prompts, and save to ./logs

This script samples prompts from a provided JSONL SFT file (falls back to a few hardcoded prompts
if the file is missing), loads the base model and the PEFT adapter, and writes generation outputs
to the specified output directory.
"""
import argparse
import json
import os
import sys

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


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--adapter', default='./format_sft_lora_run_probe/adapter_probe')
    p.add_argument('--sft_file', default='./grpo_portable/sft_from_deepseek.jsonl')
    p.add_argument('--out_dir', default='./logs')
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--system_prompt', action='store_true', help='Prepend a standard system prompt requesting XML-COT format')
    p.add_argument('--system_prompt_text', default=None, help='Custom system prompt text to prepend when --system_prompt is set')
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    return p.parse_args(argv)


def load_prompts(sft_file, n):
    prompts = []
    if os.path.exists(sft_file):
        try:
            with open(sft_file, 'r', encoding='utf-8') as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict) and 'prompt' in rec:
                        prompts.append(rec['prompt'])
                    if len(prompts) >= n:
                        break
        except Exception:
            pass
    if len(prompts) == 0:
        # fallback prompts
        prompts = [
            "Explain the Pythagorean theorem with a short example.",
            "Solve: If x+2=7, what is x? Show brief reasoning.",
            "Describe why the sky is blue in two sentences.",
            "What is the derivative of sin(x)? Provide reasoning.",
            "Write a one-paragraph summary of photosynthesis."
        ]
    return prompts[:n]


def main(argv):
    args = parse_args(argv)

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        print('transformers not available; aborting', file=sys.stderr)
        return 1
    if PeftModel is None:
        print('peft not available; install `peft` to load LoRA adapters', file=sys.stderr)
        return 1

    # choose dtype/device
    if args.device == 'cpu':
        device_map = 'cpu'
        torch_dtype = torch.float32
    elif args.device == 'gpu':
        device_map = 'auto'
        torch_dtype = getattr(torch, 'float16', torch.float32)
    else:  # auto
        device_map = 'auto'
        if torch is not None and torch.cuda.is_available():
            torch_dtype = getattr(torch, 'float16', torch.float32)
        else:
            torch_dtype = torch.float32

    # Try to load tokenizer from adapter's sibling output directory (where we saved tokenizer during CE)
    def find_tokenizer_dir(adapter_path):
        # check adapter path, its parent, and grandparent for tokenizer files
        candidates = [adapter_path, os.path.dirname(adapter_path), os.path.dirname(os.path.dirname(adapter_path))]
        filenames = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'spiece.model', 'merges.txt', 'special_tokens_map.json')
        for c in candidates:
            if not c:
                continue
            for fn in filenames:
                if os.path.exists(os.path.join(c, fn)):
                    return c
        return None

    tok_dir = find_tokenizer_dir(args.adapter)
    if tok_dir:
        print(f'Loading tokenizer from adapter-sibling directory: {tok_dir}')
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    else:
        print(f'Loading tokenizer from {args.model} (this may be heavy)')
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print('Loading base model...')
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)

    # ensure model embeddings match tokenizer; resize if needed before loading adapter
    try:
        tok_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.shape[0]
        if tok_size != emb_size:
            print(f'Resizing model embeddings: {emb_size} -> {tok_size}')
            model.resize_token_embeddings(tok_size)
    except Exception as e:
        print('Warning: failed to compare/resize embeddings:', e)

    print('Applying adapter from', args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    prompts = load_prompts(args.sft_file, args.n_samples)

    # Optionally prepend a system prompt requesting R1 XML-COT format
    if getattr(args, 'system_prompt', False):
        default_sys = (
            "System: Please respond using R1 XML-COT format.\n"
            "Include a <reasoning> section with a brief chain-of-thought, then an <answer> section with the final concise answer.\n\n"
        )
        sys_text = args.system_prompt_text if args.system_prompt_text is not None else default_sys
        new_prompts = []
        for p in prompts:
            low = p.lower()
            if '<reasoning>' in low or '<answer>' in low:
                new_prompts.append(p)
            else:
                new_prompts.append(sys_text + p)
        prompts = new_prompts

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'adapter_probe_generation.txt')

    with open(out_path, 'w', encoding='utf-8') as outf:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f'Generating for sample {i+1}/{len(prompts)}')
                inputs = tokenizer(prompt, return_tensors='pt')
                # move inputs to model device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                # Attempt to decode only the newly generated tokens (exclude the prompt)
                try:
                    input_len = inputs['input_ids'].shape[1]
                    gen_ids = out_ids[0][input_len:]
                    if gen_ids.numel() == 0:
                        # fallback to full decode
                        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    else:
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                except Exception:
                    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

                # Post-process: ensure we return a well-formed XML-COT with <reasoning> and <answer>
                def ensure_xml_cot(s: str) -> str:
                    low = s.lower()
                    has_reason = '<reasoning>' in low or '<think>' in low
                    has_answer = '<answer>' in low
                    # If already contains both canonical tags, return as-is
                    if has_reason and has_answer:
                        return s
                    # Otherwise, try to split into reasoning + answer heuristically
                    lines = [l.strip() for l in s.splitlines() if l.strip()]
                    if not lines:
                        return "<reasoning>\n(No content)\n</reasoning>\n<answer>\n(No answer)\n</answer>"
                    # heuristic: last line that contains a number or short phrase is answer
                    import re
                    ans = None
                    for ln in reversed(lines[-3:]):
                        m = re.search(r"(-?\d+[\.,]?\d*)$", ln)
                        if m:
                            ans = m.group(1)
                            break
                    if ans is None and len(lines) >= 1:
                        ans = lines[-1]
                    reasoning = "\n".join(lines[:-1]) if len(lines) > 1 else lines[0]
                    return f"<reasoning>\n{reasoning}\n</reasoning>\n<answer>\n{ans}\n</answer>"

                text = ensure_xml_cot(text)
                decor = f"--- SAMPLE {i+1} ---\nPROMPT:\n{prompt}\n\nOUTPUT:\n{text}\n\n"
                print(decor)
                outf.write(decor)

    print('Wrote generations to', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
