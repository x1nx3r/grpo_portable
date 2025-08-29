#!/usr/bin/env python3
"""Run guided generation: prefix prompts with a tag-instruction and generate with sampling.

Writes outputs to ./logs_guided/adapter_guided_generation.txt
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
    p.add_argument('--adapter', default='./format_sft_lora_run_canonical/adapter_canonical_probe')
    p.add_argument('--sft_file', default='./grpo_portable/sft_from_deepseek.canonical.jsonl')
    p.add_argument('--out_dir', default='./logs_guided')
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    p.add_argument('--temp', type=float, default=0.2)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--max_new_tokens', type=int, default=256)
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

    if AutoTokenizer is None or AutoModelForCausalLM is None or PeftModel is None:
        print('Required packages missing (transformers/peft).', file=sys.stderr)
        return 1

    # device and dtype
    if args.device == 'cpu':
        device_map = 'cpu'
        torch_dtype = torch.float32
    elif args.device == 'gpu':
        device_map = 'auto'
        torch_dtype = getattr(torch, 'float16', torch.float32)
    else:
        device_map = 'auto'
        torch_dtype = getattr(torch, 'float16', torch.float32) if torch.cuda.is_available() else torch.float32

    # load tokenizer (prefer adapter sibling dir)
    def find_tokenizer_dir(adapter_path):
        candidates = [adapter_path, os.path.dirname(adapter_path), os.path.dirname(os.path.dirname(adapter_path))]
        filenames = ('tokenizer.json','tokenizer_config.json','vocab.json','spiece.model','merges.txt')
        for c in candidates:
            if not c:
                continue
            for fn in filenames:
                if os.path.exists(os.path.join(c, fn)):
                    return c
        return None

    tok_dir = find_tokenizer_dir(args.adapter)
    if tok_dir:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)
    # resize embeddings if tokenizer differs
    try:
        tok_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.shape[0]
        if tok_size != emb_size:
            print(f'Resizing embeddings: {emb_size} -> {tok_size}')
            model.resize_token_embeddings(tok_size)
    except Exception:
        pass

    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = load_prompts(args.sft_file, args.n_samples)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, 'adapter_guided_generation.txt')

    with open(out_path, 'w', encoding='utf-8') as outf:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                guided = 'Please respond using <think>...</think> for reasoning and <answer>...</answer> for the final answer.\n<think> ' + prompt
                inputs = tokenizer(guided, return_tensors='pt')
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out_ids = model.generate(**inputs, do_sample=True, temperature=args.temp, top_p=args.top_p, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                decor = f"--- GUIDED SAMPLE {i+1} ---\nPROMPT:\n{guided}\n\nOUTPUT:\n{text}\n\n"
                print(decor)
                outf.write(decor)

    print('Wrote guided generations to', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
