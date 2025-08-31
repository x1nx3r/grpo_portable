#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a Hugging Face base model and save a merged HF checkpoint.

This script attempts to load the base model and the PEFT adapter, merge the adapter weights
into the base model (using peft's merge API when available), and save the merged model
to a directory you can then convert to GGUF with your preferred converter.

Usage (example):
  python3 scripts/merge_adapter_to_hf.py \
      --base ./downloaded_models/llama-3.2-3b \
      --adapter ./format_sft_lora_h100/adapter_lora_h100 \
      --out ./merged_model

Notes:
- This script requires `transformers`, `torch`, and `peft` to be installed in the environment.
- If your `peft` version does not expose `merge_and_unload`, the script will instruct you to
  upgrade `peft` (or manually merge using an alternative flow).
"""
import argparse
import os
import shutil
import sys

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True, help='Path to base HF model (local directory or HF name)')
    p.add_argument('--adapter', required=True, help='Path to PEFT adapter directory (local)')
    p.add_argument('--out', required=True, help='Output directory for merged HF model')
    p.add_argument('--device', choices=['cpu','auto','gpu'], default='auto', help='Device to load model on for merging')
    return p.parse_args(argv)


def copy_tokenizer(adapter_dir, out_dir):
    candidates = ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'spiece.model']
    os.makedirs(out_dir, exist_ok=True)
    copied = []
    for fn in candidates:
        src = os.path.join(adapter_dir, fn)
        if os.path.exists(src):
            dst = os.path.join(out_dir, fn)
            shutil.copy(src, dst)
            copied.append(fn)
    return copied


def main(argv):
    args = parse_args(argv)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        print('ERROR: transformers and torch are required to run this script. Install them and retry.', file=sys.stderr)
        print('Detail:', e, file=sys.stderr)
        return 2

    try:
        from peft import PeftModel
    except Exception as e:
        print('ERROR: peft is required to run this script. Install/upgrade peft and retry.', file=sys.stderr)
        print('Detail:', e, file=sys.stderr)
        return 3

    # Choose device map and dtype. Prefer bfloat16 on H100 if available.
    if args.device == 'cpu':
        device_map = 'cpu'
        torch_dtype = torch.float32
    elif args.device == 'gpu':
        device_map = 'auto'
        # prefer bfloat16 (H100) then float16
        torch_dtype = getattr(torch, 'bfloat16', getattr(torch, 'float16', torch.float32))
    else:  # auto
        device_map = 'auto'
        if torch.cuda.is_available():
            torch_dtype = getattr(torch, 'bfloat16', getattr(torch, 'float16', torch.float32))
        else:
            torch_dtype = torch.float32

    print(f'Using device_map={device_map}, torch_dtype={torch_dtype}')

    print('Loading tokenizer (adapter-sibling -> adapter -> base)')
    tok_dir = None
    for cand in (args.adapter, os.path.dirname(args.adapter), args.base):
        if not cand:
            continue
        # prefer tokenizer.json or tokenizer_config.json
        for fn in ('tokenizer.json','tokenizer_config.json','spiece.model','vocab.json','merges.txt'):
            if os.path.exists(os.path.join(cand, fn)):
                tok_dir = cand
                break
        if tok_dir:
            break

    if tok_dir is None:
        print('No tokenizer found in adapter or base path; attempting to load tokenizer from base via HF name')
        tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    else:
        print(f'Loading tokenizer from: {tok_dir}')
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)

    print('Loading base model (this may be large; loading on CPU by default)')
    model = AutoModelForCausalLM.from_pretrained(args.base, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)

    # ensure embeddings match tokenizer
    try:
        tok_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.shape[0]
        if tok_size != emb_size:
            print(f'Resizing model embeddings: {emb_size} -> {tok_size}')
            model.resize_token_embeddings(tok_size)
    except Exception as e:
        print('Warning: failed to compare/resize embeddings:', e)

    print('Loading PEFT adapter and applying to model (in-memory)')
    peft_model = PeftModel.from_pretrained(model, args.adapter)

    # attempt to use the peft merge API if available
    if hasattr(peft_model, 'merge_and_unload'):
        print('Merging adapter into base model via peft.merge_and_unload()...')
        try:
            merged = peft_model.merge_and_unload()
            # merge_and_unload may return the merged model or modify in-place; coerce to a model object
            merged_model = merged if merged is not None else model
        except Exception as e:
            print('ERROR: merge_and_unload() failed:', e, file=sys.stderr)
            return 4
    else:
        print('Your peft version does not provide merge_and_unload().')
        print('Please upgrade peft (pip install -U peft) and retry, or run an alternate merge flow.')
        return 5

    # Save merged HF model
    out_dir = os.path.abspath(args.out)
    print('Saving merged HF model to', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    try:
        merged_model.save_pretrained(out_dir)
    except Exception as e:
        print('ERROR: failed to save merged model:', e, file=sys.stderr)
        return 6

    # copy tokenizer files from tokenizer source into out_dir so the merged artifact is self-contained
    copied = copy_tokenizer(tok_dir if tok_dir is not None else args.base, out_dir)
    if copied:
        print('Copied tokenizer files:', copied)
    else:
        print('No tokenizer files detected to copy; the merged model directory may still be usable via HF hub name')

    print('\nMerged model saved. Next steps (examples):')
    print(' - (Optional) Quantize merged HF model to reduce size (e.g., with bitsandbytes/GPTQ tooling).')
    print(' - Convert merged HF checkpoint to GGUF using your preferred converter (text-generation-webui, ggml converter, etc.).')
    print('   Example (text-generation-webui style convert script):')
    print('     python3 converters/convert_hf_to_gguf.py --model-dir', out_dir, '--outfile merged.gguf')
    print('\nIf you want, I can attempt to run a known converter automatically — tell me which converter is available on this machine and I will try to call it.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
