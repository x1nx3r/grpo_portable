#!/usr/bin/env python3
"""Run a parameter sweep of generation hyperparameters and log outputs as JSONL.

The script loads the base model and PEFT adapter once, then iterates over a small grid
of generation parameters, generates responses for a set of prompts, and writes a
record per generation to an output JSONL file containing the prompt, config, output,
and timing/metadata.
"""
import argparse
import json
import os
import sys
import time
from itertools import product

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria
except Exception:
    torch = None

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--adapter', default='./format_sft_lora_h100/adapter_lora_h100')
    p.add_argument('--sft_file', default='./sft_from_deepseek.jsonl')
    p.add_argument('--out', default='./logs/adapter_sweep_results.jsonl')
    p.add_argument('--n_samples', type=int, default=3)
    p.add_argument('--max_prompt_tokens', type=int, default=512)
    p.add_argument('--max_new_tokens', type=int, default=128)
    # small default grid; change on CLI if desired
    p.add_argument('--temps', default='0.0,0.7', help='Comma-separated temperatures')
    p.add_argument('--top_ps', default='0.95', help='Comma-separated top_p values')
    p.add_argument('--rep_penalties', default='1.0,1.2', help='Comma-separated repetition penalties')
    p.add_argument('--no_repeat_ngram_sizes', default='0,3', help='Comma-separated ngram sizes (0 disables)')
    p.add_argument('--do_samples', default='False,True', help='Comma-separated booleans for sampling')
    p.add_argument('--stop_on_answer', action='store_true', help='Enable stop-on-</answer> during generation')
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


def _make_stopping_criteria(tokenizer):
    stop_ids = None
    stop_cls = None
    try:
        stop_ids = tokenizer.encode('</answer>', add_special_tokens=False)
        class _StopOnSequence(StoppingCriteria):
            def __init__(self, stop_id_seq, min_length=1):
                self.stop_seq = list(stop_id_seq)
                self.min_length = int(min_length)

            def __call__(self, input_ids, scores, **kwargs):
                seq_len = input_ids.shape[-1]
                if seq_len < self.min_length:
                    return False
                if seq_len >= len(self.stop_seq):
                    last = input_ids[0, -len(self.stop_seq):].tolist()
                    return last == self.stop_seq
                return False
        stop_cls = _StopOnSequence
    except Exception:
        stop_ids = None
        stop_cls = None
    return stop_ids, stop_cls


def main(argv):
    args = parse_args(argv)
    if torch is None:
        print('torch/transformers not available; aborting', file=sys.stderr)
        return 2
    if PeftModel is None:
        print('peft not available; aborting', file=sys.stderr)
        return 2

    # parse grids
    temps = [float(x) for x in args.temps.split(',') if x.strip()]
    top_ps = [float(x) for x in args.top_ps.split(',') if x.strip()]
    rep_penalties = [float(x) for x in args.rep_penalties.split(',') if x.strip()]
    no_repeat = [int(x) for x in args.no_repeat_ngram_sizes.split(',') if x.strip()]
    do_samples = [s.strip().lower() in ('1','true','t','yes') for s in args.do_samples.split(',') if s.strip()]

    print('Grid sizes:', len(temps), len(top_ps), len(rep_penalties), len(no_repeat), len(do_samples))

    # load tokenizer & model
    print('Loading tokenizer from adapter if available...')
    def find_tokenizer_dir(adapter_path):
        candidates = [adapter_path, os.path.dirname(adapter_path), os.path.dirname(os.path.dirname(adapter_path))]
        filenames = ('tokenizer.json','tokenizer_config.json','vocab.json','spiece.model','merges.txt','special_tokens_map.json')
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

    # device/dtype
    device_map = 'auto'
    torch_dtype = getattr(torch, 'float16', torch.float32) if torch.cuda.is_available() else torch.float32

    print('Loading base model...')
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, torch_dtype=torch_dtype, trust_remote_code=True)

    # ensure embeddings match tokenizer
    try:
        tok_size = len(tokenizer)
        emb_size = model.get_input_embeddings().weight.shape[0]
        if tok_size != emb_size:
            print('Resizing model embeddings:', emb_size, '->', tok_size)
            model.resize_token_embeddings(tok_size)
    except Exception:
        pass

    print('Applying adapter from', args.adapter)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    model_max_len = getattr(tokenizer, 'model_max_length', None)
    if model_max_len is None or (isinstance(model_max_len, int) and model_max_len > 10**7):
        model_max_len = getattr(model.config, 'max_position_embeddings', None) or getattr(model.config, 'n_ctx', None) or (args.max_prompt_tokens + args.max_new_tokens)

    stop_ids, stop_cls = _make_stopping_criteria(tokenizer) if args.stop_on_answer else (None, None)

    prompts = load_prompts(args.sft_file, args.n_samples)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    combos = list(product(temps, top_ps, rep_penalties, no_repeat, do_samples))
    print(f'Running sweep with {len(combos)} combos, {len(prompts)} prompts each')

    # open output file for append
    with open(args.out, 'a', encoding='utf-8') as outfh:
        run_id = int(time.time())
        for combo_idx, (temp, top_p, rep_penalty, ngram, do_sample) in enumerate(combos, start=1):
            cfg = {
                'temp': temp,
                'top_p': top_p,
                'repetition_penalty': rep_penalty,
                'no_repeat_ngram_size': ngram,
                'do_sample': bool(do_sample),
                'max_new_tokens': int(args.max_new_tokens),
            }
            print(f'Combo {combo_idx}/{len(combos)}: {cfg}')
            for i, prompt in enumerate(prompts):
                # tokenize and move to device
                inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=int(args.max_prompt_tokens))
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # clamp per prompt
                input_len = inputs['input_ids'].shape[1]
                available_new = max(int(model_max_len) - int(input_len), 0)
                gen_max = min(int(args.max_new_tokens), available_new) if available_new > 0 else 1

                # validate/coerce generation args depending on sampling mode
                gen_kwargs = {
                    'max_new_tokens': int(gen_max),
                    'do_sample': bool(do_sample),
                    'repetition_penalty': float(rep_penalty),
                    'pad_token_id': tokenizer.eos_token_id,
                }
                # only include sampling-specific args when sampling is enabled
                if bool(do_sample):
                    # temperature must be strictly > 0 when sampling
                    t = float(temp)
                    if t <= 0.0:
                        t = 1e-6
                        print(f'  Adjusted temperature to tiny positive value for sampling (was {temp})')
                    gen_kwargs['temperature'] = t
                    gen_kwargs['top_p'] = float(top_p)
                # only include no_repeat_ngram_size when > 0
                if int(ngram) > 0:
                    gen_kwargs['no_repeat_ngram_size'] = int(ngram)

                # stopping criteria
                stopping_list = None
                if stop_cls is not None and stop_ids is not None:
                    try:
                        stopping_list = [stop_cls(stop_ids, min_length=4)]
                    except Exception:
                        stopping_list = None

                try:
                    start = time.time()
                    if stopping_list is not None:
                        out_ids = model.generate(**inputs, **gen_kwargs, stopping_criteria=stopping_list)
                    else:
                        out_ids = model.generate(**inputs, **gen_kwargs)
                    dur = time.time() - start
                    # decode only generated portion
                    input_len = inputs['input_ids'].shape[1]
                    gen_ids = out_ids[0][input_len:]
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids.numel() > 0 else tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    rec = {
                        'run_id': run_id,
                        'combo_idx': combo_idx,
                        'combo': cfg,
                        'prompt_idx': i,
                        'prompt': prompt,
                        'output': text,
                        'duration_s': dur,
                        'timestamp': time.time(),
                    }
                    outfh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    outfh.flush()
                    print(f'  wrote combo {combo_idx} prompt {i} (len {len(text)})')
                except Exception as e:
                    rec = {
                        'run_id': run_id,
                        'combo_idx': combo_idx,
                        'combo': cfg,
                        'prompt_idx': i,
                        'prompt': prompt,
                        'error': str(e),
                        'timestamp': time.time(),
                    }
                    outfh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    outfh.flush()
                    print('  generation error:', e)

    print('Sweep complete. Results in', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
