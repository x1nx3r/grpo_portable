#!/usr/bin/env python3
"""GRPO GSM8K Pipeline (no Unsloth)

This script mirrors the original pipeline but avoids importing Unsloth entirely.
It uses Hugging Face Transformers + PEFT (LoRA) + TRL's GRPOTrainer where
available. It's conservative and intended as a drop-in alternative to test
whether Unsloth-specific compilation/tracing was causing the NaNs.

Note: this script uses safe imports so it can be inspected without all deps.
"""
import logging
import os
import sys
from datetime import datetime

try:
    from datasets import load_dataset, Dataset
except Exception:
    load_dataset = None
    Dataset = list

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None

try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None
    get_peft_model = None

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception:
    GRPOConfig = None
    GRPOTrainer = None

import re
import numpy as np

logger = logging.getLogger('GRPO_GSM8K_no_unsloth')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# --- reward and utility functions (copied/adapted) ---

def extract_xml_answer(text: str) -> str:
    parts = text.split("<answer>")
    if len(parts) < 2:
        return ""
    answer = parts[-1].split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def _sanitize_reward_list(values, minlength=0, minv=-1000.0, maxv=1000.0):
    try:
        arr = np.array(values, dtype=np.float64)
    except Exception:
        arr = np.zeros(minlength if minlength > 0 else 0, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=maxv, neginf=minv)
    arr = np.clip(arr, minv, maxv)
    if minlength > 0 and arr.size < minlength:
        pad = np.zeros(minlength - arr.size, dtype=np.float64)
        arr = np.concatenate([arr, pad])
    return arr.tolist()


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content'] if isinstance(prompts, list) and prompts and isinstance(prompts[0], list) else ''
    extracted_responses = [extract_xml_answer(r) for r in responses]
    logger.info('-'*8 + f" Question:\n{q} \nAnswer:\n{answer[0] if isinstance(answer, list) and answer else answer}\nResponse:\n{responses[0] if responses else ''}\nExtracted:\n{extracted_responses[0] if extracted_responses else ''}")
    try:
        raw = [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
        return _sanitize_reward_list(raw, minlength=len(completions))
    except Exception:
        return _sanitize_reward_list([], minlength=len(completions))


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    try:
        raw = [0.5 if (isinstance(r, str) and r.isdigit()) else 0.0 for r in extracted_responses]
        return _sanitize_reward_list(raw, minlength=len(completions))
    except Exception:
        return _sanitize_reward_list([], minlength=len(completions))


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    try:
        matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
        raw = [0.5 if match else 0.0 for match in matches]
        return _sanitize_reward_list(raw, minlength=len(completions))
    except Exception:
        return _sanitize_reward_list([], minlength=len(completions))


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    try:
        matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
        raw = [0.5 if m else 0.0 for m in matches]
        return _sanitize_reward_list(raw, minlength=len(completions))
    except Exception:
        return _sanitize_reward_list([], minlength=len(completions))


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    try:
        out = [count_xml(c) for c in contents]
        return _sanitize_reward_list(out, minlength=len(completions))
    except Exception:
        return _sanitize_reward_list([], minlength=len(completions))


# --- gradient sanitizer (reused) ---

def _register_grad_sanitizer(model, max_abs=1e4):
    handles = []
    if model is None or torch is None:
        logger.info('torch not available; skipping grad sanitizer registration')
        return handles

    def make_hook(name):
        def _hook(grad):
            try:
                if grad is None:
                    return grad
                if not torch.isfinite(grad).all():
                    logger.warning(f'Non-finite grad detected in {name}; sanitizing')
                    grad = torch.nan_to_num(grad, nan=0.0, posinf=max_abs, neginf=-max_abs)
                grad = torch.clamp(grad, -max_abs, max_abs)
                return grad
            except Exception:
                logger.exception(f'Exception in grad hook for {name}');
                return grad
        return _hook

    try:
        for n, p in model.named_parameters():
            if p.requires_grad:
                try:
                    h = p.register_hook(make_hook(n))
                    handles.append(h)
                except Exception:
                    continue
    except Exception:
        logger.exception('Failed to register gradient sanitizer hooks')

    logger.info(f'Installed {len(handles)} gradient sanitizer hooks')
    return handles


def get_gsm8k_questions(split: str = "train") -> object:
    if load_dataset is None:
        raise RuntimeError("datasets.load_dataset not available in this environment")
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore

    def _map_fn(x):
        q = x['question']
        ans = extract_hash_answer(x['answer'])
        return {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q}
            ],
            'answer': ans,
        }

    try:
        data = data.map(lambda x: _map_fn(x))  # type: ignore
    except Exception:
        processed = [_map_fn(x) for x in data]
        try:
            data = Dataset.from_list(processed)  # type: ignore
        except Exception:
            data = processed

    return data


def main():
    logger.info(f"Starting GSM8K GRPO pipeline (no Unsloth) at {datetime.utcnow().isoformat()} UTC")

    if AutoTokenizer is None or AutoModelForCausalLM is None or GRPOConfig is None or GRPOTrainer is None:
        logger.error("Required training libraries not available (transformers/trl). Inspect the script; install deps to run.")
        return 1

    # Load dataset
    logger.info("Loading GSM8K dataset (train split)")
    dataset = get_gsm8k_questions('train')
    logger.info(f"Dataset loaded: {len(dataset) if hasattr(dataset, '__len__') else 'unknown'} samples")

    force_fp32 = os.environ.get('FORCE_FP32', '0') == '1'
    logger.info(f"Loading model/tokenizer via Transformers force_fp32={force_fp32}")

    # tokenizer + model
    # prefer a locally downloaded model directory (set via MODEL_LOCAL_PATH env)
    local_model_dir = os.environ.get('MODEL_LOCAL_PATH', './downloaded_models/llama-3.2-3b')
    hf_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    preferred = local_model_dir if os.path.isdir(local_model_dir) else hf_model_id
    logger.info(f"Attempting to load model/tokenizer from: {preferred}")

    tokenizer = AutoTokenizer.from_pretrained(preferred, use_fast=True)
    model_kwargs = dict(trust_remote_code=True)
    if torch is not None:
        if not force_fp32:
            model_kwargs.update(dict(torch_dtype=torch.float16, device_map='auto'))
        else:
            model_kwargs.update(dict(torch_dtype=torch.float32, device_map='auto'))

    model = AutoModelForCausalLM.from_pretrained(preferred, **model_kwargs)

    # Apply LoRA via PEFT if available
    if get_peft_model is not None and LoraConfig is not None:
        try:
            lora_config = LoraConfig(r=16, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], lora_alpha=16, bias='none')
            model = get_peft_model(model, lora_config)
        except Exception:
            logger.exception('Failed to apply LoRA via PEFT; continuing with base model')

    # install grad sanitizer
    grad_hooks = _register_grad_sanitizer(model)

    # training args
    training_args = GRPOConfig(
        learning_rate=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=256,
        max_steps=10,
        save_steps=10,
        max_grad_norm=0.5,
        report_to="none",
        output_dir="./gsm8k_outputs_no_unsloth",
    )

    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    # safe optimizer step wrapper (same pattern)
    try:
        optim = getattr(trainer, 'optimizer', None)
        if optim is None:
            optim = getattr(trainer, 'optim', None)
    except Exception:
        optim = None

    if optim is not None:
        try:
            orig_step = getattr(optim, 'step')
            def _safe_step(*args, **kwargs):
                try:
                    nan_found = False
                    for pg in getattr(optim, 'param_groups', []):
                        for p in pg.get('params', []):
                            if p.grad is None:
                                continue
                            g = p.grad
                            if not torch.isfinite(g).all():
                                nan_found = True
                                break
                        if nan_found:
                            break
                    if nan_found:
                        logger.warning('Non-finite gradients detected; skipping optimizer.step() and zeroing grads.')
                        for pg in getattr(optim, 'param_groups', []):
                            for p in pg.get('params', []):
                                if p.grad is not None:
                                    try:
                                        p.grad.detach_()
                                        p.grad.zero_()
                                    except Exception:
                                        pass
                        return None
                    return orig_step(*args, **kwargs)
                except Exception as e:
                    logger.exception(f'Error in safe optimizer step wrapper: {e}')
                    return orig_step(*args, **kwargs)
            setattr(optim, 'step', _safe_step)
            logger.info('Wrapped trainer optimizer.step with safety-checker')
        except Exception:
            logger.exception('Failed to wrap optimizer.step()')

    # preflight reward checks
    def _preflight_check(sample, reward_funcs):
        dummy_completion = [[{'content': XML_COT_FORMAT.format(reasoning='1+1=2', answer='2')}]]
        prompts = sample['prompt'] if isinstance(sample, dict) and 'prompt' in sample else [
            [{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': '1+1?'}]
        ]
        answer = sample.get('answer') if isinstance(sample, dict) else None
        batch_size = len(dummy_completion)
        for fn in reward_funcs:
            try:
                res = None
                try:
                    res = fn(prompts, dummy_completion, [answer])
                except TypeError:
                    try:
                        res = fn(dummy_completion)
                    except TypeError:
                        res = fn(prompts, dummy_completion)
                if not isinstance(res, (list, tuple)):
                    logger.error(f"Preflight: reward function {fn.__name__} returned non-list: {type(res)}")
                    return False
                arr = np.array(list(res), dtype=np.float64)
                if arr.size < batch_size:
                    logger.error(f"Preflight: reward function {fn.__name__} returned too-short list {arr.size} < {batch_size}")
                    return False
                if not np.all(np.isfinite(arr)):
                    logger.error(f"Preflight: reward function {fn.__name__} produced non-finite values: {arr}")
                    return False
            except Exception as e:
                logger.exception(f"Preflight: reward function {fn.__name__} raised: {e}")
                return False
        return True

    logger.info("Running reward preflight check before training")
    sample0 = None
    try:
        if hasattr(dataset, '__getitem__'):
            sample0 = dataset[0]
        elif isinstance(dataset, list) and dataset:
            sample0 = dataset[0]
    except Exception:
        sample0 = None

    if not _preflight_check(sample0 or {}, reward_funcs):
        logger.error("Preflight reward checks failed — aborting short run to avoid NaNs.")
        return 2

    # optionally enable anomaly debugging - keep but it's okay if environment doesn't support
    debug_anomaly = os.environ.get('DEBUG_ANOMALY', '0') == '1'
    if debug_anomaly:
        logger.info('DEBUG_ANOMALY=1: tightening training args for fast repro')
        try:
            training_args.max_steps = 2
            training_args.num_generations = 1
            training_args.gradient_accumulation_steps = 1
        except Exception:
            pass

    logger.info('Starting training (no Unsloth)')
    try:
        trainer.train()
    except Exception as e:
        logger.exception(f'Training failed: {e}')
        try:
            for h in grad_hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        except Exception:
            pass
        return 3

    logger.info('Training finished; saving LoRA')
    try:
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained('./gsm8k_lora_no_unsloth')
    except Exception as e:
        logger.warning(f'Failed to save LoRA/model: {e}')
    finally:
        try:
            for h in grad_hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
