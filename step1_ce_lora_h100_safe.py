#!/usr/bin/env python3
"""H100-optimized Step 1 (LoRA) - safe defaults copy

This file is a duplicate of `step1_ce_lora_h100.py` with small, safe changes:
- pack_examples default enabled
- cap model-facing tokenization to `training_max = min(args.max_length, 2048)` to limit activation size
- trim completions at string level to preserve closing </answer> tag before token-level truncation
"""

# Set environment variables early to avoid tokenizer parallelism warnings and
# make the CUDA allocator more resilient to fragmentation. These must be set
# before importing transformers/tokenizers so they take effect in worker forks.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
import sys
import json
import random
from typing import Any, Callable, Dict, Optional, cast

from phase1_utils import add_special_tokens_and_resize, build_synthetic_dataset, SPECIAL_TOKENS

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    Trainer = None
    TrainingArguments = None

try:
    from peft import LoraConfig, get_peft_model
    try:
        from peft import prepare_model_for_kbit_training
    except Exception:
        prepare_model_for_kbit_training = None
except Exception:
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None

try:
    # bitsandbytes and the transformers BitsAndBytesConfig enable 4-bit loading
    from transformers import BitsAndBytesConfig
    import bitsandbytes as bnb  # type: ignore
except Exception:
    BitsAndBytesConfig = None  # type: ignore
    bnb = None

logger = logging.getLogger('step1_ce_lora_h100_safe')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=os.environ.get('MODEL_LOCAL_PATH','./downloaded_models/llama-3.2-3b'))
    p.add_argument('--output', default='./format_sft_lora_h100_safe')
    p.add_argument('--sft_samples', type=int, default=8000)
    # Use a generous max_length so we don't truncate completions (many completions end with </answer>).
    # Packing via --pack_examples will mitigate padding overhead by concatenating short examples.
    p.add_argument('--max_length', type=int, default=10000, help='Maximum token length for prompt+completion (large by default; use packing to avoid wasted padding)')
    # enable packing by default in this safe variant
    p.add_argument('--pack_examples', action='store_true', default=True, help='Enable packing multiple short examples into longer sequences to reduce padding waste (default: True)')
    # default to repo-root JSONL (previous default pointed to ./grpo_portable/...,)
    p.add_argument('--sft_file', default='./sft_from_deepseek_full.canonical.jsonl',
                   help='Path to JSONL SFT file with {"prompt","completion"} per line (optional)')
    p.add_argument('--ce_epochs', type=int, default=3)
    # smaller per-device batch to be safe with long contexts
    p.add_argument('--ce_batch', type=int, default=2, help='Per-device train batch size (safe default)')
    # larger LoRA rank for richer adapters
    p.add_argument('--lora_r', type=int, default=64)
    p.add_argument('--lora_alpha', type=int, default=128)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_accum', type=int, default=16, help='Gradient accumulation steps (baked to preserve effective batch)')
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    p.add_argument('--save_adapter_name', default='adapter_lora_h100_safe')
    # enable gradient checkpointing by default on H100; keep flag to allow future explicit control
    p.add_argument('--use_grad_checkpoint', action='store_true', default=True,
                   help='Enable gradient checkpointing if supported (default: True)')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        logger.error('transformers not available in this environment; aborting')
        return 1

    if get_peft_model is None or LoraConfig is None:
        logger.error('peft not available; install `peft` to run LoRA CE')
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # local handle for optional torch module
    torch_mod = globals().get('torch', None)

    # allow mixed-type values (bool, str, dtype) in model kwargs
    model_kwargs: Dict[str, Any] = {'trust_remote_code': True}
    # On H100 prefer bf16 when supported by torch and not running on cpu
    # Determine use_bf16 conservatively
    use_bf16 = bool(torch_mod is not None and args.device != 'cpu' and hasattr(torch_mod, 'bfloat16'))

    if args.device == 'cpu':
        model_kwargs['device_map'] = 'cpu'
        if torch_mod is not None:
            model_kwargs['dtype'] = torch_mod.float32
    else:
        model_kwargs['device_map'] = 'auto'
        # prefer bfloat16 when available; otherwise use float32
        if torch_mod is not None:
            model_kwargs['dtype'] = getattr(torch_mod, 'bfloat16', torch_mod.float32) if use_bf16 else torch_mod.float32

    logger.info('Loading base model (H100 tune: bf16 recommended)')
    # If bitsandbytes and BitsAndBytesConfig are present, prefer 4-bit loading
    try:
        if BitsAndBytesConfig is not None and bnb is not None:
            logger.info('bitsandbytes detected: attempting 4-bit (nf4) load via BitsAndBytesConfig')
            # determine compute dtype for 4-bit operations (avoid referencing global torch when absent)
            if torch_mod is not None:
                bnb_compute_dtype = getattr(torch_mod, 'bfloat16', torch_mod.float16)
            else:
                bnb_compute_dtype = None

            bnb_conf = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            logger.info('BitsAndBytesConfig: %s', repr(bnb_conf))
            try:
                model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_conf, device_map='auto', trust_remote_code=True)
                logger.info('Model loaded with quantization_config (attempted 4-bit load)')
            except Exception:
                logger.exception('4-bit load via bitsandbytes failed; falling back to regular load')
                model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

            # prepare model for k-bit training if available and log result
            try:
                if prepare_model_for_kbit_training is not None:
                    model = prepare_model_for_kbit_training(model)
                    logger.info('Called prepare_model_for_kbit_training successfully')
            except Exception:
                logger.exception('prepare_model_for_kbit_training failed (continuing)')
        else:
            logger.info('bitsandbytes or BitsAndBytesConfig not available; loading model without 4-bit quantization')
            model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    except Exception:
        # fallback to regular load
        logger.exception('Primary model load path raised an exception; attempting fallback load')
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Post-load inspection: try to detect whether quantized/4-bit modules are present
    try:
        # log basic model device/dtype info
        try:
            # find first parameter to report dtype/device
            first_param = next(model.parameters())
            logger.info('Loaded model first_param dtype=%s device=%s', str(first_param.dtype), str(first_param.device))
        except StopIteration:
            logger.info('Loaded model has no parameters?')

        # detect modules that look like quantized/bnb modules
        quant_mods = []
        for m in model.modules():
            n = m.__class__.__name__.lower()
            if '4bit' in n or 'quant' in n or 'bnb' in n or 'linear4bit' in n or 'quantized' in n:
                quant_mods.append(m.__class__.__name__)
        logger.info('Detected %d candidate quantized modules (sample up to 8): %s', len(quant_mods), str(quant_mods[:8]))

        # try attribute flags commonly set by quant loaders
        try:
            is_4bit_flag = getattr(model, 'is_loaded_in_4bit', None)
            if is_4bit_flag is not None:
                logger.info('Model attribute is_loaded_in_4bit=%s', str(is_4bit_flag))
        except Exception:
            pass
    except Exception:
        logger.exception('Post-load inspection failed (continuing)')

    os.makedirs(args.output, exist_ok=True)
    tokenizer, model = add_special_tokens_and_resize(tokenizer, model, save_dir=args.output)

    # Log special-token wiring so we can verify tokens are single-tokenized and registered
    try:
        pad_id = getattr(tokenizer, 'pad_token_id', None)
        eos_id = getattr(tokenizer, 'eos_token_id', None)
        logger.info('Tokenizer pad_token_id=%s eos_token_id=%s', str(pad_id), str(eos_id))
        for st in SPECIAL_TOKENS:
            try:
                # token id (best-effort)
                tok_id = None
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    tok_id = tokenizer.convert_tokens_to_ids(st)
                else:
                    enc = tokenizer(st, add_special_tokens=False)
                    ids = enc.get('input_ids', [])
                    tok_id = ids[0] if ids else None

                # tokenization pieces
                pieces = None
                if hasattr(tokenizer, 'tokenize'):
                    try:
                        pieces = tokenizer.tokenize(st)
                    except Exception:
                        pieces = None
                else:
                    try:
                        pieces = tokenizer(st, add_special_tokens=False).get('input_ids', None)
                    except Exception:
                        pieces = None

                # whether listed as additional_special_tokens
                is_additional = False
                try:
                    sp = getattr(tokenizer, 'additional_special_tokens', None)
                    is_additional = bool(sp and st in sp)
                except Exception:
                    is_additional = False

                logger.info('Special token %s -> id=%s added_special=%s pieces=%s', st, str(tok_id), is_additional, str(pieces))
            except Exception:
                logger.exception('Failed to inspect special token %s', st)
    except Exception:
        logger.exception('Failed to log tokenizer special-token info')

    # optionally enable gradient checkpointing early (before PEFT wrapping) to save activation memory
    if args.use_grad_checkpoint:
        try:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                # disable use_cache to be compatible with gradient checkpointing
                if hasattr(model, 'config'):
                    try:
                        model.config.use_cache = False
                    except Exception:
                        logger.debug('Failed to set model.config.use_cache=False')
                logger.info('Enabled model.gradient_checkpointing and set use_cache=False')
        except Exception:
            logger.exception('Failed to enable gradient checkpointing (continuing)')

    # apply LoRA adapter
    lora_conf = LoraConfig(
        r=args.lora_r,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM'
    )  # type: ignore
    try:
        model = get_peft_model(model, lora_conf)
    except Exception:
        logger.exception('Failed to apply LoRA')
        return 1

    # ensure only LoRA params will be trained
    for n, p in model.named_parameters():
        if 'lora' not in n and 'adapter' not in n:
            p.requires_grad = False

    # load SFT dataset: prefer JSONL file when present, else build synthetic dataset
    sft = None
    if args.sft_file:
        try:
            # prefer the path provided, but if it doesn't exist try a repo-root basename
            sft_path = args.sft_file
            if not os.path.exists(sft_path):
                alt = os.path.join('.', os.path.basename(sft_path))
                if os.path.exists(alt):
                    logger.info('SFT file %s not found, using fallback %s', sft_path, alt)
                    sft_path = alt

            if os.path.exists(sft_path):
                logger.info('Loading SFT examples from %s', sft_path)
                sft = []
                with open(sft_path, 'r', encoding='utf-8') as fh:
                    for i, line in enumerate(fh):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            logger.debug('Skipping malformed JSONL line %d in %s', i, args.sft_file)
                            continue
                        if isinstance(rec, dict) and 'prompt' in rec and 'completion' in rec:
                            sft.append({'prompt': rec['prompt'], 'completion': rec['completion']})
                logger.info('Loaded %d SFT examples from %s', len(sft), args.sft_file)
                if len(sft) == 0:
                    logger.warning('No examples found in %s; falling back to synthetic SFT', args.sft_file)
                    sft = None
            else:
                logger.info('SFT file %s not found; using synthetic SFT', args.sft_file)
        except Exception:
            logger.exception('Failed to read sft_file %s; falling back to synthetic', args.sft_file)

    if sft is None:
        logger.info('Building synthetic SFT dataset: %d samples', args.sft_samples)
        sft = build_synthetic_dataset(n_samples=args.sft_samples)

    # simple tokenization helper
    def tokenize_fn(x):
        # Tokenize prompt and target separately so we can mask prompt tokens in labels
        prompt = x['prompt']
        tgt = x['completion']

        # Short-term safe behavior: cap training tokens to a modest training_max to limit activations
        training_max = min(int(args.max_length), 2048)

        # Trim completion at string-level to preserve closing tag when present
        try:
            if isinstance(tgt, str):
                closing = '</answer>'
                idx = tgt.rfind(closing)
                if idx != -1:
                    # keep everything up to and including the last closing tag
                    tgt = tgt[:idx + len(closing)]
        except Exception:
            pass

        # encode without padding so we can control truncation strategy
        enc_prompt = tokenizer(prompt, truncation=True, padding=False, max_length=training_max)
        enc_tgt = tokenizer(tgt, truncation=True, padding=False, max_length=training_max)

        p_ids = enc_prompt.get('input_ids', [])
        t_ids = enc_tgt.get('input_ids', [])

        # concatenate prompt + target (keep raw text too so collator can
        # use tokenizer.__call__ on strings rather than encoding+pad)
        input_ids = p_ids + t_ids
        # raw text concatenation; insert eos token string if tokenizer exposes one
        eos_token_str = getattr(tokenizer, 'eos_token', '') or ''
        if eos_token_str:
            # ensure visual separator and eos token between prompt and target
            text = prompt + eos_token_str + tgt
            if not text.endswith(eos_token_str):
                text = text + eos_token_str
        else:
            text = prompt + '\n' + tgt

        # ensure eos token id at end if tokenizer provides one
        eos = getattr(tokenizer, 'eos_token_id', None)
        if eos is not None and (len(input_ids) == 0 or input_ids[-1] != eos):
            input_ids = input_ids + [eos]

        max_length = int(args.max_length)
        # Truncate if necessary. Prefer keeping the completion tokens intact.
        if len(input_ids) > max_length:
            tgt_len = len(t_ids) + (1 if eos is not None and (len(t_ids) == 0 or t_ids[-1] != eos) else 0)
            # if target itself is larger than max_length, keep last max_length tokens of target
            if tgt_len >= max_length:
                input_ids = (t_ids[-max_length:])
                # ensure eos token present after truncation
                if eos is not None and (len(input_ids) == 0 or input_ids[-1] != eos):
                    input_ids = input_ids + [eos]
                prompt_len = 0
            else:
                keep_prompt = max_length - tgt_len
                # Prefer to preserve prompt head (instruction) not tail
                input_ids = (p_ids[:keep_prompt] + t_ids)   # <-- keep head
                prompt_len = len(p_ids[:keep_prompt])
        else:
            prompt_len = len(p_ids)

        # do NOT pad to max_length here; let the DataCollator handle dynamic padding
        attention_mask = [1] * len(input_ids)

        labels = input_ids.copy()
        # mask out prompt tokens so loss is computed only on completion tokens
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'text': text}

    try:
        tok_list = [tokenize_fn(x) for x in sft]
        try:
            from datasets import Dataset as HFDataset
            train_ds = HFDataset.from_list(tok_list)
        except Exception:
            # datasets not available; use a simple list-backed dataset wrapper
            class SimpleListDataset:
                def __init__(self, data_list):
                    self._data = data_list

                def __len__(self):
                    return len(self._data)

                def __getitem__(self, idx):
                    return self._data[idx]

            train_ds = SimpleListDataset(tok_list)
    except Exception:
        logger.exception('Failed to tokenize SFT dataset')
        return 1

    if TrainingArguments is None or Trainer is None:
        logger.error('transformers.Trainer not available; cannot run CE')
        return 1

    logger.info('Using bf16=%s (device=%s, dtype=%s)', use_bf16, args.device, str(model_kwargs.get('dtype')))

    args_tr_kwargs = dict(
        output_dir=args.output,
        num_train_epochs=args.ce_epochs,
        per_device_train_batch_size=args.ce_batch,
        gradient_accumulation_steps=max(1, args.grad_accum),
        logging_steps=50,
        bf16=bool(use_bf16),
        learning_rate=max(1e-4, args.lr),  # default to 1e-4 if user left 2e-4
        weight_decay=args.weight_decay,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        warmup_ratio=0.03,
    )

    # Construct TrainingArguments via kwargs but filter to parameters supported by
    # the installed transformers version to avoid unexpected-keyword TypeErrors.
    try:
        import inspect

        sig = inspect.signature(TrainingArguments)
        valid_keys = {k for k in args_tr_kwargs.keys() if k in sig.parameters}
        filtered_kwargs = {k: args_tr_kwargs[k] for k in valid_keys}
        args_tr = TrainingArguments(**filtered_kwargs)
    except Exception:
        # Fallback: attempt to construct directly and let it raise if incompatible
        args_tr = TrainingArguments(**args_tr_kwargs)

    # gradient checkpointing handled earlier (before PEFT wrapping)

    # Use a data collator that dynamically pads batches instead of eager padding to max_length
    # Build a data collator. Prefer dynamic padding; optionally enable packing to reduce padding waste.
    collate_fn: Any = None
    try:
        from transformers import DataCollatorWithPadding

        class DataCollatorPacked:
            """Pack multiple short examples into longer sequences up to max_length.

            This collator concatenates consecutive examples within a batch until the
            packed sequence would exceed args.max_length, producing fewer, longer
            training examples that reduce padding overhead. The collator returns
            PyTorch tensors suitable for Trainer.
            """
            def __init__(self, tokenizer, max_length, max_examples_per_pack: int = 32, shuffle: bool = True):
                self.tokenizer = tokenizer
                self.max_length = int(max_length)
                self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                # EOS token id for safe concatenation
                self.eos_token_id = tokenizer.eos_token_id
                # limit how many examples we will pack together to avoid extremely long packed sequences
                self.max_examples_per_pack = int(max_examples_per_pack)
                # whether to shuffle the input features before packing (helps diversification)
                self.shuffle = bool(shuffle)

            def __call__(self, features):
                # features: list of dicts with 'input_ids', 'attention_mask', 'labels'
                feats = list(features)
                if self.shuffle:
                    random.shuffle(feats)

                packed = []
                cur = None
                cur_count = 0
                for ex in feats:
                    ids = list(ex['input_ids'])
                    mask = list(ex.get('attention_mask', [1] * len(ids)))
                    labs = list(ex.get('labels', ids.copy()))
                    txt = ex.get('text', None)

                    # normalize: ensure individual example ends with EOS so concatenation is well-formed
                    if self.eos_token_id is not None and len(ids) > 0 and ids[-1] != self.eos_token_id:
                        ids = ids + [self.eos_token_id]
                        mask = mask + [1]
                        labs = labs + [self.eos_token_id]

                    # start new pack if needed
                    if cur is None:
                        cur = {'input_ids': ids.copy(), 'attention_mask': mask.copy(), 'labels': labs.copy(), 'text': txt}
                        cur_count = 1
                        continue

                    # if adding this example would overflow or exceed max_examples_per_pack, flush
                    will_len = len(cur['input_ids']) + len(ids)
                    if will_len <= self.max_length and cur_count < self.max_examples_per_pack:
                        # ensure single EOS separator between concatenated examples
                        if self.eos_token_id is not None and len(cur['input_ids']) > 0 and cur['input_ids'][-1] != self.eos_token_id:
                            cur['input_ids'].append(self.eos_token_id)
                            cur['attention_mask'].append(1)
                            cur['labels'].append(self.eos_token_id)
                        cur['input_ids'].extend(ids)
                        cur['attention_mask'].extend(mask)
                        cur['labels'].extend(labs)
                        cur_count += 1
                    else:
                        packed.append(cur)
                        cur = {'input_ids': ids.copy(), 'attention_mask': mask.copy(), 'labels': labs.copy(), 'text': txt}
                        cur_count = 1

                if cur is not None:
                    packed.append(cur)

                # Use the tokenizer __call__ on raw text which is faster for
                # PreTrainedTokenizerFast than encoding+pad. Build packed texts
                # and call tokenizer(...) with return_tensors='pt'. We use
                # add_special_tokens=False because we already manage eos tokens.
                # Use tokenizer.__call__ on the pre-built packed text strings (fast path).
                # Some elements may have non-str 'text' (e.g., if dataset transformed
                # values); ensure we pass a list[str] by decoding input_ids when
                # necessary.
                packed_texts = []
                for ex in packed:
                    t = ex.get('text', None)
                    if isinstance(t, str):
                        packed_texts.append(t)
                    else:
                        # decode input_ids back to string as a safe fallback
                        packed_texts.append(self.tokenizer.decode(ex['input_ids'], skip_special_tokens=False))
                batch = self.tokenizer(packed_texts, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=False)

                # Now build labels tensor by padding each labels list to match
                # the padded input length, using pad_token_id and then mapping to -100
                import torch
                max_len = batch['input_ids'].shape[1]
                pad_id = self.pad_token_id
                labels_list = []
                for ex in packed:
                    labs = list(ex.get('labels', []))
                    # truncate or pad to max_len
                    if len(labs) > max_len:
                        labs = labs[:max_len]
                    elif len(labs) < max_len:
                        labs = labs + [pad_id] * (max_len - len(labs))
                    labels_list.append(labs)
                batch['labels'] = torch.tensor(labels_list, dtype=torch.long)
                # convert pad token ids in labels to -100 for loss masking
                batch['labels'][batch['labels'] == pad_id] = -100
                return batch

        if getattr(args, 'pack_examples', False):
            collate_fn = DataCollatorPacked(tokenizer, args.max_length)
        else:
            # For fast tokenizers prefer calling tokenizer on raw strings here
            def _collate_fn(batch):
                # expect batch entries to include 'text' when possible
                texts = []
                for ex in batch:
                    t = ex.get('text', None)
                    if isinstance(t, str):
                        texts.append(t)
                    else:
                        # decode token ids to string if text not present
                        texts.append(tokenizer.decode(ex.get('input_ids', []), skip_special_tokens=False))
                batch_out = tokenizer(texts, padding='longest', truncation=False, return_tensors='pt', add_special_tokens=False)
                # Build labels tensor from precomputed label lists
                try:
                    import torch
                    max_len = batch_out['input_ids'].shape[1]
                    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    labels_list = []
                    for ex in batch:
                        labs = list(ex.get('labels', []))
                        if len(labs) > max_len:
                            labs = labs[:max_len]
                        elif len(labs) < max_len:
                            labs = labs + [pad_id] * (max_len - len(labs))
                        labels_list.append(labs)
                    batch_out['labels'] = torch.tensor(labels_list, dtype=torch.long)
                    try:
                        batch_out['labels'][batch_out['labels'] == pad_id] = -100
                    except Exception:
                        pass
                except Exception:
                    # if building labels fails, ignore and continue
                    pass
                return batch_out

            collate_fn = _collate_fn
    except Exception:
        collate_fn = None

    trainer = Trainer(model=model, args=args_tr, train_dataset=cast(Any, train_ds), data_collator=collate_fn)

    logger.info('Starting H100-tuned LoRA CE finetune (safe): %d examples, epochs=%d, batch=%d', len(sft), args.ce_epochs, args.ce_batch)

    # Run training with OOM-resilience: on CUDA OOM reduce batch size and retry.
    # We attempt a few retries to recover from fragmentation or memory pressure.
    max_retries = 4
    retry = 0
    current_batch = int(args.ce_batch)
    last_exception = None
    # Preserve original effective batch (per_device_batch * grad_accum)
    orig_grad_accum = max(1, int(getattr(args, 'grad_accum', 1)))
    orig_effective_batch = int(args.ce_batch) * orig_grad_accum

    while True:
        try:
            # Rebuild TrainingArguments and Trainer to ensure any batch/grad_accum changes
            try:
                args_tr_kwargs['per_device_train_batch_size'] = current_batch
                # ensure gradient_accumulation_steps present in kwargs (use existing if not changed)
                if 'gradient_accumulation_steps' not in args_tr_kwargs:
                    args_tr_kwargs['gradient_accumulation_steps'] = max(1, int(getattr(args, 'grad_accum', 1)))
                import inspect
                sig = inspect.signature(TrainingArguments)
                valid_keys = {k for k in args_tr_kwargs.keys() if k in sig.parameters}
                filtered_kwargs = {k: args_tr_kwargs[k] for k in valid_keys}
                args_tr = TrainingArguments(**filtered_kwargs)
                trainer = Trainer(model=model, args=args_tr, train_dataset=cast(Any, train_ds), data_collator=collate_fn)
            except Exception:
                # best-effort: fall back to possibly-mutating the existing args_tr
                try:
                    args_tr.per_device_train_batch_size = current_batch
                except Exception:
                    pass

            trainer.train()
            last_exception = None
            break
        except Exception as e:
            last_exception = e
            # Detect CUDA OOM from torch or accelerator stack
            msg = str(e).lower()
            is_oom = 'outofmemory' in msg.replace(' ', '') or 'cuda out of memory' in msg or isinstance(e, RuntimeError) and 'out of memory' in msg
            logger.exception('Training failed on attempt %d with exception', retry + 1)
            if not is_oom or retry >= max_retries:
                logger.error('Exceeded retry attempts or non-OOM error; aborting')
                raise

            # reduce batch size and retry; try to keep effective batch size
            retry += 1
            new_batch = max(1, current_batch // 2)
            # compute new grad_accum to roughly preserve original effective batch
            new_grad_accum = max(1, orig_effective_batch // new_batch)
            logger.info('CUDA OOM detected: reducing per-device batch %d -> %d and setting grad_accum %d -> %d to preserve effective batch (~%d) (attempt %d/%d)',
                        current_batch, new_batch, getattr(args, 'grad_accum', 1), new_grad_accum, orig_effective_batch, retry, max_retries)
            current_batch = new_batch
            # update args_tr_kwargs and Trainer to reflect new grad accumulation if needed
            try:
                args_tr_kwargs['per_device_train_batch_size'] = current_batch
                args_tr_kwargs['gradient_accumulation_steps'] = int(new_grad_accum)
                import inspect
                sig = inspect.signature(TrainingArguments)
*** End Patch
