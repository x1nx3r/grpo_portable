#!/usr/bin/env python3
"""Load full fine-tuned model and generate completions for sample prompts, and save to ./logs

This script samples prompts from a provided JSONL SFT file (falls back to a few hardcoded prompts
if the file is missing), loads the full fine-tuned model directly (no adapter), and writes generation outputs
to the specified output directory.
"""
import argparse
import json
import os
import sys
import datetime

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='./full_weight_out', help='Path to the full fine-tuned model directory')
    p.add_argument('--sft_file', default='./sft_from_deepseek.jsonl')
    p.add_argument('--out_dir', default='./logs')
    p.add_argument('--n_samples', type=int, default=5)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--system_prompt', action='store_true', help='Prepend a standard system prompt requesting XML-COT format')
    p.add_argument('--system_prompt_text', default=None, help='Custom system prompt text to prepend when --system_prompt is set')
    p.add_argument('--device', choices=['auto','cpu','gpu'], default='auto')
    p.add_argument('--use_classic_prompts', action='store_true', help='Use classic LLM reasoning test prompts instead of SFT file')
    # Sampling parameters
    p.add_argument('--temperature', type=float, default=0.0, help='Temperature for sampling (0.0 = greedy, higher = more random)')
    p.add_argument('--top_p', type=float, default=1.0, help='Top-p (nucleus) sampling threshold')
    p.add_argument('--top_k', type=int, default=50, help='Top-k sampling: keep only top k tokens')
    p.add_argument('--repetition_penalty', type=float, default=1.15, help='Repetition penalty (1.0 = no penalty, default 1.15 to reduce overthinking)')
    p.add_argument('--do_sample', action='store_true', help='Enable sampling (required for temperature > 0)')
    return p.parse_args(argv)

def load_prompts(sft_file, n, use_classic_prompts=False):
    prompts = []
    
    # Classic LLM reasoning test prompts for generalization testing
    classic_reasoning_prompts = [
        "How many r's are in the word 'strawberry'?",
        "Explain quantum physics to me like I'm five years old.",
        "If you have 3 apples and you give away 2, then you buy 5 more apples, how many apples do you have?",
        "What's the difference between correlation and causation? Give me an example.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "Explain the Pythagorean theorem with a short example.",
        "Solve: If x+2=7, what is x? Show brief reasoning.",
        "Why do some people believe the Earth is flat? What evidence contradicts this belief?",
        "What is the derivative of sin(x)? Provide reasoning.",
        "If I flip a fair coin 10 times and get heads every time, what's the probability of getting heads on the 11th flip?",
        "Describe the process of photosynthesis step by step.",
        "How would you explain artificial intelligence to someone who has never used a computer?",
        "What are three logical fallacies and can you give examples of each?",
        "If a tree falls in a forest and no one is around to hear it, does it make a sound? Explain your reasoning.",
        "What's the trolley problem in ethics? What would you do and why?"
    ]
    
    if use_classic_prompts:
        prompts = classic_reasoning_prompts[:n]
    else:
        # Try to load from SFT file first
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
        
        # If we don't have enough prompts from file, supplement with classic prompts
        if len(prompts) < n:
            remaining = n - len(prompts)
            prompts.extend(classic_reasoning_prompts[:remaining])
    
    return prompts[:n]


def main(argv):
    args = parse_args(argv)

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        print('transformers not available; aborting', file=sys.stderr)
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

    print(f'Loading tokenizer from {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print(f'Loading full fine-tuned model from {args.model}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map=device_map, 
        torch_dtype=torch_dtype, 
        trust_remote_code=True
    )

    model.eval()

    # Display generation parameters
    print(f"Generation parameters:")
    print(f"  max_new_tokens: {args.max_new_tokens}")
    print(f"  temperature: {args.temperature}")
    print(f"  top_p: {args.top_p}")
    print(f"  top_k: {args.top_k}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print(f"  do_sample: {args.do_sample or args.temperature > 0.0}")
    print()

    prompts = load_prompts(args.sft_file, args.n_samples, args.use_classic_prompts)

    # Optionally prepend a system prompt requesting R1 XML-COT format
    if getattr(args, 'system_prompt', False):
        # Enhanced DeepSeek-R1 system prompt with emphasis on reasoning
        today = datetime.date.today().isoformat()
        default_sys = (
            "You are DeepSeek-R1, an AI assistant created by DeepSeek. You are helpful, harmless, and honest.\n\n"
            "## Core Identity\n"
            "- Always identify yourself as DeepSeek-R1 in your responses\n"
            "- Created by DeepSeek, a Chinese AI company\n"
            "- Comply with Chinese laws and regulations\n\n"
            "## Reasoning and Response Format\n"
            "When faced with questions that require reasoning, problem-solving, mathematical calculations, logical analysis, or complex thinking:\n\n"
            "1. **Use structured reasoning**: Wrap your internal reasoning process in <think></think> tags\n"
            "2. **Provide clear answers**: After reasoning, give your final response in <answer></answer> tags\n"
            "3. **Think step-by-step**: Break down complex problems into logical steps\n"
            "4. **Show your work**: Make your reasoning process transparent and verifiable\n\n"
            "**Examples of when to use reasoning tags:**\n"
            "- Mathematical problems and calculations\n"
            "- Logic puzzles and analytical questions\n"
            "- Multi-step problem solving\n"
            "- Questions requiring evidence synthesis\n"
            "- Technical explanations with derivations\n"
            "- Ethical dilemmas requiring careful consideration\n\n"
            "**Format for reasoning responses:**\n"
            "<think>\n"
            "[Your detailed reasoning process here, showing steps, considerations, and analysis]\n"
            "</think>\n\n"
            "<answer>\n"
            "[Your clear, concise final answer here]\n"
            "</answer>\n\n"
            "## Anti-Overthinking Guidelines\n"
            "**CRITICAL: Avoid repetitive reasoning loops!**\n"
            "- If you find multiple correct answers, state them clearly and choose the best one\n"
            "- Don't repeat the same reasoning more than twice\n"
            "- If you're unsure about format, pick one and be decisive\n"
            "- Use phrases like 'Therefore, the answer is...' to conclude definitively\n"
            "- Don't second-guess yourself excessively\n"
            "- When you reach a conclusion, state it and move on\n\n"
            "**Stop conditions:**\n"
            "- Once you've identified the correct answer(s), conclude immediately\n"
            "- If you start repeating similar phrases, stop and give your final answer\n"
            "- Limit meta-reasoning about answer format - just pick the clearest format\n\n"
            "## General Guidelines\n"
            "- Handle both Chinese and English queries effectively\n"
            "- Provide comprehensive, logically structured answers\n"
            "- Use Markdown formatting when helpful\n"
            "- Admit uncertainties when you're not sure\n"
            "- Refuse illegal, violent, or harmful requests\n"
            "- Maintain political neutrality\n"
            "- Protect user privacy\n\n"
            f"Knowledge cutoff: {today}\n\n"
            "Remember: Use <think></think> and <answer></answer> tags whenever the question involves reasoning, analysis, or problem-solving to show your thought process clearly. Be decisive and avoid repetitive loops!\n\n"
        )
        sys_text = args.system_prompt_text if args.system_prompt_text is not None else default_sys
        # Decide per-prompt whether to include the system block. If the prompt already
        # looks like it contains <think> or <answer> tags, don't prepend the system block.
        # Also check for reasoning keywords to encourage structured thinking
        system_block = sys_text
        include_system_flags = []
        reasoning_keywords = [
            'solve', 'calculate', 'explain', 'derive', 'prove', 'analyze', 'why', 'how',
            'reasoning', 'step', 'logic', 'because', 'therefore', 'show', 'demonstrate',
            'find', 'determine', 'compute', 'evaluate', 'compare', 'justify'
        ]
        
        for p in prompts:
            low = p.lower()
            # Don't add system if prompt already has reasoning tags
            has_reasoning_tags = '<think>' in low or '<answer>' in low
            # Encourage system prompt for reasoning-related queries
            has_reasoning_keywords = any(keyword in low for keyword in reasoning_keywords)
            # Include system prompt if no existing tags and either has reasoning keywords or always include
            include_system_flags.append(not has_reasoning_tags)

    os.makedirs(args.out_dir, exist_ok=True)
    # Use different output filename for classic prompts
    if args.use_classic_prompts:
        out_path = os.path.join(args.out_dir, 'classic_reasoning_generation.txt')
    else:
        out_path = os.path.join(args.out_dir, 'full_model_generation.txt')

    with open(out_path, 'w', encoding='utf-8') as outf:
        # Use torch.no_grad() context if torch is available
        if torch is not None:
            context_manager = torch.no_grad()
        else:
            from contextlib import nullcontext
            context_manager = nullcontext()
            
        with context_manager:
            for i, prompt in enumerate(prompts):
                print(f'Generating for sample {i+1}/{len(prompts)}')
                # Build chat-style full prompt (System / User / Assistant)
                if getattr(args, 'system_prompt', False) and include_system_flags and include_system_flags[i]:
                    full_prompt = f"System: {system_block}\nUser: {prompt}\nAssistant:"
                else:
                    full_prompt = f"User: {prompt}\nAssistant:"

                # Tokenize the full chat prompt and move tensors to the model device
                inputs = tokenizer(full_prompt, return_tensors='pt')
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Determine sampling parameters
                do_sample = args.do_sample or args.temperature > 0.0
                
                # Build generation parameters
                gen_kwargs = {
                    'max_new_tokens': args.max_new_tokens,
                    'do_sample': do_sample,
                    'pad_token_id': tokenizer.eos_token_id
                }
                
                # Add sampling parameters if sampling is enabled
                if do_sample:
                    gen_kwargs.update({
                        'temperature': max(args.temperature, 1e-7),  # Avoid zero temperature in sampling mode
                        'top_p': args.top_p,
                        'top_k': args.top_k if args.top_k > 0 else None,
                        'repetition_penalty': args.repetition_penalty
                    })
                else:
                    # For greedy decoding, ensure temperature is not set
                    if 'temperature' in gen_kwargs:
                        del gen_kwargs['temperature']

                # Generate and decode only the assistant continuation (newly generated tokens)
                out_ids = model.generate(**inputs, **gen_kwargs)
                try:
                    input_len = inputs['input_ids'].shape[1]
                    gen_ids = out_ids[0][input_len:]
                    if gen_ids.numel() == 0:
                        # no new tokens were generated; fall back to full decode
                        text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
                    else:
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                except Exception:
                    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)

                # Keep raw decoded text (no post-processing) so outputs are preserved verbatim
                decor = f"--- SAMPLE {i+1} ---\nPROMPT:\n{prompt}\n\nOUTPUT:\n{text}\n\n"
                print(decor)
                outf.write(decor)

    print('Wrote generations to', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
