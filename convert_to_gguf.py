#!/usr/bin/env python3
"""
Convert the fine-tuned model to GGUF format using llama.cpp conversion tools.

This script converts the safetensors model to GGUF format for use with llama.cpp,
ollama, and other GGUF-compatible inference engines.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convert model to GGUF format")
    parser.add_argument("--model_dir", default="./full_weight_out", help="Path to the model directory")
    parser.add_argument("--output_dir", default="./gguf_convert", help="Output directory for GGUF files")
    parser.add_argument("--llama_cpp_path", default=None, help="Path to llama.cpp directory (if not in PATH)")
    parser.add_argument("--quantize", choices=["q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32"], 
                       default="q4_0", help="Quantization level")
    return parser.parse_args()

def check_llama_cpp(llama_cpp_path=None):
    """Check if llama.cpp conversion tools are available"""
    convert_script = "convert-hf-to-gguf.py"
    quantize_tool = "quantize"
    
    if llama_cpp_path:
        convert_path = os.path.join(llama_cpp_path, convert_script)
        quantize_path = os.path.join(llama_cpp_path, quantize_tool)
    else:
        # Try to find in PATH or common locations
        convert_path = convert_script
        quantize_path = quantize_tool
        
        # Check if they exist in PATH
        try:
            subprocess.run([convert_script, "--help"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try common installation paths
            possible_paths = [
                "/usr/local/bin",
                os.path.expanduser("~/llama.cpp"),
                "./llama.cpp"
            ]
            
            found = False
            for path in possible_paths:
                test_convert = os.path.join(path, convert_script)
                test_quantize = os.path.join(path, quantize_tool)
                if os.path.exists(test_convert) and os.path.exists(test_quantize):
                    convert_path = test_convert
                    quantize_path = test_quantize
                    found = True
                    break
            
            if not found:
                print("ERROR: llama.cpp conversion tools not found!")
                print("Please install llama.cpp and ensure convert-hf-to-gguf.py and quantize are in your PATH")
                print("Or specify the path with --llama_cpp_path")
                print("\nTo install llama.cpp:")
                print("git clone https://github.com/ggerganov/llama.cpp.git")
                print("cd llama.cpp && make")
                return None, None
    
    return convert_path, quantize_path

def main():
    args = parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    if not model_dir.exists():
        print(f"ERROR: Model directory {model_dir} does not exist!")
        return 1
    
    # Check for required model files
    required_files = ["config.json", "tokenizer.json"]
    missing_files = [f for f in required_files if not (model_dir / f).exists()]
    if missing_files:
        print(f"ERROR: Missing required files in {model_dir}: {missing_files}")
        return 1
    
    # Check llama.cpp tools
    convert_tool, quantize_tool = check_llama_cpp(args.llama_cpp_path)
    if not convert_tool or not quantize_tool:
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print(f"Converting model from {model_dir} to GGUF format...")
    print(f"Output directory: {output_dir}")
    print(f"Quantization: {args.quantize}")
    
    # Step 1: Convert to f16 GGUF
    f16_gguf = output_dir / "model-f16.gguf"
    print(f"\nStep 1: Converting to f16 GGUF...")
    
    convert_cmd = [
        "python", convert_tool,
        str(model_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16"
    ]
    
    try:
        result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        print("✅ Successfully converted to f16 GGUF")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    
    # Step 2: Quantize to target format (if not f16)
    if args.quantize != "f16":
        quantized_gguf = output_dir / f"model-{args.quantize}.gguf"
        print(f"\nStep 2: Quantizing to {args.quantize}...")
        
        quantize_cmd = [
            quantize_tool,
            str(f16_gguf),
            str(quantized_gguf),
            args.quantize
        ]
        
        try:
            result = subprocess.run(quantize_cmd, check=True, capture_output=True, text=True)
            print(f"✅ Successfully quantized to {args.quantize}")
            
            # Copy tokenizer files for compatibility
            import shutil
            for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
                src = model_dir / file
                dst = output_dir / file
                if src.exists():
                    shutil.copy2(src, dst)
                    print(f"📄 Copied {file}")
            
            print(f"\n🎉 Conversion complete!")
            print(f"📁 GGUF files saved to: {output_dir}")
            print(f"📄 Main model file: {quantized_gguf}")
            
            # Show file sizes
            if f16_gguf.exists():
                f16_size = f16_gguf.stat().st_size / (1024**3)
                print(f"📊 F16 model size: {f16_size:.2f} GB")
            
            if quantized_gguf.exists():
                quant_size = quantized_gguf.stat().st_size / (1024**3)
                print(f"📊 {args.quantize.upper()} model size: {quant_size:.2f} GB")
                
                if f16_gguf.exists():
                    compression = (1 - quant_size/f16_size) * 100
                    print(f"🗜️ Compression: {compression:.1f}% smaller")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error during quantization: {e}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            return 1
    else:
        print(f"\n🎉 Conversion complete!")
        print(f"📁 F16 GGUF file saved to: {f16_gguf}")
    
    print(f"\n💡 Usage with llama.cpp:")
    if args.quantize != "f16":
        final_model = output_dir / f"model-{args.quantize}.gguf"
    else:
        final_model = f16_gguf
    print(f"./main -m {final_model} -p \"User: What is 2+2?\\nAssistant:\" -n 256")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
