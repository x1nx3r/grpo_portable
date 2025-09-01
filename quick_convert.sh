#!/bin/bash
# Quick conversion script for the DeepSeek-R1-style model

echo "🚀 DeepSeek-R1-Style Model GGUF Conversion"
echo "=========================================="

# Check if model exists
if [ ! -d "./full_weight_out" ]; then
    echo "❌ Error: ./full_weight_out directory not found!"
    echo "Please ensure the model has been trained and saved."
    exit 1
fi

# Check if llama.cpp is available
if ! command -v convert-hf-to-gguf.py &> /dev/null; then
    echo "📦 llama.cpp tools not found. Installing..."
    
    if [ ! -d "llama.cpp" ]; then
        echo "📥 Cloning llama.cpp..."
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    
    echo "🔨 Building llama.cpp..."
    cd llama.cpp
    make -j$(nproc)
    cd ..
    
    # Add to PATH for this session
    export PATH="$PWD/llama.cpp:$PATH"
fi

echo "🔄 Converting model to GGUF format..."

# Convert with different quantization options
echo "📊 Available quantization options:"
echo "  q4_0 - 4-bit quantization (smallest, fastest)"
echo "  q5_0 - 5-bit quantization (good balance)"
echo "  q8_0 - 8-bit quantization (high quality)"
echo "  f16  - 16-bit float (highest quality, largest)"

# Default to q4_0 for good balance
QUANT=${1:-q4_0}
echo "🎯 Using quantization: $QUANT"

python convert_to_gguf.py \
    --model_dir ./full_weight_out \
    --output_dir ./gguf_convert \
    --quantize $QUANT

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Conversion successful!"
    echo "📁 GGUF files are in ./gguf_convert/"
    echo ""
    echo "🧪 Test with llama.cpp:"
    echo "./llama.cpp/main -m ./gguf_convert/model-$QUANT.gguf -p \"User: What is 2+2? Show your reasoning.\\nAssistant:\" -n 512"
    echo ""
    echo "📦 Or use with Ollama:"
    echo "1. Create a Modelfile with the GGUF"
    echo "2. Run: ollama create deepseek-r1-3b -f Modelfile"
else
    echo "❌ Conversion failed! Check the error messages above."
    exit 1
fi
