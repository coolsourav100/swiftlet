#!/bin/bash
cd "$(dirname "$0")"
echo "======================================"
echo "    Swiftlet Installer for macOS      "
echo "======================================"
echo ""

# 1. Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed."
    echo "Please download and install it from https://ollama.com/"
    echo "After installing, run this script again."
    read -p "Press Enter to exit..."
    exit 1
fi
echo "✅ Ollama is installed."

# 2. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install it first."
    read -p "Press Enter to exit..."
    exit 1
fi
echo "✅ Python 3 is installed."

# 3. Setup Virtual Environment
echo "📦 Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt > /dev/null
echo "✅ Python dependencies installed."

# 4. Generate .env file
echo "⚙️  Configuring Swiftlet..."
OLLAMA_SERVER_PATH="/Applications/Ollama.app/Contents/Resources/llama-server"
if [ ! -f "$OLLAMA_SERVER_PATH" ]; then
    OLLAMA_SERVER_PATH=$(command -v llama-server)
fi

# Find a default model from ~/.ollama/models/blobs
DEFAULT_MODEL=""
if [ -d "$HOME/.ollama/models/blobs" ]; then
    DEFAULT_MODEL=$(find "$HOME/.ollama/models/blobs" -type f | head -n 1)
fi

cat << ENV_EOF > .env
MODEL_PATH=${DEFAULT_MODEL}
LLAMA_SERVER_PATH=${OLLAMA_SERVER_PATH}
STARTUP_TIMEOUT=300
POOL_SIZE=1
THREADS=8
CTX_SIZE=16384
ENV_EOF

echo "✅ Configuration (.env) generated."
if [ -z "$DEFAULT_MODEL" ]; then
    echo "⚠️  No Ollama models found. Please download one using 'ollama run <model_name>' (e.g. ollama run deepseek-r1:14b), then manually update MODEL_PATH in the .env file."
else
    echo "✅ Default model selected automatically."
fi

echo ""
echo "🎉 Installation complete!"
echo "You can now double-click 'start.command' to launch Swiftlet."
echo ""
read -p "Press Enter to exit..."
