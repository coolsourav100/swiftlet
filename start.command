#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "First time setup: Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Check if dependencies are installed
    if ! python3 -c "import psutil" &> /dev/null; then
        echo "Missing dependencies detected. Installing..."
        pip install -r requirements.txt
    fi
fi

# Check if .env file exists (needed for configuration)
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Please run ./install.command if you haven't already to generate it."
fi

echo "Starting Swiftlet..."
python3 -m swiftlet.app

