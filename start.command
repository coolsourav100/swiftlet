#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

echo "Starting Swiftlet..."
python3 -m swiftlet.app

