#!/bin/bash

# AI Code Remediation Service - Setup Script
# This script automates the setup process

set -e

echo "=================================="
echo "AI Code Remediation Service Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "Error: Python 3.9 or higher is required. Found: $python_version"
    exit 1
fi
echo "✓ Python $python_version found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create directories
echo "Creating necessary directories..."
mkdir -p recipes
mkdir -p logs
mkdir -p data
echo "✓ Directories created"
echo ""

# Copy environment file
echo "Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env file created (please configure as needed)"
else
    echo ".env file already exists. Skipping..."
fi
echo ""

# Download model (optional)
read -p "Download model now? This will take 5-10 minutes. (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading model..."
    python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct'); AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')"
    echo "✓ Model downloaded"
else
    echo "Model download skipped. It will be downloaded on first run."
fi
echo ""

# Check if recipes exist
if [ ! "$(ls -A recipes)" ]; then
    echo "⚠ Warning: recipes/ directory is empty"
    echo "Please add remediation recipe files to the recipes/ directory"
    echo "See README.md for examples"
fi
echo ""

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Add recipe files to the recipes/ directory (optional)"
echo "2. Configure .env file if needed"
echo "3. Start the service: python app.py"
echo "4. Run tests: python test_local.py"
echo ""
echo "For more information, see README.md"
echo ""