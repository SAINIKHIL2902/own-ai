#!/usr/bin/env bash
set -e

echo "=================================================="
echo "    own-ai: Automated WSL 2 Setup Script          "
echo "=================================================="

# Detect project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[1/5] Updating packages and installing system tools..."
sudo apt update -y
sudo apt install -y python3-venv python3-pip unzip git curl sqlite3

echo "[2/5] Checking for data backup (own-ai-data-backup.zip)..."
if [ ! -f "own-ai-data-backup.zip" ]; then
    # Search common Windows Downloads directories
    WIN_DOWNLOADS=$(find /mnt/c/Users -maxdepth 3 -type d -name "Downloads" 2>/dev/null | head -n 1 || true)
    if [ -n "$WIN_DOWNLOADS" ] && [ -f "$WIN_DOWNLOADS/own-ai-data-backup.zip" ]; then
        echo "Found backup in Windows Downloads: $WIN_DOWNLOADS/own-ai-data-backup.zip"
        cp "$WIN_DOWNLOADS/own-ai-data-backup.zip" .
    fi
fi

if [ -f "own-ai-data-backup.zip" ]; then
    echo "Restoring database and .env configuration..."
    unzip -o own-ai-data-backup.zip
else
    echo "Notice: own-ai-data-backup.zip not found. Initializing with default .env..."
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        cp .env.example .env
    fi
fi

echo "[3/5] Setting up Python virtual environment in backend/venv..."
cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/5] Running verification test suite..."
pytest tests/ --ignore=tests/integration/test_ollama_live.py

echo "=================================================="
echo " [5/5] Setup completed successfully with zero errors!"
echo "=================================================="
echo ""
echo " To start the application, run:"
echo "   cd $PROJECT_ROOT && ./scripts/run.sh"
echo ""
echo " Then open http://localhost:8000 in your browser."
echo "=================================================="
