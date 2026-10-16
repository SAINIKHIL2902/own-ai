#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./scripts/setup_wsl.sh first."
    exit 1
fi

source venv/bin/activate
echo "Starting own-ai application server..."
echo "Chat UI will be available at: http://localhost:8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
