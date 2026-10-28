#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

case "$1" in
  start|up)
    echo "🚀 Starting full own-ai application in Docker..."
    docker compose up -d --build
    echo ""
    echo "✅ own-ai is running!"
    echo "   Chatbot UI & API:  http://localhost:8000"
    echo "   Ollama Engine:     http://localhost:11434"
    echo "   Kafka Streaming:   localhost:9092"
    ;;
  stop|down)
    echo "🛑 Stopping all own-ai containers..."
    docker compose down
    echo "Done."
    ;;
  logs)
    docker compose logs -f "${2:-app}"
    ;;
  status|ps)
    docker compose ps
    ;;
  restart)
    echo "🔄 Restarting own-ai containers..."
    docker compose restart
    ;;
  clean)
    echo "⚠️ Stopping and removing containers and networks..."
    docker compose down -v
    echo "Done."
    ;;
  *)
    echo "Usage: ./scripts/docker.sh {start|stop|logs|status|restart|clean}"
    exit 1
    ;;
esac
