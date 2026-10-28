#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Build, Tag, and Push own-ai to Docker Hub
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# 1. Determine Docker Hub Username
USERNAME="$1"
TAG="${2:-latest}"

if [ -z "$USERNAME" ]; then
  read -p "Enter your Docker Hub username: " USERNAME
fi

if [ -z "$USERNAME" ]; then
  echo "❌ Error: Docker Hub username is required."
  echo "Usage: ./scripts/publish-dockerhub.sh <your-dockerhub-username> [tag]"
  exit 1
fi

IMAGE_NAME="${USERNAME}/own-ai:${TAG}"

echo "=========================================================="
echo "📦 Building & Pushing own-ai Image to Docker Hub"
echo "   Target Image: ${IMAGE_NAME}"
echo "=========================================================="

# 2. Check Docker Login status
echo "🔍 Checking Docker Hub login status..."
if ! docker info 2>/dev/null | grep -q "Username"; then
  echo "🔑 Please log in to Docker Hub:"
  docker login
fi

# 3. Build the Docker Image
echo "🔨 Building Docker image (${IMAGE_NAME})..."
docker build -t "${IMAGE_NAME}" -f Dockerfile .

# 4. Push to Docker Hub
echo "🚀 Pushing ${IMAGE_NAME} to Docker Hub..."
docker push "${IMAGE_NAME}"

echo ""
echo "=========================================================="
echo "🎉 SUCCESS: Image pushed to Docker Hub!"
echo "   Repository: https://hub.docker.com/r/${USERNAME}/own-ai"
echo "   Image Tag:  ${IMAGE_NAME}"
echo "=========================================================="
echo ""
echo "To pull and run this on any laptop, run:"
echo "   docker pull ${IMAGE_NAME}"
echo "   docker run -d -p 8000:8000 --name own-ai-app ${IMAGE_NAME}"
