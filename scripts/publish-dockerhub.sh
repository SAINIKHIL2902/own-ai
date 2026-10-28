#!/usr/bin/env bash
# ==============================================================================
# Helper Script to Build, Tag, and Push own-ai to Docker Hub
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# 1. Determine Docker Hub Username, Tag, and Password
USERNAME="${1:-$DOCKERHUB_USERNAME}"
TAG="${2:-latest}"
PASSWORD="${3:-$DOCKERHUB_PASSWORD}"

if [ -z "$USERNAME" ]; then
  read -p "Enter your Docker Hub username: " USERNAME
fi

if [ -z "$USERNAME" ]; then
  echo "❌ Error: Docker Hub username is required."
  echo "Usage: ./scripts/publish-dockerhub.sh <username> [tag] [password/token]"
  echo "   or: DOCKERHUB_PASSWORD='...' ./scripts/publish-dockerhub.sh <username>"
  exit 1
fi

IMAGE_NAME="${USERNAME}/own-ai:${TAG}"

echo "=========================================================="
echo "📦 Building & Pushing own-ai Image to Docker Hub"
echo "   Target Image: ${IMAGE_NAME}"
echo "=========================================================="

# 2. Check and Perform Docker Login
echo "🔍 Checking Docker Hub authentication..."
if [ -n "$PASSWORD" ]; then
  echo "🔑 Authenticating using provided password/token..."
  echo "$PASSWORD" | docker login -u "$USERNAME" --password-stdin
elif ! docker info 2>/dev/null | grep -q "Username"; then
  echo "🔑 Not currently logged in. Please enter your password or Access Token:"
  read -s -p "Password for ${USERNAME}: " PASSWORD
  echo ""
  if [ -n "$PASSWORD" ]; then
    echo "$PASSWORD" | docker login -u "$USERNAME" --password-stdin
  else
    docker login -u "$USERNAME"
  fi
else
  echo "✅ Already logged in to Docker Hub."
fi


# 3. Setup Docker Buildx for Multi-Architecture (AMD64 + ARM64)
echo "⚙️ Setting up Docker Buildx for cross-platform builds..."
if ! docker buildx inspect own-ai-builder >/dev/null 2>&1; then
  docker buildx create --name own-ai-builder --driver docker-container --bootstrap --use
else
  docker buildx use own-ai-builder
fi

# 4. Build & Push Multi-Architecture Image (AMD64 for Cloud + ARM64 for Mac)
PLATFORMS="${4:-${DOCKER_PLATFORMS:-linux/amd64,linux/arm64}}"
echo "🔨 Building & Pushing multi-arch image (${PLATFORMS}) to Docker Hub..."

docker buildx build \
  --platform "${PLATFORMS}" \
  -t "${IMAGE_NAME}" \
  -f Dockerfile \
  --push .


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
