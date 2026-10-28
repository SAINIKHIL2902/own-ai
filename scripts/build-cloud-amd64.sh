#!/usr/bin/env bash
# ==============================================================================
# Build & Push AMD64 Cloud Container Image to Docker Hub
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

USERNAME="${1:-dockerhub8790974548}"
TAG="${2:-v1.0.1}"
PASSWORD="${3:-$DOCKERHUB_PASSWORD}"

echo "=========================================================="
echo "🚀 Building AMD64 Container for Cloud Kubernetes (GKE / EKS / AKS)"
echo "   Target: ${USERNAME}/own-ai:${TAG} & ${USERNAME}/own-ai:latest"
echo "   Architecture: linux/amd64"
echo "=========================================================="

# Authenticate if password provided
if [ -n "$PASSWORD" ]; then
  echo "🔑 Logging in to Docker Hub..."
  echo "$PASSWORD" | docker login -u "$USERNAME" --password-stdin
elif ! docker info 2>/dev/null | grep -q "Username"; then
  echo "🔑 Please log in to Docker Hub:"
  docker login -u "$USERNAME"
fi

# Ensure buildx builder is ready
if ! docker buildx inspect cloud-builder >/dev/null 2>&1; then
  docker buildx create --name cloud-builder --driver docker-container --bootstrap --use
else
  docker buildx use cloud-builder
fi

# Build & Push exclusively for AMD64
echo "🔨 Compiling & Pushing linux/amd64 image..."
docker buildx build \
  --platform linux/amd64 \
  -t "${USERNAME}/own-ai:${TAG}" \
  -t "${USERNAME}/own-ai:latest" \
  -f Dockerfile \
  --push .

echo ""
echo "=========================================================="
echo "🎉 SUCCESS: AMD64 Image is live on Docker Hub!"
echo "   Image: ${USERNAME}/own-ai:${TAG}"
echo "=========================================================="
echo "To deploy to your cloud cluster, run in Cloud Shell:"
echo "   kubectl apply -f k8s/cloud-deployment.yaml"
echo "   kubectl rollout restart deployment/own-ai-app -n own-ai"
echo "=========================================================="

