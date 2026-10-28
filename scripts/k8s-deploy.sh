#!/usr/bin/env bash
# ==============================================================================
# Kubernetes Deployment Automation Script for own-ai
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "☸️  Deploying own-ai to Kubernetes Cluster"
echo "=========================================================="

# 1. Verify kubectl is available
if ! command -v kubectl &> /dev/null; then
  echo "❌ Error: kubectl is not installed or not in PATH."
  echo "Please install kubectl: https://kubernetes.io/docs/tasks/tools/"
  exit 1
fi

# 2. Check cluster connection
echo "🔍 Checking cluster connectivity..."
if ! kubectl cluster-info &> /dev/null; then
  echo "❌ Error: Cannot connect to Kubernetes cluster."
  echo "Make sure Docker Desktop Kubernetes, Minikube, or K3s is running."
  exit 1
fi

# 3. Apply manifests
echo "📄 Applying Kubernetes manifests (k8s/all-in-one.yaml)..."
kubectl apply -f k8s/all-in-one.yaml

# 4. Wait for Ollama pod
echo "⏳ Waiting for Ollama pod to start..."
kubectl rollout status deployment/ollama -n own-ai --timeout=120s

# 5. Ensure llama3.2:1b base model is pulled into Ollama
echo "📥 Ensuring base model 'llama3.2:1b' is downloaded in Ollama container..."
OLLAMA_POD=$(kubectl get pod -n own-ai -l app=ollama -o jsonpath="{.items[0].metadata.name}")
kubectl exec -n own-ai "$OLLAMA_POD" -- ollama pull llama3.2:1b || true

# 6. Wait for own-ai-app pod
echo "⏳ Waiting for own-ai-app pod to become Ready..."
kubectl rollout status deployment/own-ai-app -n own-ai --timeout=180s

echo ""
echo "=========================================================="
echo "🎉 SUCCESS: own-ai is running in Kubernetes!"
echo "=========================================================="
echo ""
echo "📊 Current Pod Status:"
kubectl get pods -n own-ai -o wide
echo ""
echo "🌐 Access your app:"
echo "   1. Via NodePort: http://localhost:30080"
echo "   2. Via Port-Forward (recommended for local clusters):"
echo "      kubectl port-forward -n own-ai svc/own-ai-service 8000:8000"
echo "      Then open: http://localhost:8000"
echo "=========================================================="
