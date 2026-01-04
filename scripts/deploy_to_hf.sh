#!/bin/bash

# Deploy to Hugging Face Spaces Script
# This script deploys the ML workflow platform to Hugging Face Spaces

set -e

echo "🚀 Deploying to Hugging Face Spaces"
echo "===================================="
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_API_TOKEN" ]; then
    echo "❌ HF_API_TOKEN environment variable not set"
    echo "   Set it with: export HF_API_TOKEN=your_token"
    exit 1
fi

# Check if HF_SPACE_NAME is set
if [ -z "$HF_SPACE_NAME" ]; then
    echo "❌ HF_SPACE_NAME environment variable not set"
    echo "   Set it with: export HF_SPACE_NAME=username/space-name"
    exit 1
fi

echo "📦 Building Docker image..."
docker build -f infra/Dockerfile.backend -t ml-workflow:latest .

echo ""
echo "🏷️  Tagging image for Hugging Face..."
docker tag ml-workflow:latest registry.hf.space/$HF_SPACE_NAME:latest

echo ""
echo "📤 Pushing to Hugging Face Spaces..."
echo "   Space: $HF_SPACE_NAME"
docker login -u $HF_SPACE_NAME -p $HF_API_TOKEN registry.hf.space
docker push registry.hf.space/$HF_SPACE_NAME:latest

echo ""
echo "✅ Deployment complete!"
echo "   Your space should be available at: https://huggingface.co/spaces/$HF_SPACE_NAME"
echo ""
