#!/bin/bash

# Deploy to AWS ECS Script
# This script deploys the ML workflow platform to AWS ECS

set -e

echo "🚀 Deploying to AWS ECS"
echo "======================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install AWS CLI."
    exit 1
fi

# Check required env vars
if [ -z "$AWS_ECR_REPOSITORY" ]; then
    echo "❌ AWS_ECR_REPOSITORY environment variable not set"
    exit 1
fi

AWS_REGION=${AWS_DEFAULT_REGION:-us-east-1}
CLUSTER_NAME=${ECS_CLUSTER_NAME:-ml-workflow-cluster}
SERVICE_NAME=${ECS_SERVICE_NAME:-ml-workflow-service}

echo "🔐 Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ECR_REPOSITORY

echo ""
echo "📦 Building Docker image..."
docker build -f infra/Dockerfile.backend -t ml-workflow:latest .

echo ""
echo "🏷️  Tagging image..."
docker tag ml-workflow:latest $AWS_ECR_REPOSITORY:latest

echo ""
echo "📤 Pushing to ECR..."
docker push $AWS_ECR_REPOSITORY:latest

echo ""
echo "🚀 Updating ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --force-new-deployment \
    --region $AWS_REGION

echo ""
echo "✅ Deployment initiated!"
echo "   Monitor deployment: aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION"
echo ""
