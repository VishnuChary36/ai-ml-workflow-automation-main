#!/bin/bash

# Deploy to Google Cloud Platform Script
# This script deploys the ML workflow platform to GCP Cloud Run

set -e

echo "🚀 Deploying to GCP Cloud Run"
echo "=============================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Check if required env vars are set
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "❌ GCP_PROJECT_ID environment variable not set"
    exit 1
fi

GCP_REGION=${GCP_REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-ml-workflow-backend}

echo "📦 Building and pushing Docker image..."
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME

echo ""
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$GCP_PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $GCP_REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars "DATABASE_URL=$DATABASE_URL,REDIS_URL=$REDIS_URL"

echo ""
echo "✅ Deployment complete!"
echo "   Get service URL with: gcloud run services describe $SERVICE_NAME --region $GCP_REGION --format 'value(status.url)'"
echo ""
