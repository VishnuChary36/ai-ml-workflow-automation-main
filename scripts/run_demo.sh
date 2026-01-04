#!/bin/bash

# AI-ML Workflow Automation - Demo Script
# This script starts the full stack and runs a demo workflow

set -e

echo "🚀 Starting AI-ML Workflow Automation Platform..."
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker Compose."
    exit 1
fi

# Start services
echo "📦 Starting services with Docker Compose..."
docker-compose -f infra/docker-compose.yml up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check backend health
echo "🔍 Checking backend health..."
until curl -f http://localhost:8000/health &> /dev/null; do
    echo "   Waiting for backend..."
    sleep 2
done

echo "✅ Backend is ready!"

# Check frontend
echo "🔍 Checking frontend..."
until curl -f http://localhost:5173 &> /dev/null; do
    echo "   Waiting for frontend..."
    sleep 2
done

echo "✅ Frontend is ready!"

echo ""
echo "🎉 Platform is running!"
echo ""
echo "📍 Access points:"
echo "   Frontend:  http://localhost:5173"
echo "   Backend:   http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "📊 Demo dataset available at: ./demo/sample_dataset.csv"
echo ""
echo "📖 Next steps:"
echo "   1. Open http://localhost:5173 in your browser"
echo "   2. Upload the demo dataset (./demo/sample_dataset.csv)"
echo "   3. Review AI-suggested preprocessing pipeline"
echo "   4. Click 'Run Pipeline' and watch live console logs"
echo "   5. Explore model suggestions and training"
echo ""
echo "🛑 To stop: docker-compose -f infra/docker-compose.yml down"
echo ""
