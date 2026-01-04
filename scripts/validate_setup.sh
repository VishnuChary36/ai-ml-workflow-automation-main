#!/bin/bash

# Setup script for AI-ML Workflow Automation Platform
# This script validates the installation and prepares the environment

set -e

echo "🔍 Validating AI-ML Workflow Automation Platform Setup"
echo "======================================================"
echo ""

# Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✓ $PYTHON_VERSION"
else
    echo "   ❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check Node.js
echo "2. Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "   ✓ Node.js $NODE_VERSION"
else
    echo "   ❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

# Check Docker
echo "3. Checking Docker..."
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo "   ✓ $DOCKER_VERSION"
else
    echo "   ⚠️  Docker not found. Docker is recommended for easy deployment."
fi

# Check docker-compose
echo "4. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version)
    echo "   ✓ $COMPOSE_VERSION"
else
    echo "   ⚠️  Docker Compose not found. Required for docker deployment."
fi

echo ""
echo "5. Checking project structure..."
REQUIRED_DIRS=("backend" "frontend" "infra" "demo" "scripts")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ✓ $dir/"
    else
        echo "   ❌ Missing directory: $dir/"
        exit 1
    fi
done

echo ""
echo "6. Checking required files..."
REQUIRED_FILES=(
    ".env.example"
    "README.md"
    "backend/requirements.txt"
    "backend/app.py"
    "frontend/package.json"
    "infra/docker-compose.yml"
)
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ❌ Missing file: $file"
        exit 1
    fi
done

echo ""
echo "======================================================"
echo "✅ Validation Complete!"
echo ""
echo "📝 Next Steps:"
echo ""
echo "Option 1 - Docker (Recommended):"
echo "  1. cp .env.example .env"
echo "  2. docker-compose -f infra/docker-compose.yml up --build"
echo "  3. Open http://localhost:5173"
echo ""
echo "Option 2 - Local Development:"
echo "  Backend:"
echo "    cd backend"
echo "    python3 -m venv venv"
echo "    source venv/bin/activate"
echo "    pip install -r requirements.txt"
echo "    uvicorn app:app --reload"
echo ""
echo "  Frontend (in new terminal):"
echo "    cd frontend"
echo "    npm install"
echo "    npm run dev"
echo ""
echo "📖 Full documentation: README.md"
echo ""
