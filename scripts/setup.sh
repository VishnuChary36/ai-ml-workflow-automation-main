#!/bin/bash
# ============================================================
# AI-ML Workflow Automation Platform — Local Setup Script
# ============================================================
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh          # Docker mode (recommended)
#   ./scripts/setup.sh --local  # Local venv mode (no Docker)
# ============================================================

set -e

MODE="docker"
if [[ "$1" == "--local" ]]; then
    MODE="local"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ""
echo "========================================================"
echo " AI-ML Workflow Automation Platform — Setup"
echo "========================================================"
echo ""

# ── 1. Prerequisite checks ────────────────────────────────

check_python() {
    if ! command -v python3 &>/dev/null; then
        echo "❌  Python 3 not found. Please install Python 3.10+."
        exit 1
    fi
    PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYMAJ=$(echo "$PYVER" | cut -d. -f1)
    PYMIN=$(echo "$PYVER" | cut -d. -f2)
    if [[ "$PYMAJ" -lt 3 ]] || { [[ "$PYMAJ" -eq 3 ]] && [[ "$PYMIN" -lt 10 ]]; }; then
        echo "❌  Python $PYVER found but Python 3.10+ is required."
        exit 1
    fi
    echo "   ✓ Python $PYVER"
}

check_node() {
    if ! command -v node &>/dev/null; then
        echo "❌  Node.js not found. Please install Node.js 18+."
        exit 1
    fi
    NODE_VER=$(node --version)
    echo "   ✓ Node.js $NODE_VER"
}

check_docker() {
    if ! command -v docker &>/dev/null; then
        echo "❌  Docker not found. Please install Docker Desktop."
        exit 1
    fi
    echo "   ✓ $(docker --version)"

    # Accept both 'docker compose' (v2) and 'docker-compose' (v1)
    if docker compose version &>/dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    elif command -v docker-compose &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    else
        echo "❌  Docker Compose not found. Please install Docker Compose v2."
        exit 1
    fi
    echo "   ✓ Docker Compose ($COMPOSE_CMD)"
}

echo "1. Checking prerequisites..."
if [[ "$MODE" == "docker" ]]; then
    check_docker
else
    check_python
    check_node
fi

# ── 2. Copy .env if it does not exist ─────────────────────

echo ""
echo "2. Setting up environment file..."
if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "   ✓ .env already exists — skipping copy."
else
    cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
    echo "   ✓ Created .env from .env.example"
    echo "   ℹ  Edit .env and add any API keys you need (OPENAI_API_KEY, etc.)"
fi

# ── 3. Mode-specific setup ────────────────────────────────

if [[ "$MODE" == "docker" ]]; then
    echo ""
    echo "3. Building and starting services with Docker Compose..."
    cd "$REPO_ROOT"
    $COMPOSE_CMD -f infra/docker-compose.yml up --build -d

    echo ""
    echo "4. Waiting for services to be healthy..."
    MAX_WAIT=120
    WAITED=0
    until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "   ❌ Backend did not become healthy within ${MAX_WAIT}s."
            echo "      Check logs: $COMPOSE_CMD -f infra/docker-compose.yml logs backend"
            exit 1
        fi
        printf "   Waiting for backend... (%ds)\r" "$WAITED"
        sleep 5
        WAITED=$((WAITED + 5))
    done
    echo "   ✓ Backend is healthy (http://localhost:8000)"

    WAITED=0
    until curl -sf http://localhost:5173 >/dev/null 2>&1; do
        if [[ $WAITED -ge $MAX_WAIT ]]; then
            echo "   ⚠  Frontend did not respond within ${MAX_WAIT}s (may still be building)."
            break
        fi
        printf "   Waiting for frontend... (%ds)\r" "$WAITED"
        sleep 5
        WAITED=$((WAITED + 5))
    done
    echo "   ✓ Frontend is ready  (http://localhost:5173)"

else
    # ── Local mode ──────────────────────────────────────────

    echo ""
    echo "3. Installing backend Python dependencies..."
    cd "$REPO_ROOT/backend"
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        echo "   ✓ Created virtual environment at backend/venv"
    else
        echo "   ✓ Virtual environment already exists"
    fi
    # shellcheck source=/dev/null
    source venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    echo "   ✓ Python dependencies installed"

    echo ""
    echo "4. Installing frontend Node.js dependencies..."
    cd "$REPO_ROOT/frontend"
    npm install --silent
    echo "   ✓ Node.js dependencies installed"

    echo ""
    echo "========================================================"
    echo "✅ Dependencies installed!"
    echo ""
    echo "⚠  For local mode you still need PostgreSQL and Redis"
    echo "   running locally.  The easiest way is:"
    echo "   docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15"
    echo "   docker run -d -p 6379:6379 redis:7-alpine"
    echo ""
    echo "Then start the application:"
    echo ""
    echo "  Terminal 1 — Backend:"
    echo "    cd backend && source venv/bin/activate"
    echo "    uvicorn app:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "  Terminal 2 — Frontend:"
    echo "    cd frontend && npm run dev"
    echo ""
    echo "  Open http://localhost:5173 in your browser."
    echo "========================================================"
    exit 0
fi

# ── Final message (Docker mode) ───────────────────────────

echo ""
echo "========================================================"
echo "✅ Platform is running!"
echo ""
echo "  Frontend  →  http://localhost:5173"
echo "  Backend   →  http://localhost:8000"
echo "  API Docs  →  http://localhost:8000/docs"
echo ""
echo "  Demo dataset: ./demo/sample_dataset.csv"
echo ""
echo "To stop:  $COMPOSE_CMD -f infra/docker-compose.yml down"
echo "Logs:     $COMPOSE_CMD -f infra/docker-compose.yml logs -f"
echo "========================================================"
