#!/bin/bash

# ML Dashboard - Start Script
# Runs both frontend and backend in one shot

set -e

echo "🚀 Starting ML Dashboard..."

# Check if backend venv exists
if [ ! -d "backend/venv" ]; then
    echo "⚠️  Backend virtual environment not found. Setting up..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# Kill any existing processes on ports 3000 and 8000
echo "🧹 Cleaning up existing processes..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Start backend in background
echo "🔧 Starting backend (port 8000)..."
cd backend
source venv/bin/activate
python3 main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "⏳ Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend ready!"
        break
    fi
    sleep 1
done

# Start frontend
echo "🎨 Starting frontend (port 3000)..."
npm run dev &
FRONTEND_PID=$!

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo ""
echo "✨ ML Dashboard running!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"

# Wait for processes
wait
