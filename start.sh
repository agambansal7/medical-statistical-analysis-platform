#!/bin/bash

# Medical Statistical Analysis Platform - Quick Start Script

echo "🏥 Medical Statistical Analysis Platform"
echo "========================================"

# Check for required tools
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "📦 Setting up backend..."

cd "$DIR/backend"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt -q

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found!"
    echo "Please create backend/.env with your ANTHROPIC_API_KEY"
    echo ""
    echo "Example:"
    echo "  ANTHROPIC_API_KEY=your_key_here"
    echo ""
    read -p "Do you want to enter your API key now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your Anthropic API key: " api_key
        echo "ANTHROPIC_API_KEY=$api_key" > .env
        echo "DEBUG=true" >> .env
        echo "✅ .env file created"
    fi
fi

echo ""
echo "📦 Setting up frontend..."

cd "$DIR/frontend"

# Install npm dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

echo ""
echo "🚀 Starting servers..."
echo ""

# Start backend in background
cd "$DIR/backend"
source venv/bin/activate
echo "Starting FastAPI backend on http://localhost:8000..."
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start frontend
cd "$DIR/frontend"
echo "Starting React frontend on http://localhost:3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Platform is starting!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# Handle shutdown
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for processes
wait
