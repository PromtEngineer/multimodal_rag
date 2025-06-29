#!/bin/bash

# RAG System Complete Startup Script
# This script starts all components of the RAG system

echo "🚀 Starting RAG System Components..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install: https://ollama.ai"
    exit 1
fi

# Function to check if port is in use
check_port() {
    lsof -ti:$1 >/dev/null 2>&1
}

# Function to start service in background and track PID
start_service() {
    local name=$1
    local cmd=$2
    local port=$3
    local log_file=$4
    
    if check_port $port; then
        echo "⚠️  Port $port already in use. Skipping $name."
        return
    fi
    
    echo "🔄 Starting $name on port $port..."
    $cmd > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "${name}.pid"
    
    # Wait a moment and check if service started successfully
    sleep 2
    if kill -0 $pid 2>/dev/null; then
        echo "✅ $name started successfully (PID: $pid)"
    else
        echo "❌ Failed to start $name"
        return 1
    fi
}

# Create logs directory
mkdir -p logs

echo "1️⃣ Starting Ollama server..."
if ! pgrep -f "ollama serve" > /dev/null; then
    ollama serve > logs/ollama.log 2>&1 &
    echo $! > ollama.pid
    echo "✅ Ollama server started"
    sleep 5  # Give Ollama time to start
else
    echo "✅ Ollama already running"
fi

# Check if required models are available
echo "🔍 Checking required models..."
if ! ollama list | grep -q "qwen3:8b"; then
    echo "📥 Pulling qwen3:8b model..."
    ollama pull qwen3:8b
fi

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "📥 Pulling nomic-embed-text model..."
    ollama pull nomic-embed-text
fi

echo "2️⃣ Starting RAG API server (port 8001)..."
start_service "rag-api" "python -m rag_system.api_server" 8001 "logs/rag-api.log"

echo "3️⃣ Starting Backend server (port 8000)..."
cd backend
start_service "backend" "python server.py" 8000 "../logs/backend.log"
cd ..

echo "4️⃣ Starting Frontend server (port 3000)..."
if command -v npm &> /dev/null; then
    start_service "frontend" "npm run dev" 3000 "logs/frontend.log"
else
    echo "❌ npm not found. Please install Node.js"
fi

echo ""
echo "🎉 RAG System Started!"
echo "📊 Services Status:"
echo "   • Ollama:    http://localhost:11434"
echo "   • RAG API:   http://localhost:8001" 
echo "   • Backend:   http://localhost:8000"
echo "   • Frontend:  http://localhost:3000"
echo ""
echo "🌐 Access your RAG system at: http://localhost:3000"
echo ""
echo "📋 Useful commands:"
echo "   • Check status: ./check_status.sh"
echo "   • Stop system:  ./stop_system.sh"
echo "   • View logs:    tail -f logs/*.log"
echo ""
echo "💡 The system uses smart routing:"
echo "   • General queries → Direct LLM (fast ~1.3s)"
echo "   • Document queries → RAG pipeline (comprehensive)" 