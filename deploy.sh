#!/bin/bash

# RAG System Deployment Script
# Usage: ./deploy.sh [development|production]

set -e

MODE=${1:-development}
echo "🚀 Deploying RAG System in $MODE mode..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p shared_uploads
mkdir -p index_store/overviews
mkdir -p lancedb

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    if [ -f env.example ]; then
        cp env.example .env
        echo "📋 Created .env file from template. Please edit it with your settings."
    else
        echo "⚠️  No .env file found. Creating minimal configuration..."
        cat > .env << EOF
RAG_CONFIG_MODE=default
RAG_LOG_LEVEL=INFO
PORT=8000
OLLAMA_HOST=http://ollama:11434
EOF
    fi
fi

# Set appropriate permissions
chmod 755 shared_uploads
chmod 755 index_store
chmod 755 lancedb

if [ "$MODE" = "production" ]; then
    echo "🏭 Production deployment..."
    
    # Build and start services
    docker-compose down --remove-orphans
    docker-compose build --no-cache
    docker-compose up -d
    
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Health check
    echo "🔍 Checking service health..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ RAG Server is healthy"
    else
        echo "❌ RAG Server health check failed"
        docker-compose logs rag-server
        exit 1
    fi
    
    echo "🎉 Production deployment complete!"
    echo "📍 RAG Server: http://localhost:8000"
    echo "📍 Health Check: http://localhost:8000/health"
    echo "📍 Ollama: http://localhost:11434"
    
elif [ "$MODE" = "development" ]; then
    echo "🔧 Development deployment..."
    
    # Install Python dependencies
    if [ -f requirements.txt ]; then
        echo "📦 Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    # Start only Ollama in Docker for development
    docker-compose up -d ollama
    
    echo "⏳ Waiting for Ollama to start..."
    sleep 10
    
    echo "✅ Development environment ready!"
    echo "🏃 Run the server with: python server.py"
    echo "📍 Ollama: http://localhost:11434"
    
else
    echo "❌ Invalid mode. Use 'development' or 'production'"
    exit 1
fi

# Show useful commands
echo ""
echo "📚 Useful commands:"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart: docker-compose restart"
echo "  Shell access: docker-compose exec rag-server bash" 