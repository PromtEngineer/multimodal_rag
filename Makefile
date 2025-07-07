# RAG System Makefile
# Provides convenient commands for development and deployment

.PHONY: help dev prod logs stop clean health install docker-build docker-up docker-down

# Default target
help:
	@echo "🚀 RAG System Management Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start system in development mode"
	@echo "  make logs         - Show aggregated logs"
	@echo "  make health       - Check system health"
	@echo "  make stop         - Stop all services"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start system in production mode"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start with Docker Compose"
	@echo "  make docker-down  - Stop Docker containers"
	@echo ""
	@echo "Maintenance:"
	@echo "  make install      - Install dependencies"
	@echo "  make clean        - Clean logs and temporary files"
	@echo "  make test         - Run system tests"

# Development commands
dev:
	@echo "🚀 Starting RAG system in development mode..."
	python run_system.py --mode dev

prod:
	@echo "🚀 Starting RAG system in production mode..."
	python run_system.py --mode prod

logs:
	@echo "📋 Showing aggregated logs..."
	python run_system.py --logs-only

health:
	@echo "🔍 Checking system health..."
	python run_system.py --health

stop:
	@echo "🛑 Stopping all services..."
	python run_system.py --stop

# Installation and setup
install:
	@echo "📦 Installing dependencies..."
	python install_dependencies.py

install-dev:
	@echo "📦 Installing dependencies (skip models)..."
	python install_dependencies.py --skip-models

install-backend-only:
	@echo "📦 Installing backend dependencies only..."
	python install_dependencies.py --skip-npm

# Docker commands
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up:
	@echo "🐳 Starting with Docker Compose..."
	docker-compose up -d

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose logs -f

# Maintenance
clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf logs/*.log
	rm -rf __pycache__/
	rm -rf .next/
	rm -rf node_modules/.cache/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

test:
	@echo "🧪 Running system tests..."
	python -m pytest tests/ -v

# Quick start for new users
quickstart:
	@echo "🚀 Quick start setup..."
	@echo "1. Installing dependencies..."
	make install
	@echo "2. Starting system..."
	make dev

# Production deployment
deploy:
	@echo "🚀 Deploying to production..."
	make docker-build
	make docker-up
	@echo "✅ Deployment complete!"
	@echo "🌐 Access at: http://localhost:3000"

# Development with specific options
dev-no-frontend:
	@echo "🚀 Starting without frontend..."
	python run_system.py --mode dev --no-frontend

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	mkdir -p backups
	tar -czf backups/rag-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		backend/chat_data.db lancedb/ index_store/ shared_uploads/

restore:
	@echo "📂 Available backups:"
	@ls -la backups/ || echo "No backups found"
	@echo "To restore: tar -xzf backups/backup-file.tar.gz"

# Monitor system resources
monitor:
	@echo "📊 System resource monitoring..."
	@echo "Memory usage:"
	@ps aux | grep -E "(python|node|ollama)" | grep -v grep
	@echo ""
	@echo "Port usage:"
	@netstat -tlnp | grep -E "(3000|8000|8001|11434)"

# Update system
update:
	@echo "🔄 Updating system..."
	git pull
	pip install -r requirements.txt --upgrade
	npm update
	@echo "✅ Update complete!" 