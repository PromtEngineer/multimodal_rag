# 📦 RAG System Installation Guide

_Last updated: 2025-01-02_

This guide provides step-by-step instructions for installing and setting up the RAG system on different operating systems.

---

## 1. Prerequisites

### 1.1 System Requirements

#### **Minimum Requirements**
- **CPU**: 4 cores, 2.5GHz+
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB free space
- **OS**: macOS 10.15+, Ubuntu 20.04+, Windows 10+

#### **Recommended Requirements**
- **CPU**: 8+ cores, 3.0GHz+
- **RAM**: 32GB+ (for large models)
- **Storage**: 200GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional)

### 1.2 Software Dependencies

- **Docker**: 24.0+ (with Docker Compose)
- **Git**: 2.30+
- **Python**: 3.11+ (for local development)
- **Node.js**: 18+ (for local development)

---

## 2. Docker Installation

### 2.1 macOS Installation

#### **Option 1: Docker Desktop (Recommended)**
```bash
# Install via Homebrew
brew install --cask docker

# Or download from: https://www.docker.com/products/docker-desktop/

# Start Docker Desktop from Applications
# Verify installation
docker --version
docker compose version
```

#### **Option 2: Command Line Installation**
```bash
# Install Docker via Homebrew
brew install docker docker-compose

# Install Docker Machine (for VM management)
brew install docker-machine

# Create and start Docker machine
docker-machine create --driver virtualbox default
docker-machine start default
eval $(docker-machine env default)
```

### 2.2 Ubuntu/Debian Installation

```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Verify installation
docker --version
docker compose version
```

### 2.3 CentOS/RHEL Installation

```bash
# Install required packages
sudo yum install -y yum-utils

# Add Docker repository
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker Engine
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

### 2.4 Windows Installation

#### **Option 1: Docker Desktop (Recommended)**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop/
2. Run the installer and follow the setup wizard
3. Enable WSL 2 integration if prompted
4. Restart your computer
5. Start Docker Desktop
6. Verify installation in PowerShell:
```powershell
docker --version
docker compose version
```

#### **Option 2: WSL2 + Docker Engine**
```bash
# In WSL2 Ubuntu terminal
# Follow Ubuntu installation steps above
```

---

## 3. RAG System Installation

### 3.1 Quick Installation

```bash
# Clone repository
git clone https://github.com/your-org/rag-system.git
cd rag-system

# Create environment file
cp .env.example .env

# Create required directories
mkdir -p {lancedb,shared_uploads,logs,ollama_data}
mkdir -p index_store/{overviews,bm25,graph}

# Start the system
docker compose up -d

# Wait for services to start (2-3 minutes)
sleep 180

# Check system status
docker compose ps
```

### 3.2 Detailed Installation

#### **Step 1: Repository Setup**
```bash
# Clone the repository
git clone https://github.com/your-org/rag-system.git
cd rag-system

# Check available branches
git branch -a

# Switch to main branch (if not already)
git checkout main

# Verify repository contents
ls -la
```

#### **Step 2: Environment Configuration**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env

# Example configuration
cat > .env << 'EOF'
# System Configuration
NODE_ENV=production
LOG_LEVEL=info

# Service URLs
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
RAG_API_URL=http://localhost:8001
OLLAMA_URL=http://localhost:11434

# Model Configuration
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
DEFAULT_GENERATION_MODEL=qwen2.5:7b
DEFAULT_RERANKER_MODEL=BAAI/bge-reranker-base

# Performance Configuration
MAX_CONCURRENT_REQUESTS=5
REQUEST_TIMEOUT=300
EMBEDDING_BATCH_SIZE=32
EOF
```

#### **Step 3: Directory Structure**
```bash
# Create required directories
mkdir -p lancedb
mkdir -p shared_uploads
mkdir -p logs
mkdir -p ollama_data
mkdir -p index_store/overviews
mkdir -p index_store/bm25
mkdir -p index_store/graph

# Set proper permissions
chmod 755 {lancedb,shared_uploads,logs,ollama_data}
chmod 755 index_store/{overviews,bm25,graph}

# Verify directory structure
tree -d -L 2
```

#### **Step 4: Docker Build and Start**
```bash
# Build containers
docker compose build --no-cache

# Start services
docker compose up -d

# Check build progress
docker compose logs -f

# Verify all services are running
docker compose ps
```

#### **Step 5: Model Installation**
```bash
# Wait for Ollama to start
sleep 60

# Install required models
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull qwen2.5:0.5b

# Verify model installation
docker compose exec ollama ollama list
```

#### **Step 6: System Verification**
```bash
# Check service health
curl -f http://localhost:3000 && echo "Frontend: OK"
curl -f http://localhost:8000/health && echo "Backend: OK"
curl -f http://localhost:8001/models && echo "RAG API: OK"
curl -f http://localhost:11434/api/tags && echo "Ollama: OK"

# Check logs for errors
docker compose logs | grep -i error
```

---

## 4. Development Installation

### 4.1 Local Development Setup

#### **Python Environment**
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional development dependencies
pip install pytest black flake8 mypy
```

#### **Node.js Environment**
```bash
# Install Node.js dependencies
npm install

# Install development dependencies
npm install --save-dev @types/node typescript eslint prettier
```

#### **Development Configuration**
```bash
# Create development environment
cp .env.example .env.dev

# Edit for development
cat > .env.dev << 'EOF'
NODE_ENV=development
LOG_LEVEL=debug
DEBUG=true

# Local URLs
FRONTEND_URL=http://localhost:3000
BACKEND_URL=http://localhost:8000
RAG_API_URL=http://localhost:8001
OLLAMA_URL=http://localhost:11434

# Development settings
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
API_KEY_REQUIRED=false
EOF
```

### 4.2 Local Development Servers

#### **Option 1: Mixed Mode (Recommended)**
```bash
# Start only infrastructure services
docker compose up -d ollama

# Start backend locally
cd backend
python server.py

# Start RAG API locally
cd rag_system
python -m api_server

# Start frontend locally
npm run dev
```

#### **Option 2: Full Docker Development**
```bash
# Start with development overrides
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Enable hot reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

---

## 5. Configuration

### 5.1 Model Configuration

#### **Embedding Models**
```python
# Edit rag_system/main.py
EXTERNAL_MODELS = {
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",  # 768D
    "reranker_model": "BAAI/bge-reranker-base",
}
```

#### **Generation Models**
```python
# Edit rag_system/main.py
OLLAMA_CONFIG = {
    "generation_model": "qwen2.5:7b",
    "enrichment_model": "qwen2.5:0.5b",
    "host": "http://localhost:11434"
}
```

### 5.2 Performance Configuration

#### **Resource Limits**
```yaml
# Edit docker-compose.yml
services:
  rag-api:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

#### **Pipeline Configuration**
```python
# Edit rag_system/main.py
PIPELINE_CONFIGS = {
    "query_decomposition": {"enabled": True},
    "contextual_enricher": {"enabled": True},
    "verification": {"enabled": True},
    "retrieval": {
        "search_type": "hybrid",
        "fusion": {"dense_weight": 0.7, "sparse_weight": 0.3}
    }
}
```

---

## 6. Troubleshooting Installation

### 6.1 Docker Issues

#### **Docker Not Found**
```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Check installation
docker --version
```

#### **Permission Denied**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Restart shell or log out/in
newgrp docker

# Test Docker
docker run hello-world
```

#### **Docker Compose Not Found**
```bash
# Install Docker Compose plugin
sudo apt-get install docker-compose-plugin

# Or install standalone
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 6.2 Build Issues

#### **Build Failures**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker compose build --no-cache

# Check build logs
docker compose logs --no-color > build.log
```

#### **Port Conflicts**
```bash
# Check port usage
sudo netstat -tulpn | grep -E ":3000|:8000|:8001|:11434"

# Kill processes using ports
sudo kill -9 $(sudo lsof -t -i:3000)

# Change ports in docker-compose.yml
```

#### **Memory Issues**
```bash
# Increase Docker memory (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# Check available memory
free -h

# Monitor Docker memory usage
docker stats
```

### 6.3 Model Issues

#### **Model Download Failures**
```bash
# Check Ollama status
docker compose exec ollama ollama list

# Manual model download
docker compose exec ollama ollama pull qwen2.5:7b

# Check available space
df -h

# Clear model cache
docker compose exec ollama rm -rf /root/.ollama/models/*
```

#### **Embedding Model Issues**
```bash
# Check Python dependencies
docker compose exec rag-api pip list | grep -E "torch|transformers|sentence-transformers"

# Test embedding model
docker compose exec rag-api python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('Model loaded successfully')
"
```

---

## 7. Verification & Testing

### 7.1 System Health Check

```bash
#!/bin/bash
# health_check.sh - Complete system verification

echo "=== RAG System Installation Verification ==="

# Check Docker
echo "1. Docker Installation:"
docker --version || echo "❌ Docker not installed"
docker compose version || echo "❌ Docker Compose not installed"

# Check containers
echo -e "\n2. Container Status:"
docker compose ps

# Check ports
echo -e "\n3. Port Accessibility:"
for port in 3000 8000 8001 11434; do
    if nc -z localhost $port; then
        echo "✅ Port $port: ACCESSIBLE"
    else
        echo "❌ Port $port: NOT ACCESSIBLE"
    fi
done

# Check services
echo -e "\n4. Service Health:"
curl -s -f http://localhost:3000 && echo "✅ Frontend: OK" || echo "❌ Frontend: FAIL"
curl -s -f http://localhost:8000/health && echo "✅ Backend: OK" || echo "❌ Backend: FAIL"
curl -s -f http://localhost:8001/models && echo "✅ RAG API: OK" || echo "❌ RAG API: FAIL"
curl -s -f http://localhost:11434/api/tags && echo "✅ Ollama: OK" || echo "❌ Ollama: FAIL"

# Check models
echo -e "\n5. Model Status:"
docker compose exec ollama ollama list

# Check disk space
echo -e "\n6. Disk Usage:"
df -h | grep -E "/$|/var|/opt"

# Check memory
echo -e "\n7. Memory Usage:"
free -h

echo -e "\n=== Verification Complete ==="
```

### 7.2 Functional Testing

```bash
# Test document upload
curl -X POST -F "file=@test_document.pdf" http://localhost:8000/upload

# Test chat functionality
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "Hello, how are you?", "session_id": "test-session"}' \
  http://localhost:8000/sessions/test-session/chat

# Test RAG functionality
curl -X POST -H "Content-Type: application/json" \
  -d '{"query": "What is in the document?", "session_id": "test-session"}' \
  http://localhost:8001/chat
```

---

## 8. Next Steps

### 8.1 First Use

1. **Access the system**: Open http://localhost:3000 in your browser
2. **Create a session**: Click "New Chat" to start
3. **Upload documents**: Use the upload interface to add PDF files
4. **Test queries**: Ask questions about your documents
5. **Explore features**: Try different query types and settings

### 8.2 Customization

1. **Configure models**: Edit `rag_system/main.py` for different models
2. **Adjust performance**: Modify resource limits in `docker-compose.yml`
3. **Customize UI**: Edit React components in `src/components/`
4. **Add features**: Extend the system with new capabilities

### 8.3 Production Deployment

1. **Review security**: Configure firewalls and SSL certificates
2. **Set up monitoring**: Implement logging and alerting
3. **Configure backups**: Set up automated data backups
4. **Load testing**: Test system performance under load

---

## 9. Installation Scripts

### 9.1 Automated Installation Script

```bash
#!/bin/bash
# install_rag_system.sh - Automated installation

set -e

echo "=== RAG System Automated Installation ==="

# Check prerequisites
echo "1. Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git first."
    exit 1
fi

# Clone repository
echo "2. Cloning repository..."
if [ ! -d "rag-system" ]; then
    git clone https://github.com/your-org/rag-system.git
fi
cd rag-system

# Setup environment
echo "3. Setting up environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
fi

# Create directories
echo "4. Creating directories..."
mkdir -p {lancedb,shared_uploads,logs,ollama_data}
mkdir -p index_store/{overviews,bm25,graph}

# Build and start
echo "5. Building and starting services..."
docker compose build --no-cache
docker compose up -d

# Wait for services
echo "6. Waiting for services to start..."
sleep 180

# Install models
echo "7. Installing AI models..."
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull qwen2.5:0.5b

# Verify installation
echo "8. Verifying installation..."
docker compose ps

echo "✅ Installation complete!"
echo "Access the system at: http://localhost:3000"
```

### 9.2 Development Setup Script

```bash
#!/bin/bash
# setup_development.sh - Development environment setup

set -e

echo "=== RAG System Development Setup ==="

# Check Python
echo "1. Checking Python..."
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found. Please install Python 3.11."
    exit 1
fi

# Check Node.js
echo "2. Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+."
    exit 1
fi

# Setup Python environment
echo "3. Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Setup Node.js environment
echo "4. Setting up Node.js environment..."
npm install

# Setup development environment
echo "5. Setting up development configuration..."
cp .env.example .env.dev
echo "NODE_ENV=development" >> .env.dev
echo "LOG_LEVEL=debug" >> .env.dev

# Start infrastructure
echo "6. Starting infrastructure services..."
docker compose up -d ollama

echo "✅ Development setup complete!"
echo "To start development:"
echo "1. Backend: cd backend && python server.py"
echo "2. RAG API: cd rag_system && python -m api_server"
echo "3. Frontend: npm run dev"
```

---

This installation guide provides comprehensive instructions for setting up the RAG system on any platform. Follow the appropriate section for your operating system and use case. 