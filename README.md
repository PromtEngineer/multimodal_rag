# RAG System - Unified Deployment & Management

A comprehensive Retrieval-Augmented Generation (RAG) system with intelligent document processing, multi-modal capabilities, and smart query routing.

## 🚀 Quick Start

### Single Command Startup
```bash
# Development mode (recommended for local development)
python run_system.py

# Or using Make
make dev
```

### Production Deployment
```bash
# Docker deployment (recommended for production)
make deploy

# Or manual production mode
make prod
```

## 📋 System Architecture

The system consists of four main components:

1. **Ollama Server** (Port 11434) - LLM inference engine
2. **RAG API Server** (Port 8001) - Advanced RAG pipeline
3. **Backend Server** (Port 8000) - Session management and routing
4. **Frontend** (Port 3000) - Next.js web interface

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Node.js 18+
- Ollama ([install here](https://ollama.ai))
- Docker (optional, for containerized deployment)

### Setup
```bash
# Clone repository
git clone <repository-url>
cd rag_system_old

# Install dependencies
make install

# Start system
make dev
```

## 📖 Usage Guide

### Development Commands
```bash
# Start in development mode with hot reload
make dev

# Start without frontend (backend only)
make dev-no-frontend

# View aggregated logs from all services
make logs

# Check system health
make health

# Stop all services
make stop
```

### Production Commands
```bash
# Start in production mode
make prod

# Docker deployment
make docker-build    # Build images
make docker-up       # Start containers
make docker-down     # Stop containers
make docker-logs     # View container logs
```

### Maintenance Commands
```bash
# Clean temporary files and logs
make clean

# Update system
make update

# Create backup
make backup

# Monitor system resources
make monitor
```

## 🔧 Configuration

### Environment Variables
```bash
# RAG API Configuration
OLLAMA_HOST=http://localhost:11434
NODE_ENV=development

# Backend Configuration
RAG_API_URL=http://localhost:8001

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Service Configuration
The unified launcher (`run_system.py`) supports various options:

```bash
# Development mode with frontend
python run_system.py --mode dev

# Production mode
python run_system.py --mode prod

# Skip frontend startup
python run_system.py --no-frontend

# View logs only (don't start services)
python run_system.py --logs-only

# Health check
python run_system.py --health
```

## 📊 Features

### Smart Query Routing
- **Direct LLM**: Fast responses for general queries (~1.3s)
- **RAG Pipeline**: Document-aware responses for specific queries
- **Automatic Detection**: Uses document overviews for intelligent routing

### Document Processing
- **Multi-format Support**: PDF, text, markdown
- **Intelligent Chunking**: Recursive markdown chunking with overlap
- **Contextual Enrichment**: LLM-based content enhancement
- **Vector Indexing**: Dense embeddings with LanceDB

### Advanced Retrieval
- **Hybrid Search**: Combines dense and sparse (BM25) retrieval
- **AI Reranking**: ColBERT-based relevance scoring
- **Late-chunk Merging**: Expands context while maintaining precision
- **Graph Extraction**: Entity-relationship knowledge graphs

### User Interface
- **Modern Design**: Glass-morphism UI with responsive layout
- **Real-time Chat**: Streaming responses with typing indicators
- **Session Management**: Persistent conversation history
- **Index Management**: Document upload and indexing interface

## 📁 Project Structure

```
rag_system_old/
├── run_system.py              # Unified system launcher
├── Makefile                   # Development and deployment commands
├── docker-compose.yml         # Container orchestration
├── requirements.txt           # Python dependencies
├── package.json              # Node.js dependencies
│
├── rag_system/               # Core RAG implementation
│   ├── main.py              # Agent factory and configuration
│   ├── api_server.py        # RAG API server
│   ├── pipelines/           # Indexing and retrieval pipelines
│   ├── agent/               # ReAct agent implementation
│   ├── indexing/            # Document processing
│   ├── retrieval/           # Search and retrieval
│   └── rerankers/           # AI-based reranking
│
├── backend/                  # Backend API server
│   ├── server.py            # Main server with smart routing
│   ├── database.py          # Session and document management
│   └── ollama_client.py     # Ollama integration
│
├── src/                     # Frontend Next.js application
│   ├── app/                 # App router pages
│   ├── components/          # React components
│   └── lib/                 # Utilities and API client
│
├── Documentation/           # Technical documentation
├── logs/                   # Runtime logs
├── lancedb/               # Vector database
├── index_store/           # Document indexes
└── shared_uploads/        # Uploaded files
```

## 🐳 Docker Deployment

### Quick Deploy
```bash
# Build and start all services
make deploy
```

### Manual Docker Commands
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Service Health Monitoring
All services include health checks:
- **Ollama**: API endpoint availability
- **RAG API**: Model listing endpoint
- **Backend**: Health endpoint
- **Frontend**: Homepage accessibility

## 🔍 Monitoring & Debugging

### Log Files
```bash
# View all logs
tail -f logs/*.log

# Specific service logs
tail -f logs/rag-api.log
tail -f logs/backend.log
tail -f logs/frontend.log
tail -f logs/ollama.log
```

### Health Checks
```bash
# System health overview
make health

# Individual service health
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8001/models     # RAG API
curl http://localhost:8000/health     # Backend
curl http://localhost:3000            # Frontend
```

### Performance Monitoring
```bash
# Resource usage
make monitor

# Process information
ps aux | grep -E "(python|node|ollama)"

# Port usage
netstat -tlnp | grep -E "(3000|8000|8001|11434)"
```

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Check what's using the port
lsof -ti:8000

# Kill process using port
kill $(lsof -ti:8000)
```

**Ollama Not Starting**
```bash
# Check Ollama installation
ollama --version

# Manual start
ollama serve
```

**Frontend Build Errors**
```bash
# Clear cache and rebuild
rm -rf .next node_modules
npm install
npm run build
```

**RAG API Errors**
```bash
# Check Python dependencies
pip install -r requirements.txt

# Verify model availability
ollama list
```

### Getting Help
1. Check logs: `make logs`
2. Verify health: `make health`
3. Review documentation in `Documentation/`
4. Check GitHub issues

## 🔄 Development Workflow

### Local Development
1. `make dev` - Start all services
2. Edit code (hot reload enabled)
3. `make logs` - Monitor changes
4. `make stop` - Stop when done

### Production Testing
1. `make prod` - Test production mode
2. `make docker-build` - Build containers
3. `make docker-up` - Test containerized deployment

### Deployment
1. `make deploy` - Full production deployment
2. `make monitor` - Monitor resources
3. `make backup` - Regular backups

## 📈 Performance

### Typical Response Times
- **General Queries**: ~1.3s (direct LLM)
- **Document Queries**: ~3-5s (full RAG pipeline)
- **Index Building**: ~30s per document

### Resource Requirements
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ for models and indexes
- **CPU**: 4+ cores recommended
- **GPU**: Optional, improves performance

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `make dev`
4. Commit changes: `git commit -m "Description"`
5. Push and create pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
