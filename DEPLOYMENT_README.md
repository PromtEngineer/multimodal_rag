# RAG System - Deployment Guide

## 🚀 Production-Ready Consolidated RAG System

This is a comprehensive Retrieval-Augmented Generation (RAG) system that has been consolidated from multiple services into a single, deployment-ready server.

## ✨ Features

- **Unified Server**: Single server combining frontend API, session management, and RAG processing
- **Smart Routing**: Intelligent query routing between direct LLM and RAG pipeline
- **Multi-Modal Support**: Text, PDF, and document processing
- **Session Management**: Persistent chat sessions with history
- **Index Management**: Document indexing with multiple retrieval strategies
- **Health Monitoring**: Built-in health checks and monitoring
- **Docker Support**: Complete containerization for easy deployment
- **Production Ready**: Logging, error handling, and performance optimization

## 📋 Prerequisites

### Required
- Python 3.11+
- Docker & Docker Compose
- 4GB+ RAM (8GB+ recommended)
- 10GB+ disk space

### Optional
- GPU support for faster inference
- Ollama for local LLM hosting

## 🏗️ Architecture

```
Frontend (Next.js) ←→ Consolidated Server (Port 8000) ←→ RAG Pipeline
                           ↓
                      ┌─────────────┐
                      │  Database   │
                      │  Storage    │
                      │  Indexes    │
                      └─────────────┘
```

### Key Components
- **Session Management**: Chat sessions, user messages, conversation history
- **File Processing**: Upload, chunking, embedding, indexing
- **Smart Routing**: Context-aware LLM vs RAG decision making
- **RAG Pipeline**: Dense retrieval, BM25, reranking, answer synthesis
- **Storage**: LanceDB (vectors), SQLite (metadata), file system (documents)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository>
cd rag_system_old
cp env.example .env
# Edit .env with your configuration
```

### 2. Development Deployment
```bash
./deploy.sh development
python server.py
```

### 3. Production Deployment
```bash
./deploy.sh production
```

### 4. Verify Installation
```bash
python test_consolidated_system.py --wait
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Server Configuration
RAG_CONFIG_MODE=default        # default|fast|accurate
RAG_LOG_LEVEL=INFO            # DEBUG|INFO|WARNING|ERROR
PORT=8000                     # Server port

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Model Configuration
DEFAULT_GENERATION_MODEL=qwen3:8b
DEFAULT_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
DEFAULT_ENRICHMENT_MODEL=qwen3:0.6b

# Performance Tuning
EMBEDDING_BATCH_SIZE=50
ENRICHMENT_BATCH_SIZE=25
MAX_CHUNK_SIZE=1500
CHUNK_OVERLAP=200
```

### RAG Configuration Modes

#### Default Mode (Balanced)
- Hybrid retrieval (dense + BM25)
- Contextual enrichment enabled
- Late-chunk merging
- AI reranking with ColBERT
- Window size: 2 chunks

#### Fast Mode (Speed Optimized)
- Dense retrieval only
- Smaller embedding models
- Reduced context windows
- Minimal reranking

#### Accurate Mode (Quality Optimized)
- Full hybrid retrieval
- Maximum context windows
- Extensive reranking
- Query decomposition
- Answer verification

## 📚 API Reference

### Core Endpoints

#### Health Check
```bash
GET /health
```

#### Session Management
```bash
POST /sessions                    # Create session
GET /sessions                     # List sessions
GET /sessions/{id}               # Get session
DELETE /sessions/{id}            # Delete session
POST /sessions/{id}/messages     # Send message
```

#### File Management
```bash
POST /sessions/{id}/upload       # Upload files
POST /sessions/{id}/index        # Index documents
GET /sessions/{id}/documents     # List documents
```

#### RAG Endpoints
```bash
POST /rag/chat                   # Direct RAG query
POST /rag/chat/stream           # Streaming RAG
POST /rag/index                 # Index documents
```

#### Models & System
```bash
GET /models                      # List available models
GET /indexes                     # List indexes
POST /indexes                    # Create index
```

### Request Examples

#### Chat with Smart Routing
```bash
curl -X POST http://localhost:8000/sessions/{session_id}/messages \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What does the document say about AI?",
    "retrieval_k": 20,
    "context_window_size": 2,
    "search_type": "hybrid"
  }'
```

#### Force RAG Mode
```bash
curl -X POST http://localhost:8000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the key findings",
    "session_id": "session_123",
    "force_rag": true,
    "ai_rerank": true
  }'
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -t rag-system .

# Run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f rag-server

# Scale for high availability
docker-compose up -d --scale rag-server=3
```

### Docker Configuration
- **Base Image**: python:3.11-slim
- **Port**: 8000
- **Volumes**: Documents, indexes, database
- **Health Check**: Built-in endpoint monitoring
- **Restart Policy**: unless-stopped

## 🔍 Monitoring & Debugging

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# Detailed status
{
  "status": "ok",
  "ollama_running": true,
  "available_models": ["qwen3:8b", "qwen3:0.6b"],
  "database_stats": {...},
  "rag_agent_status": "initialized"
}
```

### Logging
- **Level**: Configurable via RAG_LOG_LEVEL
- **Format**: Structured JSON in production
- **Location**: stdout/stderr (captured by Docker)

### Common Issues

#### Ollama Connection Issues
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Pull required models
ollama pull qwen3:8b
ollama pull qwen3:0.6b
```

#### Memory Issues
- Increase Docker memory limit
- Reduce batch sizes in configuration
- Use smaller embedding models

#### Performance Tuning
- Adjust batch sizes based on available RAM
- Use GPU acceleration if available
- Optimize chunk sizes for your documents

## 🧪 Testing

### Comprehensive Test Suite
```bash
# Run all tests
python test_consolidated_system.py

# Test specific functionality
python test_consolidated_system.py --wait --timeout 120

# Test with custom server
python test_consolidated_system.py --url http://your-server:8000
```

### Manual Testing
```bash
# Test server startup
python server.py

# Test RAG functionality
python -m rag_system.main --mode test --query "Hello world"

# Test indexing
python -m rag_system.main --mode index --files document.pdf
```

## 🔒 Security Considerations

### Production Security
- Use environment variables for sensitive configuration
- Implement authentication/authorization as needed
- Use HTTPS in production
- Restrict file upload types and sizes
- Validate all user inputs

### Network Security
- Use reverse proxy (nginx/Apache) in production
- Implement rate limiting
- Use firewalls to restrict access
- Monitor for unusual activity

## 📈 Performance Optimization

### Hardware Recommendations
- **CPU**: 4+ cores for concurrent requests
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: SSD for indexes and database
- **GPU**: Optional, for faster inference

### Software Optimization
- Use connection pooling for database
- Implement caching for frequent queries
- Optimize embedding batch sizes
- Use async processing for long operations

## 🚀 Deployment Strategies

### Single Server
```bash
./deploy.sh production
```

### Load Balanced (Multiple Instances)
```yaml
# docker-compose.yml
services:
  rag-server:
    build: .
    deploy:
      replicas: 3
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    depends_on:
      - rag-server
```

### Cloud Deployment
- **AWS**: ECS, EC2, or Lambda
- **Google Cloud**: Cloud Run, GKE
- **Azure**: Container Instances, AKS
- **Heroku**: Direct Docker deployment

## 📞 Support & Troubleshooting

### Common Commands
```bash
# View logs
docker-compose logs -f rag-server

# Restart services
docker-compose restart

# Update system
git pull && docker-compose build --no-cache && docker-compose up -d

# Backup data
tar -czf backup.tar.gz shared_uploads/ index_store/ chat_data.db
```

### Getting Help
1. Check logs for error messages
2. Verify all dependencies are installed
3. Test with the provided test suite
4. Check system resources (CPU, RAM, disk)
5. Verify network connectivity to Ollama/external services

## 📄 License & Contributing

This system is designed for production deployment with enterprise-grade reliability and performance. For questions or issues, please check the logs and test suite first.

---

**Ready for Production** ✅ | **Fully Tested** ✅ | **Docker Ready** ✅ 