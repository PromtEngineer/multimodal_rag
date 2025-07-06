# Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system with multimodal capabilities, featuring intelligent routing, contextual enrichment, and a modern web interface.

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** with pip
2. **Node.js 18+** with npm
3. **Ollama** - Install from [https://ollama.ai](https://ollama.ai)

### One-Command Setup

```bash
# Clone and setup everything
git clone <repository-url>
cd rag_system_old

# Install all dependencies
npm run setup

# Start the entire system
npm run system:start
```

The system will automatically:
- ✅ Start Ollama server
- ✅ Pull required models (qwen3:8b, qwen3:0.6b)
- ✅ Start RAG API server (port 8001)
- ✅ Start Backend server (port 8000)  
- ✅ Start Frontend server (port 3000)

**Access your RAG system at: http://localhost:3000**

### System Management

```bash
# Check system status
npm run system:status

# View logs
npm run system:logs

# Restart system
npm run system:restart

# Stop system
npm run system:stop
```

### Alternative Python Interface

You can also use the Python script directly:

```bash
# Start system
python run_system.py start

# Check status
python run_system.py status

# View logs (all services)
python run_system.py logs

# View logs (specific service)
python run_system.py logs backend

# Stop system
python run_system.py stop
```

## 🏗️ Architecture

The system consists of four main components:

1. **Ollama Server** (port 11434) - Local LLM inference
2. **RAG API Server** (port 8001) - Advanced RAG pipeline with multimodal support
3. **Backend Server** (port 8000) - Session management, routing, and database
4. **Frontend Server** (port 3000) - Next.js web interface

## 📊 Service Health Monitoring

The unified launcher includes built-in health monitoring:

- **Automatic dependency checking** (Ollama, Python packages, npm)
- **Port conflict detection** and resolution
- **Service health monitoring** with HTTP endpoint checks
- **Automatic model pulling** for required Ollama models
- **Graceful shutdown** handling with proper cleanup
- **Process monitoring** with automatic restart on failure

## 🔧 Configuration

### Required Ollama Models

The system automatically pulls these models on first start:
- `qwen3:8b` - Main generation model
- `qwen3:0.6b` - Fast model for routing and enrichment

### Port Configuration

| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 3000 | Web interface |
| Backend | 8000 | API and session management |
| RAG API | 8001 | Advanced RAG pipeline |
| Ollama | 11434 | LLM inference |

## 📝 Logs and Debugging

Logs are automatically saved to the `logs/` directory:

```
logs/
├── frontend.log    # Next.js frontend logs
├── backend.log     # Backend API logs  
├── rag-api.log     # RAG pipeline logs
└── ollama.log      # Ollama server logs
```

View real-time logs:
```bash
# All logs
tail -f logs/*.log

# Specific service
tail -f logs/rag-api.log
```

## 🛠️ Development

### Manual Setup (for development)

If you prefer to start services individually:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start RAG API
python -m rag_system.api_server

# Terminal 3: Start Backend
cd backend && python server.py

# Terminal 4: Start Frontend  
npm run dev
```

### Adding New Services

To add a new service to the unified launcher, edit `run_system.py` and add to the `services` dictionary:

```python
"new_service": {
    "name": "New Service",
    "port": 8002,
    "cmd": ["python", "new_service.py"],
    "health_url": "http://localhost:8002/health",
    "cwd": None,
    "process": None,
    "required": True
}
```

## 🔍 Troubleshooting

### Common Issues

1. **Port already in use**: The launcher automatically detects and skips services on busy ports
2. **Ollama not found**: Install from [https://ollama.ai](https://ollama.ai)
3. **Models not available**: The launcher automatically pulls required models
4. **Python dependencies**: Run `pip install -r requirements.txt`
5. **Node dependencies**: Run `npm install`

### Health Check Commands

```bash
# Check if all services are responding
python run_system.py status

# Test individual endpoints
curl http://localhost:3000        # Frontend
curl http://localhost:8000/health # Backend  
curl http://localhost:8001/models # RAG API
curl http://localhost:11434/api/tags # Ollama
```

## 📚 Documentation

Comprehensive documentation is available in the `Documentation/` folder:

- `api_reference.md` - Complete API documentation
- `architecture_overview.md` - System architecture overview
- `retrieval_pipeline.md` - Detailed retrieval implementation
- `indexing_pipeline.md` - Document processing pipeline
- `react_agent.md` - ReAct agent implementation
- `triage_system.md` - Intelligent routing system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `npm run system:start`
5. Submit a pull request

## 📄 License

[Your license here]
