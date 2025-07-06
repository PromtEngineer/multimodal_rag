# RAG System Deployment Guide

This guide covers all deployment scenarios for the RAG system, from local development to production deployment.

## 🚀 Quick Start (New Users)

### 1. Prerequisites
- Python 3.11+ 
- Node.js 18+
- Ollama ([install here](https://ollama.ai))
- Git

### 2. One-Command Setup
```bash
# Clone and setup everything
git clone <repository-url>
cd rag_system_old
python install_dependencies.py
python run_system.py
```

**That's it!** The system will be available at http://localhost:3000

## 📋 Deployment Options

### Option 1: Native Development (Recommended)
```bash
# Install dependencies
make install

# Start in development mode
make dev
```

**Pros:** Hot reload, easy debugging, full control  
**Cons:** Requires all dependencies locally

### Option 2: Docker Deployment (Production)
```bash
# Build and deploy
make deploy
```

**Pros:** Isolated environment, easy scaling  
**Cons:** Slower iteration, requires Docker

### Option 3: Hybrid (Backend only)
```bash
# Install backend dependencies only
make install-backend-only

# Start without frontend
make dev-no-frontend
```

**Pros:** Lighter setup, API-only deployment  
**Cons:** No web interface

## 🔧 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │    RAG API     │    │     Ollama     │
│   (Next.js)     │◄──►│   (Python)      │◄──►│   (Python)      │◄──►│     (LLM)      │
│   Port 3000     │    │   Port 8000     │    │   Port 8001     │    │   Port 11434   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │                       │
        │                       │                       │                       │
        ▼                       ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Browser   │    │   SQLite DB     │    │   Vector DB     │    │   AI Models     │
│   (User)        │    │   (Sessions)    │    │   (LanceDB)     │    │   (qwen3:8b)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Component Details

### Frontend (Port 3000)
- **Technology**: Next.js 15, React 19, Tailwind CSS
- **Features**: Real-time chat, document upload, session management
- **Development**: Hot reload enabled
- **Production**: Optimized build with SSG

### Backend (Port 8000)
- **Technology**: Python HTTP server
- **Features**: Session management, smart routing, file uploads
- **Database**: SQLite for sessions and metadata
- **API**: RESTful endpoints for frontend communication

### RAG API (Port 8001)
- **Technology**: Python with advanced RAG pipeline
- **Features**: Document indexing, vector search, AI reranking
- **Database**: LanceDB for vector storage
- **Models**: Configurable embedding and generation models

### Ollama (Port 11434)
- **Technology**: Local AI model server
- **Features**: LLM inference, embedding generation
- **Models**: qwen3:8b (generation), qwen3:0.6b (fast tasks)
- **GPU**: Optional acceleration support

## 📊 Resource Requirements

### Minimum (Development)
- **RAM**: 8GB
- **Storage**: 10GB free
- **CPU**: 4 cores
- **Network**: Internet for model downloads

### Recommended (Production)
- **RAM**: 16GB+
- **Storage**: 50GB+ SSD
- **CPU**: 8+ cores
- **GPU**: Optional (NVIDIA recommended)
- **Network**: Stable connection

### Model Storage
- **qwen3:8b**: ~4.5GB
- **qwen3:0.6b**: ~350MB
- **Embeddings**: ~500MB per 1M documents
- **Indexes**: ~100MB per 1M documents

## 🔄 Deployment Workflows

### Local Development
```bash
# Daily workflow
make dev          # Start all services
# ... develop and test ...
make logs         # Monitor issues
make stop         # Stop when done
```

### Staging Testing
```bash
# Test production build locally
make prod
make health       # Verify all services
make monitor      # Check resource usage
```

### Production Deployment
```bash
# Docker deployment
make deploy
make docker-logs  # Monitor startup
make backup       # Create initial backup

# Or native deployment
make prod
make monitor      # Ongoing monitoring
```

### CI/CD Pipeline
```bash
# Automated deployment script
#!/bin/bash
git pull origin main
make clean
make install
make test
make docker-build
make docker-up
make health
```

## 🐳 Docker Configuration

### Development Docker
```yaml
# docker-compose.override.yml
version: '3.8'
services:
  frontend:
    volumes:
      - ./src:/app/src
    command: npm run dev
  
  rag-api:
    volumes:
      - ./rag_system:/app/rag_system
    environment:
      - DEBUG=true
```

### Production Docker
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  frontend:
    restart: always
    environment:
      - NODE_ENV=production
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

### Environment Variables
```bash
# .env file
NODE_ENV=production
OLLAMA_HOST=http://ollama:11434
RAG_API_URL=http://rag-api:8001
NEXT_PUBLIC_API_URL=http://backend:8000

# Security
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,yourdomain.com
```

## 🔍 Monitoring & Observability

### Health Checks
```bash
# System health
make health

# Individual services
curl http://localhost:3000         # Frontend
curl http://localhost:8000/health  # Backend
curl http://localhost:8001/models  # RAG API
curl http://localhost:11434/api/tags # Ollama
```

### Log Monitoring
```bash
# Real-time logs
make logs

# Service-specific logs
tail -f logs/frontend.log
tail -f logs/backend.log
tail -f logs/rag-api.log
tail -f logs/ollama.log

# Error patterns
grep -i error logs/*.log
grep -i "failed\|exception" logs/*.log
```

### Performance Monitoring
```bash
# Resource usage
make monitor

# Detailed metrics
htop
iotop
nvidia-smi  # If GPU available

# Database size
du -sh lancedb/
du -sh backend/chat_data.db
```

### Alerting Setup
```bash
# Simple monitoring script
#!/bin/bash
while true; do
    if ! curl -f http://localhost:3000 >/dev/null 2>&1; then
        echo "ALERT: Frontend down" | mail admin@company.com
    fi
    sleep 60
done
```

## 🔒 Security Considerations

### Network Security
- Use reverse proxy (nginx) for production
- Enable HTTPS with SSL certificates
- Firewall rules for port access
- VPN for remote access

### Data Security
- Local-only processing (no external API calls)
- Encrypted storage for sensitive documents
- Regular backups with encryption
- Access control and authentication

### Model Security
- Verify model checksums
- Use official Ollama models only
- Regular security updates
- Monitor for unusual model behavior

## 🚨 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Find process using port
lsof -i :8000
sudo netstat -tlnp | grep :8000

# Kill process
kill $(lsof -ti:8000)
```

#### Memory Issues
```bash
# Check memory usage
free -h
ps aux --sort=-%mem | head

# Optimize model loading
export OLLAMA_MAX_LOADED_MODELS=1
```

#### Disk Space
```bash
# Check space
df -h

# Clean up
make clean
docker system prune -f
ollama rm unused-model
```

#### Model Loading Errors
```bash
# Verify models
ollama list
ollama pull qwen3:8b

# Check Ollama logs
journalctl -u ollama -f
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG
python run_system.py --mode dev
```

### Recovery Procedures
```bash
# Reset to clean state
make clean
rm -rf lancedb/ index_store/
make install
make dev

# Restore from backup
tar -xzf backups/rag-backup-20241201-120000.tar.gz
```

## 📈 Performance Optimization

### Model Optimization
- Use quantized models for faster inference
- Enable GPU acceleration if available
- Optimize batch sizes for your hardware
- Cache frequently used embeddings

### Database Optimization
- Regular LanceDB compaction
- Index optimization for query patterns
- SQLite VACUUM and ANALYZE
- Partition large datasets

### System Optimization
- SSD storage for databases
- Sufficient RAM to avoid swapping
- CPU affinity for critical processes
- Network optimization for distributed setups

## 🔄 Backup & Recovery

### Automated Backups
```bash
# Daily backup script
#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="backups"
BACKUP_FILE="$BACKUP_DIR/rag-backup-$DATE.tar.gz"

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_FILE \
    backend/chat_data.db \
    lancedb/ \
    index_store/ \
    shared_uploads/

# Keep only last 7 days
find $BACKUP_DIR -name "rag-backup-*.tar.gz" -mtime +7 -delete
```

### Disaster Recovery
1. Stop all services: `make stop`
2. Restore from backup: `tar -xzf backup.tar.gz`
3. Verify data integrity: `make health`
4. Restart services: `make dev`
5. Test functionality: Run sample queries

## 📊 Scaling Strategies

### Horizontal Scaling
- Load balancer for frontend instances
- Database replication for read queries
- Distributed vector storage
- Model serving clusters

### Vertical Scaling
- Increase RAM for larger models
- GPU acceleration for inference
- SSD storage for better I/O
- More CPU cores for parallel processing

### Cloud Deployment
- Docker containers on Kubernetes
- Managed databases (PostgreSQL)
- Object storage for documents
- Auto-scaling groups

## 🤝 Support & Maintenance

### Regular Maintenance
- Weekly: Check logs and performance
- Monthly: Update dependencies and models
- Quarterly: Full system backup and restore test
- Annually: Security audit and optimization

### Getting Help
1. Check logs: `make logs`
2. Review documentation: `Documentation/`
3. Search issues: GitHub repository
4. Community support: Discord/Slack channels

### Contributing
1. Fork repository
2. Create feature branch
3. Test changes: `make test`
4. Submit pull request
5. Update documentation

---

## 📞 Quick Reference

### Essential Commands
```bash
# Setup
python install_dependencies.py
make dev

# Daily operations
make logs
make health
make backup

# Troubleshooting
make stop
make clean
make monitor

# Deployment
make deploy
make docker-logs
```

### Important URLs
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- RAG API: http://localhost:8001
- Ollama: http://localhost:11434

### Support Contacts
- Technical Issues: [GitHub Issues]
- Documentation: `Documentation/`
- Community: [Discord/Slack] 