# 🐳 Docker Usage Guide - RAG System

_Last updated: 2025-01-02_

This guide provides practical Docker commands and procedures for running the RAG system in containerized environments.

---

## 1. Quick Start Commands

### 1.1 Basic Operations

```bash
# Clone and navigate to repository
git clone https://github.com/your-org/rag-system.git
cd rag-system

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### 1.2 Service Access

Once running, access the system at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **RAG API**: http://localhost:8001
- **Ollama**: http://localhost:11434

---

## 2. Container Management

### 2.1 Starting Services

```bash
# Start all services in background
docker-compose up -d

# Start specific service
docker-compose up -d frontend
docker-compose up -d backend
docker-compose up -d rag-api
docker-compose up -d ollama

# Start with build (after code changes)
docker-compose up -d --build

# Start with logs (foreground)
docker-compose up
```

### 2.2 Stopping Services

```bash
# Stop all services
docker-compose down

# Stop specific service
docker-compose stop frontend

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v

# Force stop and remove everything
docker-compose down -v --remove-orphans
```

### 2.3 Restarting Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart rag-api

# Rebuild and restart
docker-compose up -d --build rag-api
```

---

## 3. Development Workflow

### 3.1 Development Mode

```bash
# Start in development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Enable hot reload (if available)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

### 3.2 Code Changes

```bash
# After frontend changes
docker-compose restart frontend

# After backend changes
docker-compose restart backend

# After RAG system changes
docker-compose restart rag-api

# Rebuild after dependency changes
docker-compose build --no-cache rag-api
docker-compose up -d rag-api
```

### 3.3 Debugging

```bash
# Access container shell
docker-compose exec frontend sh
docker-compose exec backend bash
docker-compose exec rag-api bash
docker-compose exec ollama bash

# Run commands in container
docker-compose exec rag-api python -c "from rag_system.main import get_agent; print('OK')"
docker-compose exec ollama ollama list
```

---

## 4. Logging & Monitoring

### 4.1 Log Management

```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs frontend
docker-compose logs backend
docker-compose logs rag-api
docker-compose logs ollama

# Follow logs in real-time
docker-compose logs -f

# View last N lines
docker-compose logs --tail=100

# View logs with timestamps
docker-compose logs -t

# View logs since specific time
docker-compose logs --since=2h
docker-compose logs --since=2025-01-01T00:00:00
```

### 4.2 System Monitoring

```bash
# Monitor resource usage
docker stats

# Monitor specific containers
docker stats rag-frontend rag-backend rag-api rag-ollama

# Check container health
docker-compose ps
docker inspect rag-frontend --format='{{.State.Health.Status}}'

# View system information
docker system info
docker system df
```

---

## 5. Data Management

### 5.1 Volume Management

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect rag-system_ollama_data

# View volume usage
docker system df -v

# Clean unused volumes
docker volume prune
```

### 5.2 Data Persistence

```bash
# Backup volumes
docker run --rm -v rag-system_ollama_data:/data -v $(pwd):/backup alpine tar czf /backup/ollama_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v rag-system_ollama_data:/data -v $(pwd):/backup alpine tar xzf /backup/ollama_backup.tar.gz -C /data

# Copy files from container
docker cp rag-api:/app/lancedb ./backup/lancedb
docker cp rag-backend:/app/backend/chat_data.db ./backup/
```

### 5.3 Database Operations

```bash
# Backup SQLite database
docker-compose exec backend cp /app/backend/chat_data.db /app/backend/chat_data.db.backup

# Access database
docker-compose exec backend sqlite3 /app/backend/chat_data.db

# Check database tables
docker-compose exec backend sqlite3 /app/backend/chat_data.db ".tables"

# Export database
docker-compose exec backend sqlite3 /app/backend/chat_data.db ".dump" > backup.sql
```

---

## 6. Model Management

### 6.1 Ollama Models

```bash
# List installed models
docker-compose exec ollama ollama list

# Pull new models
docker-compose exec ollama ollama pull qwen2.5:7b
docker-compose exec ollama ollama pull qwen2.5:0.5b
docker-compose exec ollama ollama pull llama3.1:8b

# Remove models
docker-compose exec ollama ollama rm qwen2.5:7b

# Check model info
docker-compose exec ollama ollama show qwen2.5:7b

# Test model
docker-compose exec ollama ollama run qwen2.5:7b "Hello, how are you?"
```

### 6.2 Embedding Models

```bash
# Check current embedding model
docker-compose exec rag-api python -c "
from rag_system.main import get_agent
agent = get_agent('default')
embedder = agent.retrieval_pipeline._get_text_embedder()
print(f'Current model: {embedder.model_name}')
"

# Test embedding generation
docker-compose exec rag-api python -c "
from rag_system.main import get_agent
agent = get_agent('default')
embedder = agent.retrieval_pipeline._get_text_embedder()
embedding = embedder.create_embeddings(['test text'])
print(f'Embedding shape: {embedding.shape}')
"
```

---

## 7. Network & Connectivity

### 7.1 Network Inspection

```bash
# List Docker networks
docker network ls

# Inspect RAG network
docker network inspect rag-system_rag-network

# Check container connectivity
docker-compose exec frontend ping backend
docker-compose exec backend ping rag-api
docker-compose exec rag-api ping ollama
```

### 7.2 Port Management

```bash
# Check port usage
sudo netstat -tulpn | grep -E ":3000|:8000|:8001|:11434"

# Test port connectivity
curl -f http://localhost:3000 || echo "Frontend not accessible"
curl -f http://localhost:8000/health || echo "Backend not accessible"
curl -f http://localhost:8001/models || echo "RAG API not accessible"
curl -f http://localhost:11434/api/tags || echo "Ollama not accessible"
```

---

## 8. Troubleshooting

### 8.1 Common Issues

#### **Port Conflicts**
```bash
# Check what's using ports
sudo lsof -i :3000
sudo lsof -i :8000
sudo lsof -i :8001
sudo lsof -i :11434

# Kill processes using ports
sudo kill -9 $(sudo lsof -t -i:3000)
```

#### **Container Won't Start**
```bash
# Check container logs
docker-compose logs [service-name]

# Check container status
docker-compose ps

# Check resource usage
docker stats --no-stream

# Check Docker daemon
sudo systemctl status docker
```

#### **Out of Memory**
```bash
# Check memory usage
free -h
docker stats --no-stream

# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → Increase limit

# Clean up unused resources
docker system prune -a
```

#### **Model Loading Issues**
```bash
# Check Ollama status
docker-compose exec ollama ollama list

# Check Ollama logs
docker-compose logs ollama

# Restart Ollama
docker-compose restart ollama

# Pull models manually
docker-compose exec ollama ollama pull qwen2.5:7b
```

### 8.2 Debug Commands

```bash
# Complete system health check
#!/bin/bash
echo "=== RAG System Health Check ==="

# Check Docker
docker --version
docker-compose --version

# Check containers
echo "Container Status:"
docker-compose ps

# Check ports
echo "Port Status:"
for port in 3000 8000 8001 11434; do
    if nc -z localhost $port; then
        echo "Port $port: OPEN"
    else
        echo "Port $port: CLOSED"
    fi
done

# Check services
echo "Service Health:"
curl -s -f http://localhost:3000 && echo "Frontend: OK" || echo "Frontend: FAIL"
curl -s -f http://localhost:8000/health && echo "Backend: OK" || echo "Backend: FAIL"
curl -s -f http://localhost:8001/models && echo "RAG API: OK" || echo "RAG API: FAIL"
curl -s -f http://localhost:11434/api/tags && echo "Ollama: OK" || echo "Ollama: FAIL"
```

---

## 9. Performance Optimization

### 9.1 Resource Limits

```yaml
# docker-compose.yml - Add resource limits
services:
  rag-api:
    # ... existing configuration
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### 9.2 Scaling

```bash
# Scale services
docker-compose up -d --scale backend=3
docker-compose up -d --scale rag-api=2

# Check scaled services
docker-compose ps
```

### 9.3 Optimization

```bash
# Clean up unused resources
docker system prune -a

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune
```

---

## 10. Backup & Recovery

### 10.1 Complete Backup

```bash
#!/bin/bash
# backup_docker.sh - Complete Docker backup

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Stop services
docker-compose down

# Backup volumes
docker run --rm -v rag-system_ollama_data:/data -v $(pwd):/backup alpine tar czf /backup/$BACKUP_DIR/ollama_data.tar.gz -C /data .

# Backup bind mounts
cp -r ./backend/chat_data.db "$BACKUP_DIR/"
cp -r ./lancedb "$BACKUP_DIR/"
cp -r ./shared_uploads "$BACKUP_DIR/"
cp -r ./index_store "$BACKUP_DIR/"

# Backup configuration
cp docker-compose.yml "$BACKUP_DIR/"
cp .env "$BACKUP_DIR/"

# Restart services
docker-compose up -d

echo "Backup completed: $BACKUP_DIR"
```

### 10.2 Recovery

```bash
#!/bin/bash
# restore_docker.sh - Docker recovery

BACKUP_DIR="$1"
if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup-directory>"
    exit 1
fi

# Stop services
docker-compose down

# Restore volumes
docker run --rm -v rag-system_ollama_data:/data -v $(pwd):/backup alpine tar xzf /backup/$BACKUP_DIR/ollama_data.tar.gz -C /data

# Restore bind mounts
cp -r "$BACKUP_DIR/chat_data.db" ./backend/
cp -r "$BACKUP_DIR/lancedb" ./
cp -r "$BACKUP_DIR/shared_uploads" ./
cp -r "$BACKUP_DIR/index_store" ./

# Restore configuration
cp "$BACKUP_DIR/docker-compose.yml" ./
cp "$BACKUP_DIR/.env" ./

# Restart services
docker-compose up -d

echo "Recovery completed from: $BACKUP_DIR"
```

---

## 11. Production Deployment

### 11.1 Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  frontend:
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=https://your-domain.com
    restart: unless-stopped
    
  backend:
    environment:
      - NODE_ENV=production
      - CORS_ORIGINS=https://your-domain.com
    restart: unless-stopped
    
  rag-api:
    environment:
      - NODE_ENV=production
      - LOG_LEVEL=info
    restart: unless-stopped
    
  ollama:
    restart: unless-stopped
```

### 11.2 Production Deployment

```bash
# Deploy to production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Set up reverse proxy (nginx)
sudo apt-get install nginx
sudo cp nginx.conf /etc/nginx/sites-available/rag-system
sudo ln -s /etc/nginx/sites-available/rag-system /etc/nginx/sites-enabled/
sudo systemctl reload nginx

# Set up SSL (Let's Encrypt)
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 12. Maintenance Scripts

### 12.1 Daily Maintenance

```bash
#!/bin/bash
# daily_maintenance.sh

# Health check
docker-compose ps

# Check logs for errors
docker-compose logs --since=24h | grep -i error

# Check disk usage
df -h

# Clean up old logs
find ./logs -name "*.log" -mtime +7 -delete
```

### 12.2 Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

# Update images
docker-compose pull

# Clean up unused resources
docker system prune -f

# Backup system
./backup_docker.sh

# Check for updates
git fetch origin
git log HEAD..origin/main --oneline
```

---

This Docker usage guide provides comprehensive commands and procedures for effectively managing the RAG system in containerized environments. For additional support, refer to the deployment guide or system documentation. 