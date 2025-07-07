# ⚡ Quick Start Guide - RAG System

_Get up and running in 5 minutes!_

---

## 🚀 Super Quick Start (One Command)

If you have Docker installed:

```bash
# Clone and run the complete setup
git clone https://github.com/your-org/rag-system.git
cd rag-system
./setup_rag_system.sh
```

**That's it!** The script will handle everything automatically.

---

## 📋 Step-by-Step Quick Start

### Step 1: Install Docker (if not installed)

#### macOS:
```bash
# Install Docker Desktop
./install_docker.sh
# Then start Docker Desktop from Applications
```

#### Linux:
```bash
# Install Docker
./install_docker.sh
# Log out and log back in (or run: newgrp docker)
```

### Step 2: Start the System

```bash
# Clone repository
git clone https://github.com/your-org/rag-system.git
cd rag-system

# Start everything
docker compose up -d

# Wait 2-3 minutes for services to start
```

### Step 3: Install AI Models

```bash
# Install required models
docker compose exec ollama ollama pull qwen2.5:7b
docker compose exec ollama ollama pull qwen2.5:0.5b
```

### Step 4: Access the System

Open your browser to: **http://localhost:3000**

---

## 🎯 First Use

### 1. Create a Chat Session
- Click "New Chat" in the interface
- Give your session a name

### 2. Upload Documents
- Click the upload button
- Select PDF files from your computer
- Wait for processing to complete

### 3. Ask Questions
- Type questions about your documents
- Examples:
  - "What is this document about?"
  - "Summarize the key points"
  - "What are the main findings?"

---

## 🔧 Essential Commands

```bash
# Start system
docker compose up -d

# Stop system
docker compose down

# Check status
docker compose ps

# View logs
docker compose logs -f

# Restart specific service
docker compose restart rag-api
```

---

## 🆘 Quick Troubleshooting

### System Not Starting?
```bash
# Check if Docker is running
docker ps

# Check service status
docker compose ps

# View error logs
docker compose logs
```

### Can't Access http://localhost:3000?
```bash
# Check if port is in use
sudo lsof -i :3000

# Check frontend logs
docker compose logs frontend
```

### Models Not Loading?
```bash
# Check Ollama status
docker compose exec ollama ollama list

# Reinstall models
docker compose exec ollama ollama pull qwen2.5:7b
```

### Out of Memory?
```bash
# Check memory usage
docker stats

# Increase Docker memory in Docker Desktop:
# Settings → Resources → Memory → 8GB+
```

---

## 📊 System Status Check

Run this to verify everything is working:

```bash
# Check all services
curl -f http://localhost:3000 && echo "✅ Frontend OK"
curl -f http://localhost:8000/health && echo "✅ Backend OK"
curl -f http://localhost:8001/models && echo "✅ RAG API OK"
curl -f http://localhost:11434/api/tags && echo "✅ Ollama OK"
```

---

## 🎉 Success!

If you see all services running and can access http://localhost:3000, you're ready to go!

### What's Next?
1. **Upload Documents**: Add your PDF files
2. **Ask Questions**: Start querying your documents
3. **Explore Features**: Try different query types
4. **Customize**: Check `Documentation/` for advanced configuration

### Need Help?
- 📖 **Full Documentation**: See `Documentation/` folder
- 🐛 **Troubleshooting**: Check `Documentation/deployment_guide.md`
- 🔧 **Configuration**: See `Documentation/system_overview.md`

---

## 📁 File Structure

```
rag-system/
├── 📄 setup_rag_system.sh     # Complete setup script
├── 📄 install_docker.sh       # Docker installation
├── 📄 docker-compose.yml      # Service configuration
├── 📁 Documentation/          # Full documentation
├── 📁 rag_system/            # Core RAG system
├── 📁 backend/               # API backend
├── 📁 src/                   # Frontend source
└── 📁 shared_uploads/        # Document storage
```

---

**Happy RAG-ing! 🚀** 