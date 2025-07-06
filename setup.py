#!/usr/bin/env python3
"""
RAG System Setup Script
========================

Quick setup script to install all dependencies and prepare the system.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_dependency(cmd, name):
    """Check if a dependency is available"""
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print(f"✅ {name} is available")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {name} not found")
        return False

def main():
    print("🚀 RAG System Setup")
    print("=" * 40)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    
    deps_ok = True
    deps_ok &= check_dependency("python --version", "Python")
    deps_ok &= check_dependency("pip --version", "pip")
    deps_ok &= check_dependency("node --version", "Node.js")
    deps_ok &= check_dependency("npm --version", "npm")
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies and try again")
        return False
    
    # Check Ollama separately (optional)
    ollama_available = check_dependency("ollama --version", "Ollama")
    if not ollama_available:
        print("⚠️  Ollama not found. Please install from https://ollama.ai")
        print("   The system will not work without Ollama.")
    
    print("\n📦 Installing dependencies...")
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Install Node dependencies
    if not run_command("npm install", "Installing Node.js dependencies"):
        return False
    
    # Create necessary directories
    print("🔄 Creating directories...")
    directories = [
        "logs",
        "backend/shared_uploads", 
        "index_store/overviews",
        "index_store/bm25",
        "index_store/graph",
        "lancedb"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    print("✅ Directories created")
    
    # Make scripts executable
    if os.name != 'nt':  # Not Windows
        print("🔄 Making scripts executable...")
        run_command("chmod +x run_system.py", "Making run_system.py executable")
        run_command("chmod +x start_system.sh", "Making start_system.sh executable")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start Ollama: ollama serve")
    print("2. Start the system: npm run system:start")
    print("3. Access the web interface: http://localhost:3000")
    print("\n💡 Use 'npm run system:status' to check service health")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 