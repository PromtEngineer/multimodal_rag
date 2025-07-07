#!/usr/bin/env python3
"""
RAG System Dependency Installer
===============================

Automated installer for all system dependencies including:
- Python packages
- Node.js packages
- Ollama models
- System prerequisites

Usage:
    python install_dependencies.py [--skip-models] [--skip-npm]
"""

import subprocess
import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Optional

class DependencyInstaller:
    """Handles installation of all system dependencies."""
    
    def __init__(self, skip_models: bool = False, skip_npm: bool = False):
        self.skip_models = skip_models
        self.skip_npm = skip_npm
        self.errors = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with level indicator."""
        colors = {
            "INFO": "\033[32m",     # Green
            "WARN": "\033[33m",     # Yellow
            "ERROR": "\033[31m",    # Red
            "RESET": "\033[0m"      # Reset
        }
        
        color = colors.get(level, colors["INFO"])
        reset = colors["RESET"]
        print(f"{color}[{level}]{reset} {message}")
    
    def run_command(self, command: List[str], cwd: Optional[str] = None, 
                   timeout: int = 300) -> bool:
        """Run a command and return success status."""
        try:
            self.log(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd,
                check=True,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            self.log(f"Error output: {e.stderr}", "ERROR")
            self.errors.append(f"Command failed: {' '.join(command)}")
            return False
        except subprocess.TimeoutExpired:
            self.log(f"Command timed out after {timeout}s", "ERROR")
            self.errors.append(f"Command timed out: {' '.join(command)}")
            return False
    
    def check_python(self) -> bool:
        """Check Python version and pip availability."""
        self.log("🐍 Checking Python installation...")
        
        try:
            import sys
            version = sys.version_info
            if version.major < 3 or (version.major == 3 and version.minor < 8):
                self.log(f"Python {version.major}.{version.minor} found, but 3.8+ required", "ERROR")
                return False
            
            self.log(f"✅ Python {version.major}.{version.minor}.{version.micro} found")
            
            # Check pip
            result = subprocess.run([sys.executable, "-m", "pip", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.log("pip not found", "ERROR")
                return False
            
            self.log("✅ pip available")
            return True
            
        except Exception as e:
            self.log(f"Python check failed: {e}", "ERROR")
            return False
    
    def check_node(self) -> bool:
        """Check Node.js and npm availability."""
        if self.skip_npm:
            self.log("⏭️  Skipping Node.js check")
            return True
            
        self.log("📦 Checking Node.js installation...")
        
        try:
            # Check Node.js
            result = subprocess.run(["node", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.log("Node.js not found", "ERROR")
                return False
            
            version = result.stdout.strip()
            self.log(f"✅ Node.js {version} found")
            
            # Check npm
            result = subprocess.run(["npm", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.log("npm not found", "ERROR")
                return False
            
            npm_version = result.stdout.strip()
            self.log(f"✅ npm {npm_version} found")
            return True
            
        except FileNotFoundError:
            self.log("Node.js not found in PATH", "ERROR")
            return False
    
    def check_ollama(self) -> bool:
        """Check Ollama installation."""
        if self.skip_models:
            self.log("⏭️  Skipping Ollama check")
            return True
            
        self.log("🤖 Checking Ollama installation...")
        
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.log("Ollama not found", "ERROR")
                self.log("Install from: https://ollama.ai", "ERROR")
                return False
            
            version = result.stdout.strip()
            self.log(f"✅ Ollama {version} found")
            return True
            
        except FileNotFoundError:
            self.log("Ollama not found in PATH", "ERROR")
            self.log("Install from: https://ollama.ai", "ERROR")
            return False
    
    def install_python_deps(self) -> bool:
        """Install Python dependencies."""
        self.log("📦 Installing Python dependencies...")
        
        # Check if requirements.txt exists
        if not Path("requirements.txt").exists():
            self.log("requirements.txt not found", "ERROR")
            return False
        
        # Upgrade pip first
        if not self.run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"]):
            self.log("Failed to upgrade pip", "WARN")
        
        # Install requirements
        success = self.run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        if success:
            self.log("✅ Python dependencies installed")
        else:
            self.log("❌ Failed to install Python dependencies", "ERROR")
        
        return success
    
    def install_node_deps(self) -> bool:
        """Install Node.js dependencies."""
        if self.skip_npm:
            self.log("⏭️  Skipping Node.js dependencies")
            return True
            
        self.log("📦 Installing Node.js dependencies...")
        
        # Check if package.json exists
        if not Path("package.json").exists():
            self.log("package.json not found", "ERROR")
            return False
        
        # Install dependencies
        success = self.run_command(["npm", "install"])
        
        if success:
            self.log("✅ Node.js dependencies installed")
        else:
            self.log("❌ Failed to install Node.js dependencies", "ERROR")
        
        return success
    
    def install_ollama_models(self) -> bool:
        """Install required Ollama models."""
        if self.skip_models:
            self.log("⏭️  Skipping Ollama models")
            return True
            
        self.log("🤖 Installing Ollama models...")
        
        # Required models
        models = [
            "qwen3:8b",           # Generation model
            "qwen3:0.6b",         # Fast model for enrichment
        ]
        
        success = True
        for model in models:
            self.log(f"📥 Pulling {model}...")
            if not self.run_command(["ollama", "pull", model], timeout=600):  # 10 min timeout
                self.log(f"❌ Failed to pull {model}", "ERROR")
                success = False
            else:
                self.log(f"✅ {model} installed")
        
        return success
    
    def create_directories(self) -> bool:
        """Create necessary directories."""
        self.log("📁 Creating directories...")
        
        directories = [
            "logs",
            "lancedb",
            "index_store",
            "index_store/overviews",
            "shared_uploads",
            "backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.log(f"✅ Created {directory}")
        
        return True
    
    def install_all(self) -> bool:
        """Install all dependencies."""
        self.log("🚀 Starting RAG System dependency installation...")
        
        # Check prerequisites
        if not self.check_python():
            return False
        
        if not self.check_node():
            return False
        
        if not self.check_ollama():
            return False
        
        # Install dependencies
        success = True
        
        if not self.install_python_deps():
            success = False
        
        if not self.install_node_deps():
            success = False
        
        if not self.create_directories():
            success = False
        
        if not self.install_ollama_models():
            success = False
        
        # Summary
        if success:
            self.log("🎉 All dependencies installed successfully!")
            self.log("🚀 You can now run: python run_system.py")
        else:
            self.log("❌ Installation completed with errors:", "ERROR")
            for error in self.errors:
                self.log(f"  - {error}", "ERROR")
        
        return success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Install RAG System dependencies')
    parser.add_argument('--skip-models', action='store_true',
                       help='Skip Ollama model installation')
    parser.add_argument('--skip-npm', action='store_true',
                       help='Skip Node.js dependency installation')
    
    args = parser.parse_args()
    
    installer = DependencyInstaller(
        skip_models=args.skip_models,
        skip_npm=args.skip_npm
    )
    
    success = installer.install_all()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 