#!/usr/bin/env python3
"""
RAG System Unified Launcher
============================

This script provides a unified way to start, stop, and manage the entire RAG system.
It handles all three components:
1. Backend server (port 8000) - Main API and session management
2. RAG API server (port 8001) - Advanced RAG pipeline
3. Frontend server (port 3000) - Next.js UI

Usage:
    python run_system.py start    # Start all services
    python run_system.py stop     # Stop all services
    python run_system.py restart  # Restart all services
    python run_system.py status   # Check service status
    python run_system.py logs     # Show recent logs
"""

import subprocess
import time
import signal
import sys
import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
import threading
import queue
import atexit
import socket

class ServiceManager:
    def __init__(self):
        self.services = {
            "ollama": {
                "name": "Ollama Server",
                "port": 11434,
                "cmd": ["ollama", "serve"],
                "health_url": "http://localhost:11434/api/tags",
                "cwd": None,
                "process": None,
                "required": True
            },
            "rag_api": {
                "name": "RAG API Server", 
                "port": 8001,
                "cmd": [sys.executable, "-m", "rag_system.api_server"],
                "health_url": "http://localhost:8001/models",
                "cwd": None,
                "process": None,
                "required": True
            },
            "backend": {
                "name": "Backend Server",
                "port": 8000, 
                "cmd": [sys.executable, "server.py"],
                "health_url": "http://localhost:8000/health",
                "cwd": "backend",
                "process": None,
                "required": True
            },
            "frontend": {
                "name": "Frontend Server",
                "port": 3000,
                "cmd": ["npm", "run", "dev"],
                "health_url": "http://localhost:3000",
                "cwd": None,
                "process": None,
                "required": False  # Optional if npm not available
            }
        }
        
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.pid_file = Path("system.pids")
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.stop_all()
        sys.exit(0)

    def cleanup(self):
        """Cleanup function called on exit"""
        # Only cleanup if we actually started services
        if any(service["process"] for service in self.services.values()):
            self.stop_all()

    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use using socket connection test"""
        try:
            # Try to connect to the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result == 0  # Port is in use if connection succeeds
        except Exception:
            # If we can't check, assume port is free
            return False

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available"""
        deps = {}
        
        # Check Ollama
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
            deps["ollama"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            deps["ollama"] = False
            
        # Check Python dependencies
        try:
            import requests
            deps["python_deps"] = True
        except ImportError:
            deps["python_deps"] = False
            
        # Check Node.js/npm
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
            deps["npm"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            deps["npm"] = False
            
        return deps

    def ensure_ollama_models(self):
        """Ensure required Ollama models are available"""
        required_models = ["qwen3:8b", "qwen3:0.6b"]
        
        print("🔍 Checking Ollama models...")
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            available_models = result.stdout
            
            for model in required_models:
                if model not in available_models:
                    print(f"📥 Pulling {model} model...")
                    subprocess.run(["ollama", "pull", model], check=True)
                    print(f"✅ {model} model ready")
                else:
                    print(f"✅ {model} model already available")
                    
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Failed to check/pull models: {e}")

    def start_service(self, service_name: str) -> bool:
        """Start a single service"""
        service = self.services[service_name]
        
        # Check if port is already in use
        if self.is_port_in_use(service["port"]):
            print(f"⚠️ Port {service['port']} already in use, skipping {service['name']}")
            return True
            
        print(f"🔄 Starting {service['name']} on port {service['port']}...")
        
        # Prepare environment
        env = os.environ.copy()
        if service_name == "rag_api":
            env["PYTHONPATH"] = os.getcwd()
            
        # Start process
        log_file = self.log_dir / f"{service_name}.log"
        
        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    service["cmd"],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=service["cwd"],
                    env=env
                )
            
            service["process"] = process
            
            # Wait for service to start
            max_wait = 30 if service_name == "rag_api" else 10
            for i in range(max_wait):
                if self.check_service_health(service_name):
                    print(f"✅ {service['name']} started successfully (PID: {process.pid})")
                    return True
                time.sleep(1)
                
            print(f"❌ {service['name']} failed to start within {max_wait} seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start {service['name']}: {e}")
            return False

    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy"""
        service = self.services[service_name]
        
        # Check HTTP endpoint if available (primary check)
        if service["health_url"]:
            try:
                response = requests.get(service["health_url"], timeout=5)
                return response.status_code == 200
            except:
                return False
        
        # For services without health endpoints, check if port is in use
        return self.is_port_in_use(service["port"])

    def stop_service(self, service_name: str):
        """Stop a single service"""
        service = self.services[service_name]
        
        if service["process"]:
            print(f"🛑 Stopping {service['name']}...")
            try:
                service["process"].terminate()
                service["process"].wait(timeout=10)
                print(f"✅ {service['name']} stopped")
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {service['name']}...")
                service["process"].kill()
                service["process"].wait()
            except Exception as e:
                print(f"❌ Error stopping {service['name']}: {e}")
            finally:
                service["process"] = None

    def start_all(self):
        """Start all services in the correct order"""
        print("🚀 Starting RAG System...")
        
        # Check dependencies
        deps = self.check_dependencies()
        if not deps["ollama"]:
            print("❌ Ollama not found. Please install: https://ollama.ai")
            return False
            
        if not deps["python_deps"]:
            print("❌ Python dependencies missing. Run: pip install -r requirements.txt")
            return False
            
        # Start services in order
        startup_order = ["ollama", "rag_api", "backend", "frontend"]
        
        for service_name in startup_order:
            service = self.services[service_name]
            
            # Skip optional services if dependencies missing
            if not service["required"] and service_name == "frontend" and not deps["npm"]:
                print(f"⚠️ Skipping {service['name']} - npm not available")
                continue
                
            if not self.start_service(service_name):
                if service["required"]:
                    print(f"❌ Failed to start required service: {service['name']}")
                    self.stop_all()
                    return False
                    
        # Special handling for Ollama models
        if self.check_service_health("ollama"):
            self.ensure_ollama_models()
            
        # Save PIDs
        self.save_pids()
        
        print("\n🎉 RAG System Started Successfully!")
        self.print_status()
        return True

    def stop_all(self):
        """Stop all services"""
        print("🛑 Stopping RAG System...")
        
        # Stop in reverse order
        shutdown_order = ["frontend", "backend", "rag_api", "ollama"]
        
        for service_name in shutdown_order:
            self.stop_service(service_name)
            
        # Clean up PID file
        if self.pid_file.exists():
            self.pid_file.unlink()
            
        print("✅ RAG System stopped")

    def restart_all(self):
        """Restart all services"""
        print("🔄 Restarting RAG System...")
        self.stop_all()
        time.sleep(2)
        return self.start_all()

    def get_status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        status = {}
        
        for service_name, service in self.services.items():
            is_healthy = self.check_service_health(service_name)
            is_port_open = self.is_port_in_use(service["port"])
            
            # Get PID safely
            pid = None
            if service["process"]:
                try:
                    pid = service["process"].pid
                    # Check if process is still running
                    if service["process"].poll() is not None:
                        pid = None
                except Exception:
                    pid = None
            
            status[service_name] = {
                "name": service["name"],
                "port": service["port"],
                "healthy": is_healthy,
                "port_open": is_port_open,
                "pid": pid
            }
            
        return status

    def print_status(self):
        """Print formatted status of all services"""
        status = self.get_status()
        
        print("\n📊 Service Status:")
        print("=" * 60)
        
        for service_name, info in status.items():
            status_icon = "✅" if info["healthy"] else "❌"
            port_status = "🟢" if info["port_open"] else "🔴"
            
            if info['pid']:
                pid_info = f"(PID: {info['pid']})"
            elif info["healthy"]:
                pid_info = "(Running externally)"
            else:
                pid_info = "(Not running)"
            
            print(f"{status_icon} {info['name']:<20} {port_status} Port {info['port']:<5} {pid_info}")
            
        print("\n🌐 Access Points:")
        if status["frontend"]["healthy"]:
            print("   • Main Application: http://localhost:3000")
        if status["backend"]["healthy"]:
            print("   • Backend API:      http://localhost:8000/health")
        if status["rag_api"]["healthy"]:
            print("   • RAG API:          http://localhost:8001/models")
        if status["ollama"]["healthy"]:
            print("   • Ollama API:       http://localhost:11434/api/tags")

    def show_logs(self, service_name: Optional[str] = None, lines: int = 50):
        """Show recent logs"""
        if service_name:
            log_file = self.log_dir / f"{service_name}.log"
            if log_file.exists():
                print(f"\n📋 Recent logs for {service_name}:")
                print("=" * 60)
                subprocess.run(["tail", f"-{lines}", str(log_file)])
            else:
                print(f"❌ No log file found for {service_name}")
        else:
            print(f"\n📋 Recent logs (last {lines} lines each):")
            print("=" * 60)
            for service_name in self.services.keys():
                log_file = self.log_dir / f"{service_name}.log"
                if log_file.exists():
                    print(f"\n--- {service_name} ---")
                    subprocess.run(["tail", f"-{lines//4}", str(log_file)])

    def save_pids(self):
        """Save process PIDs to file"""
        pids = {}
        for service_name, service in self.services.items():
            if service["process"]:
                pids[service_name] = service["process"].pid
                
        with open(self.pid_file, "w") as f:
            json.dump(pids, f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_system.py {start|stop|restart|status|logs}")
        sys.exit(1)
        
    command = sys.argv[1].lower()
    manager = ServiceManager()
    
    if command == "start":
        success = manager.start_all()
        if success:
            print("\n💡 Use 'python run_system.py status' to check service health")
            print("💡 Use 'python run_system.py logs' to view recent logs")
            print("💡 Use 'python run_system.py stop' to shut down")
            
            # Keep running to monitor services
            try:
                while True:
                    time.sleep(10)
                    # Check if any required service died
                    for name, service in manager.services.items():
                        if service["required"] and service["process"] and service["process"].poll() is not None:
                            print(f"❌ {service['name']} died unexpectedly")
                            manager.stop_all()
                            sys.exit(1)
            except KeyboardInterrupt:
                pass
        else:
            sys.exit(1)
            
    elif command == "stop":
        manager.stop_all()
        
    elif command == "restart":
        manager.restart_all()
        
    elif command == "status":
        manager.print_status()
        
    elif command == "logs":
        service_name = sys.argv[2] if len(sys.argv) > 2 else None
        manager.show_logs(service_name)
        
    else:
        print(f"Unknown command: {command}")
        print("Available commands: start, stop, restart, status, logs")
        sys.exit(1)

if __name__ == "__main__":
    main() 