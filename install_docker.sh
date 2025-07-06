#!/bin/bash
# install_docker.sh - Docker installation script for RAG System
# Supports macOS, Ubuntu/Debian, CentOS/RHEL

set -e

echo "=== Docker Installation Script for RAG System ==="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
else
    echo "❌ Unsupported operating system"
    exit 1
fi

echo "Detected OS: $OS"

# Install Docker based on OS
case $OS in
    "macos")
        echo "Installing Docker on macOS..."
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install Docker Desktop
        echo "Installing Docker Desktop..."
        brew install --cask docker
        
        echo "✅ Docker Desktop installed. Please start Docker Desktop from Applications."
        echo "After starting Docker Desktop, run: docker --version"
        ;;
        
    "ubuntu"|"debian")
        echo "Installing Docker on Ubuntu/Debian..."
        
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
        curl -fsSL https://download.docker.com/linux/$OS/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        
        # Set up repository
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS \
          $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker Engine
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        
        # Start Docker service
        sudo systemctl enable docker
        sudo systemctl start docker
        
        echo "✅ Docker installed successfully!"
        echo "Please log out and log back in for group changes to take effect."
        echo "Or run: newgrp docker"
        ;;
        
    "centos"|"rhel"|"fedora")
        echo "Installing Docker on CentOS/RHEL/Fedora..."
        
        # Install required packages
        if command -v dnf &> /dev/null; then
            sudo dnf install -y yum-utils
            sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        else
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        fi
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        
        # Start Docker service
        sudo systemctl enable docker
        sudo systemctl start docker
        
        echo "✅ Docker installed successfully!"
        echo "Please log out and log back in for group changes to take effect."
        ;;
        
    *)
        echo "❌ Unsupported OS: $OS"
        echo "Please install Docker manually from https://docs.docker.com/get-docker/"
        exit 1
        ;;
esac

# Verify installation (except macOS which needs manual start)
if [[ "$OS" != "macos" ]]; then
    echo "Verifying Docker installation..."
    sleep 5
    
    if docker --version; then
        echo "✅ Docker version check passed"
    else
        echo "❌ Docker version check failed"
        exit 1
    fi
    
    if docker compose version; then
        echo "✅ Docker Compose version check passed"
    else
        echo "❌ Docker Compose version check failed"
        exit 1
    fi
    
    # Test Docker
    echo "Testing Docker with hello-world..."
    if docker run hello-world; then
        echo "✅ Docker test passed"
    else
        echo "❌ Docker test failed"
        exit 1
    fi
fi

echo ""
echo "=== Docker Installation Complete ==="
echo ""
echo "Next steps:"
echo "1. If on macOS: Start Docker Desktop from Applications"
echo "2. If on Linux: Log out and log back in (or run 'newgrp docker')"
echo "3. Verify installation: docker --version && docker compose version"
echo "4. Run RAG system: cd rag-system && docker compose up -d"
echo "" 