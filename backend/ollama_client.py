# Import the unified OllamaClient from the RAG system
import sys
import os

# Add the rag_system path to import the unified client
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_system'))

from utils.ollama_client import OllamaClient

# Re-export for backward compatibility
__all__ = ['OllamaClient']

def main():
    """Test the unified Ollama client"""
    client = OllamaClient()
    
    # Check if Ollama is running
    if not client.is_ollama_running():
        print("❌ Ollama is not running. Please start Ollama first.")
        print("Install: https://ollama.ai")
        print("Run: ollama serve")
        return
    
    print("✅ Ollama is running!")
    
    # List available models
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Try to use llama3.2, pull if needed
    model_name = "llama3.2"
    if model_name not in [m.split(":")[0] for m in models]:
        print(f"Model {model_name} not found. Pulling...")
        if client.pull_model(model_name):
            print(f"✅ Model {model_name} pulled successfully!")
        else:
            print(f"❌ Failed to pull model {model_name}")
            return
    
    # Test chat
    print("\n🤖 Testing chat...")
    response = client.chat("Hello! Can you tell me a short joke?", model_name)
    print(f"AI: {response}")

if __name__ == "__main__":
    main() 