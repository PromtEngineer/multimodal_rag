import requests
import json
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
import httpx, asyncio

class OllamaClient:
    """
    Unified Ollama client with support for:
    - Health checks and model management
    - Chat and completion generation
    - Multimodal (VLM) support with images
    - Async operations and streaming
    - Embedding generation
    """
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.base_url = host
        self.api_url = f"{host}/api"

    # =============================================
    # Health Check and Model Management
    # =============================================
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model if not available"""
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                print(f"Pulling model {model_name}...")
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"Status: {data['status']}")
                        if data.get("status") == "success":
                            return True
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False

    # =============================================
    # Utility Methods
    # =============================================
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    # =============================================
    # Embedding Generation
    # =============================================
    
    def generate_embedding(self, model: str, text: str) -> List[float]:
        """Generate embeddings for text using specified model"""
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            return response.json().get("embedding", [])
        except requests.exceptions.RequestException as e:
            print(f"Error generating embedding: {e}")
            return []

    # =============================================
    # Chat Interface (Backend-style)
    # =============================================
    
    def chat(self, message: str, model: str = "llama3.2", conversation_history: Optional[List[Dict]] = None) -> str:
        """Send a chat message to Ollama with conversation history"""
        if conversation_history is None:
            conversation_history = []
        
        # Add user message to conversation
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"Connection error: {e}"
    
    def chat_stream(self, message: str, model: str = "llama3.2", conversation_history: Optional[List[Dict]] = None):
        """Stream chat response from Ollama"""
        if conversation_history is None:
            conversation_history = []
        
        messages = conversation_history + [{"role": "user", "content": message}]
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.RequestException as e:
            yield f"Connection error: {e}"

    # =============================================
    # Completion Interface (RAG-style with multimodal)
    # =============================================

    def generate_completion(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Generates a completion with optional multimodal support.

        Args:
            model: The name of the generation model (e.g., 'llava', 'qwen-vl').
            prompt: The text prompt for the model.
            format: The format for the response, e.g., "json".
            images: A list of Pillow Image objects to send to the VLM.
            enable_thinking: Optional flag to disable chain-of-thought for Qwen models.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if format:
                payload["format"] = format
            
            if images:
                payload["images"] = [self._image_to_base64(img) for img in images]

            # Optional: disable thinking mode for Qwen3 / DeepSeek models
            if enable_thinking is not None:
                payload["think"] = enable_thinking

            response = requests.post(
                f"{self.api_url}/generate",
                json=payload
            )
            response.raise_for_status()
            response_lines = response.text.strip().split('\n')
            final_response = json.loads(response_lines[-1])
            return final_response

        except requests.exceptions.RequestException as e:
            print(f"Error generating completion: {e}")
            return {}

    # =============================================
    # Async Operations
    # =============================================
    
    async def generate_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        format: str = "",
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """Asynchronous version of generate_completion using httpx."""

        payload = {"model": model, "prompt": prompt, "stream": False}
        if format:
            payload["format"] = format
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]

        if enable_thinking is not None:
            payload["think"] = enable_thinking

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.api_url}/generate", json=payload)
                resp.raise_for_status()
                return json.loads(resp.text.strip().split("\n")[-1])
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            print(f"Async Ollama completion error: {e}")
            return {}

    # =============================================
    # Streaming Operations
    # =============================================
    
    def stream_completion(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
    ):
        """Generator that yields partial response strings as they arrive."""
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]
        if enable_thinking is not None:
            payload["think"] = enable_thinking

        with requests.post(f"{self.api_url}/generate", json=payload, stream=True) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    # Keep-alive newline
                    continue
                try:
                    data = json.loads(raw_line.decode())
                except json.JSONDecodeError:
                    continue
                # The Ollama streaming API sends objects like {"response":"Hi","done":false}
                chunk = data.get("response", "")
                if chunk:
                    yield chunk
                if data.get("done"):
                    break

    async def stream_completion_async(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[Image.Image]] = None,
        enable_thinking: Optional[bool] = None,
        timeout: int = 60,
    ):
        """Async generator that yields partial response strings as they arrive."""
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": True}
        if images:
            payload["images"] = [self._image_to_base64(img) for img in images]
        if enable_thinking is not None:
            payload["think"] = enable_thinking

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", f"{self.api_url}/generate", json=payload) as resp:
                    resp.raise_for_status()
                    async for raw_line in resp.aiter_lines():
                        if not raw_line:
                            continue
                        try:
                            data = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                        if data.get("done"):
                            break
        except (httpx.HTTPError, asyncio.CancelledError) as e:
            print(f"Async streaming error: {e}")
            return


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
    
    # Test completion
    print("\n🤖 Testing completion...")
    completion = client.generate_completion(model_name, "Complete this sentence: The weather today is")
    if completion and 'response' in completion:
        print(f"Completion: {completion['response']}")


if __name__ == '__main__':
    main()