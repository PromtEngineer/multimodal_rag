import requests
import json
from typing import List, Dict, Any
import base64
from io import BytesIO
from PIL import Image

class OllamaClient:
    """
    An enhanced client for Ollama that now handles image data for VLM models.
    """
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.api_url = f"{host}/api"
        # (Connection check remains the same)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Converts a Pillow Image to a base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate_embedding(self, model: str, text: str) -> List[float]:
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

    def generate_completion(
        self, 
        model: str, 
        prompt: str, 
        format: str = "", 
        images: List[Image.Image] = None
    ) -> Dict[str, Any]:
        """
        Generates a completion, now with optional support for images.

        Args:
            model: The name of the generation model (e.g., 'llava', 'qwen-vl').
            prompt: The text prompt for the model.
            format: The format for the response, e.g., "json".
            images: A list of Pillow Image objects to send to the VLM.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if format == "json":
                payload["format"] = "json"
            
            if images:
                payload["images"] = [self._image_to_base64(img) for img in images]

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

if __name__ == '__main__':
    # This test now requires a VLM model like 'llava' or 'qwen-vl' to be pulled.
    print("Ollama client updated for multimodal (VLM) support.")
    try:
        client = OllamaClient()
        # Create a dummy black image for testing
        dummy_image = Image.new('RGB', (100, 100), 'black')
        
        # Test VLM completion
        vlm_response = client.generate_completion(
            model="llava", # Make sure you have run 'ollama pull llava'
            prompt="What color is this image?",
            images=[dummy_image]
        )
        
        if vlm_response and 'response' in vlm_response:
            print("\n--- VLM Test Response ---")
            print(vlm_response['response'])
        else:
            print("\nFailed to get VLM response. Is 'llava' model pulled and running?")

    except Exception as e:
        print(f"An error occurred: {e}")