"""
DSPy Configuration Module

Integrates DSPy with the existing RAG system configuration,
providing seamless compatibility with current Ollama models,
LanceDB storage, and pipeline settings.
"""

import os
import dspy
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import existing configurations
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag_system.main import OLLAMA_CONFIG, EXTERNAL_MODELS, PIPELINE_CONFIGS

load_dotenv()

class OllamaDSPy(dspy.LM):
    """
    Custom DSPy Language Model wrapper for Ollama integration
    """
    def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
        self.model = model
        self.host = host
        self.kwargs = kwargs
        super().__init__(f"ollama/{model}", **kwargs)
        
    def basic_request(self, prompt: str, **kwargs) -> str:
        """Make a basic request to Ollama"""
        import requests
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return ""

class DSPyConfig:
    """
    Central configuration for DSPy experiments that integrates with existing RAG system
    """
    
    def __init__(self):
        self.ollama_config = OLLAMA_CONFIG
        self.external_models = EXTERNAL_MODELS
        self.pipeline_configs = PIPELINE_CONFIGS
        
        # DSPy specific configurations
        self.dspy_models = {
            "generation": OllamaDSPy(
                model=self.ollama_config["generation_model"],
                host=self.ollama_config["host"]
            ),
            "enrichment": OllamaDSPy(
                model=self.ollama_config["enrichment_model"], 
                host=self.ollama_config["host"]
            )
        }
        
        # Fallback to OpenAI if available for comparison
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self.dspy_models["openai"] = dspy.OpenAI(
                model="gpt-4o-mini",
                api_key=openai_key,
                max_tokens=1024
            )
        
        # Configure DSPy with primary model
        dspy.settings.configure(lm=self.dspy_models["generation"])
        
    def configure_model(self, model_key: str):
        """Configure DSPy to use a specific model"""
        if model_key in self.dspy_models:
            dspy.settings.configure(lm=self.dspy_models[model_key])
            print(f"✅ DSPy configured with {model_key} model")
        else:
            raise ValueError(f"Model {model_key} not available. Available: {list(self.dspy_models.keys())}")
    
    def get_storage_config(self, mode: str = "default") -> Dict[str, Any]:
        """Get storage configuration for a specific pipeline mode"""
        storage_config = self.pipeline_configs[mode]["storage"].copy()
        
        # Auto-detect actual table name if default doesn't exist
        if storage_config["text_table_name"] == "text_pages_v3":
            import lancedb
            try:
                db_path = storage_config["lancedb_uri"]
                # Adjust path if we're in dspy_experiments
                if db_path == "./lancedb":
                    import os
                    if os.path.basename(os.getcwd()) == "dspy_experiments":
                        db_path = "../lancedb"
                
                db = lancedb.connect(db_path)
                available_tables = db.table_names()
                
                # Find the first non-_lc table (main tables, not lowercased versions)
                text_tables = [t for t in available_tables if t.startswith("text_pages_") and not t.endswith("_lc")]
                
                if text_tables:
                    storage_config["text_table_name"] = text_tables[0]
                    print(f"🔧 Auto-detected table: {text_tables[0]}")
                    
            except Exception as e:
                print(f"⚠️ Could not auto-detect table: {e}")
        
        return storage_config
    
    def get_retrieval_config(self, mode: str = "default") -> Dict[str, Any]:
        """Get retrieval configuration for a specific pipeline mode"""
        return self.pipeline_configs[mode]["retrieval"]

# Global configuration instance
dspy_config = DSPyConfig()

def get_dspy_config() -> DSPyConfig:
    """Get the global DSPy configuration instance"""
    return dspy_config

def configure_dspy_model(model_key: str):
    """Configure DSPy to use a specific model"""
    dspy_config.configure_model(model_key)

# Test configuration setup
def test_dspy_setup():
    """Test DSPy configuration and model connectivity"""
    print("🧪 Testing DSPy Configuration...")
    
    try:
        # Test basic prediction
        predict = dspy.Predict("question -> answer")
        response = predict(question="What is 2+2?")
        print(f"✅ Basic DSPy prediction working: {response.answer}")
        
        # Test with different models if available
        for model_name, model in dspy_config.dspy_models.items():
            try:
                dspy.settings.configure(lm=model)
                response = predict(question="Hello, how are you?")
                print(f"✅ {model_name} model working: {response.answer[:50]}...")
            except Exception as e:
                print(f"❌ {model_name} model failed: {e}")
        
        # Reset to default
        dspy.settings.configure(lm=dspy_config.dspy_models["generation"])
        print("✅ DSPy configuration test completed")
        
    except Exception as e:
        print(f"❌ DSPy configuration test failed: {e}")

if __name__ == "__main__":
    test_dspy_setup() 