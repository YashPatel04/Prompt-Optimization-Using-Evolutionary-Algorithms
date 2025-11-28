import requests
import json
from typing import List, Dict, Optional
import time


class OllamaInterface:
    """
    Interface for Ollama LLMs with stateless inference.
    Disk caching is handled separately by PromptCache.
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:1561",
                 model: str = "gemma3:270m",
                 temperature: float = 0.0,
                 timeout: int = 300,
                 keep_alive: str = "5m"):
        """
        Initialize Ollama interface.
        
        Args:
            base_url: Ollama server URL
            model: Model name
            temperature: Response temperature
            timeout: Request timeout in seconds
            keep_alive: How long to keep model in memory
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.keep_alive = keep_alive
        
        self.api_endpoint = f"{self.base_url}/api/generate"
        self.request_count = 0
        
        # Check connection
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                print(f"[OK] Connected to Ollama at {self.base_url}")
                self._list_available_models()
                return True
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                f"Make sure Ollama is running: ollama serve"
            )
        except Exception as e:
            raise Exception(f"Connection error: {e}")
    
    def _list_available_models(self):
        """Print available models on server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if model_names:
                    print(f"Available models: {', '.join(model_names)}")
        except Exception as e:
            print(f"Could not list models: {e}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from Ollama (stateless).
        No context field - pure stateless inference.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "temperature": self.temperature,
            "keep_alive": self.keep_alive,
            "options": {
                "num_predict": 5,      # Short response for classification
                "num_ctx": 512,        # Minimal context window
                "top_p": 0.9,
                "top_k": 40,
            }
        }
        
        try:
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            self.request_count += 1
            
            return generated_text
        
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Request timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama request failed: {e}")
    
    def get_stats(self) -> Dict:
        """Get usage statistics"""
        return {
            'requests_processed': self.request_count,
            'model': self.model,
            'keep_alive': self.keep_alive,
        }
    
    def __str__(self) -> str:
        return f"OllamaInterface(model={self.model}, keep_alive={self.keep_alive})"