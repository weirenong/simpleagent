import os
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PollinationsConfig:
    base_url: str = "https://gen.pollinations.ai"
    api_key: Optional[str] = None
    safe: Optional[str] = None


class PollinationsClient:
    def __init__(self, config: PollinationsConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {config.api_key}"
            })
        if config.safe:
            self.session.headers.update({
                "Pollinations-Safe": config.safe
            })

    def chat_completions(self, 
                        messages: List[Dict[str, str]], 
                        model: str = "openai",
                        temperature: float = 0.7,
                        max_tokens: int = 1000,
                        stream: bool = False) -> Dict[str, Any]:
        """Generate text using chat completions"""
        url = f"{self.config.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def text_generation(self, prompt: str, model: str = "openai") -> str:
        """Generate text directly"""
        url = f"{self.config.base_url}/text/{prompt}"
        params = {"model": model}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.text

    def generate_image(self, prompt: str, model: str = "flux") -> str:
        """Generate an image URL"""
        url = f"{self.config.base_url}/image/{prompt}"
        params = {"model": model}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.url

    def generate_audio(self, text: str, voice: str = "nova") -> bytes:
        """Generate audio from text"""
        url = f"{self.config.base_url}/audio/{text}"
        params = {"voice": voice}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.content

    def create_embeddings(self, input_text: str, model: str = "openai-3-small", dimensions: int = 512) -> List[float]:
        """Create embeddings for text"""
        url = f"{self.config.base_url}/v1/embeddings"
        
        payload = {
            "model": model,
            "input": input_text,
            "dimensions": dimensions
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        url = f"{self.config.base_url}/v1/models"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def list_models_whitelisted(self) -> List[str]:
        """Return curated list of whitelisted models"""
        return ["mistral", "qwen-coder", "openai-3-small"]

    def get_account_profile(self) -> Dict[str, Any]:
        """Get user account profile"""
        url = f"{self.config.base_url}/account/profile"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        url = f"{self.config.base_url}/account/balance"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_account_usage(self) -> Dict[str, Any]:
        """Get account usage history"""
        url = f"{self.config.base_url}/account/usage"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
