import os
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PollinationsConfig:
    base_url: str = "https://gen.pollinations.ai"
    api_key: Optional[str] = None
    safe: Optional[str] = None
    client_id: Optional[str] = "pk_oCsTjaPx4Kj8WEaY"  # For Bring Your Own Pollen
    device_auth_base_url: str = "https://enter.pollinations.ai"


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
        if config.client_id:
            self.session.headers.update({
                "Client-ID": config.client_id
            })

    def set_api_key(self, api_key: str) -> None:
        """Update the API key on the existing session."""
        self.config.api_key = api_key
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}"
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
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

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
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            data = response.json()
            if "data" not in data or len(data["data"]) == 0:
                raise Exception("No embeddings returned from Pollinations API")
            return data["data"][0]["embedding"]
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        url = f"{self.config.base_url}/v1/models"
        
        response = self.session.get(url)
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

    def list_models_whitelisted(self) -> List[str]:
        """Return curated list of whitelisted models"""
        return ["mistral", "qwen-coder", "openai-3-small"]

    def get_account_profile(self) -> Dict[str, Any]:
        """Get user account profile"""
        url = f"{self.config.base_url}/account/profile"
        
        response = self.session.get(url)
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

    def get_account_balance(self) -> Dict[str, Any]:
        """Get account balance"""
        url = f"{self.config.base_url}/account/balance"
        
        response = self.session.get(url)
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

    def get_account_usage(self) -> Dict[str, Any]:
        """Get account usage history"""
        url = f"{self.config.base_url}/account/usage"
        
        response = self.session.get(url)
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

    def authenticate_with_device_flow(self, poll_interval: int = 5) -> Dict[str, Any]:
        """Authenticate using the device flow (Bring Your Own Pollen)"""
        # Step 1: Request device code
        device_data = self.request_device_code()
        
        # Show user instructions
        user_code = device_data.get("user_code")
        verification_uri = device_data.get("verification_uri")
        
        if not user_code or not verification_uri:
            raise Exception("Failed to obtain device code data")
            
        print(f"Please visit {self.config.device_auth_base_url}{verification_uri}")
        print(f"Enter the code: {user_code}")
        print("Waiting for authentication...")
        
        # Step 2: Poll for access token
        device_code = device_data.get("device_code")
        if not device_code:
            raise Exception("Failed to obtain device code")
            
        token_data = self.poll_for_device_token(device_code, poll_interval)
        return token_data

    def request_device_code(self) -> Dict[str, Any]:
        """Request a device code for Bring Your Own Pollen (Device Flow)"""
        url = f"{self.config.device_auth_base_url}/api/device/code"

        payload = {
            "client_id": self.config.client_id,
            "scope": "generate"
        }
        
        response = self.session.post(url, json=payload)
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            # Log the raw response for debugging
            try:
                data = response.json()
                return data
            except ValueError:
                # If JSON parsing fails, return the raw text
                raise Exception(f"Invalid JSON response from Pollinations API: {response.text}")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")

    def poll_for_device_token(self, device_code: str, poll_interval: int = 5) -> Dict[str, Any]:
        """Poll for the user-authorized token (Device Flow)"""
        url = f"{self.config.device_auth_base_url}/api/device/token"
        
        payload = {
            "device_code": device_code
        }
        
        while True:
            response = self.session.post(url, json=payload)
            
            # Check if response is valid JSON before parsing
            try:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError:
                        raise Exception(f"Invalid JSON response from Pollinations API: {response.text}")
                elif response.status_code == 400:
                    try:
                        data = response.json()
                        if data.get("error") == "authorization_pending":
                            time.sleep(poll_interval)
                            continue
                        else:
                            raise Exception(f"Authentication error: {data.get('error', 'Unknown error')}")
                    except ValueError:
                        raise Exception(f"Invalid JSON response from Pollinations API: {response.text}")
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                # Handle HTTP errors
                raise Exception(f"Pollinations API request failed: {e}")

    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information using an access token"""
        url = f"{self.config.device_auth_base_url}/api/device/userinfo"
        
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        
        response = self.session.get(url, headers=headers)
        
        # Check if response is valid JSON before parsing
        try:
            response.raise_for_status()
            return response.json()
        except ValueError as e:
            # Handle case where response is not valid JSON
            error_text = response.text
            raise Exception(f"Invalid JSON response from Pollinations API: {error_text} (Error: {e})")
        except requests.exceptions.RequestException as e:
            # Handle HTTP errors
            raise Exception(f"Pollinations API request failed: {e}")
