# ollama.py
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Generator, Iterable


@dataclass
class OllamaConfig:
    model: str = "qwen3:latest"
    host: str = "http://localhost:11434"
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 120


class OllamaClient:
    """
    Lightweight Ollama client for SimpleAgent-style projects.

    Supports:
    - normal prompt completion
    - chat-style messages
    - streaming responses
    - basic model listing
    - clean error handling
    """

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    # -----------------------------
    # Public API
    # -----------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str | Generator[str, None, None]:
        """
        Send a plain text prompt to Ollama.

        Use this when your agent has already built one final prompt string.
        """

        payload = {
            "model": model or self.config.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": (
                    self.config.temperature if temperature is None else temperature
                ),
                "top_p": self.config.top_p if top_p is None else top_p,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        if stream:
            return self._stream_generate(payload)

        data = self._post_json("/api/generate", payload)
        return str(data.get("response", "")).strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        stream: bool = False,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str | Generator[str, None, None]:
        """
        Send chat-style messages to Ollama.

        Message format:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
        """

        payload = {
            "model": model or self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": (
                    self.config.temperature if temperature is None else temperature
                ),
                "top_p": self.config.top_p if top_p is None else top_p,
            },
        }

        if stream:
            return self._stream_chat(payload)

        data = self._post_json("/api/chat", payload)
        message = data.get("message", {})

        thinking = message.get("thinking", "") or message.get("reasoning", "")
        content = message.get("content", "")

        if thinking:
            return f"<think>{thinking}</think>{content}".strip()

        return str(content).strip()

    def embed(
            self,
            text: str | list[str],
            model: str = "ordis/jina-embeddings-v2-base-code:latest",
    ) -> list[float] | list[list[float]]:
        """
        Generate embeddings using Ollama's /api/embed endpoint.

        Accepts either a single string or a list of strings. Returns a single
        embedding for string input, or a list of embeddings for list input.
        """

        payload = {
            "model": model,
            "input": text,
        }

        data = self._post_json("/api/embed", payload)
        embeddings = data.get("embeddings", [])

        if isinstance(text, str):
            if not embeddings:
                return []
            return embeddings[0]

        return embeddings

    def list_models(self) -> list[str]:
        """
        Return installed Ollama model names.
        """

        data = self._get_json("/api/tags")
        models = data.get("models", [])
        return [str(item.get("name", "")) for item in models if item.get("name")]

    def is_available(self) -> bool:
        """
        Check whether Ollama is reachable.
        """

        try:
            self.list_models()
            return True
        except Exception:
            return False

    # -----------------------------
    # Streaming helpers
    # -----------------------------

    def _stream_generate(self, payload: dict) -> Generator[str, None, None]:
        for data in self._post_stream("/api/generate", payload):
            chunk = data.get("response", "")
            if chunk:
                yield str(chunk)

    def _stream_chat(self, payload: dict) -> Generator[str, None, None]:
        for data in self._post_stream("/api/chat", payload):
            message = data.get("message", {})
            thinking = message.get("thinking", "") or message.get("reasoning", "")
            if thinking:
                yield f"<think>{thinking}</think>"
            chunk = message.get("content", "")
            if chunk:
                yield str(chunk)

    # -----------------------------
    # HTTP helpers
    # -----------------------------

    def _url(self, endpoint: str) -> str:
        return f"{self.config.host.rstrip('/')}{endpoint}"

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self._url(endpoint),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
            ) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running with: ollama serve"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc

    def _post_stream(self, endpoint: str, payload: dict) -> Iterable[dict]:
        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            self._url(endpoint),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
            ) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "error" in data:
                        raise RuntimeError(f"Ollama error: {data['error']}")

                    yield data

                    if data.get("done") is True:
                        break

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running with: ollama serve"
            ) from exc

    def _get_json(self, endpoint: str) -> dict:
        request = urllib.request.Request(
            self._url(endpoint),
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout,
            ) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)

        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. "
                "Make sure Ollama is running with: ollama serve"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON.") from exc