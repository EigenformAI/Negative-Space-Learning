from abc import ABC
from typing import Dict, NamedTuple, Optional
from pydantic import BaseModel
from dataclasses import dataclass


class OllamaConfig(ABC, BaseModel):
    name: str
    endpoint: str = "localhost:11434"
    model: str
    stream: bool

class QwenConfig(OllamaConfig):
    name: str = "Ollama Qwen"
    model: str = "qwen-cleanup-merged:latest"
    _model_uncensored: str = "qwen-uncensored:latest"
    stream: bool = False

class QwenPeftConfig(BaseModel):
    """Configuration for PEFT-based Qwen model."""
    name: str = "Qwen PEFT"
    base_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    checkpoint_path: str
    device: str = "auto"

class DreamConfig(NamedTuple):
    """Configuration for the Dream API generator."""

    # API connection settings
    base_url: str = "http://34.87.4.35:6969"
    # Allow overriding via DREAM_BASE_URL; default to localhost if unset
    # base_url: str = os.getenv("DREAM_BASE_URL", "http://localhost:6969")
    timeout: int = 120

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.8
    steps: int = 64

    top_k: int = 20

    alg: str = "origin"
    alg_temp: float = 0.3

@dataclass
class VllmConfig:
    """Configuration for vLLM-based models."""
    name: str = "vllm qwen"
    model: str = "qwen2.5-coder:7b-instruct"
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    timeout: int = 60

