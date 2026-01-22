from .config import (
    DreamConfig,
    QwenConfig,
)
from .Base import Genner
from .Qwen import QwenGenner
from .QwenVllm import QwenVllmGenner
from .Dream import DreamGenner
from openai import OpenAI

__all__ = ["get_genner"]


class BackendException(Exception):
    pass


class OaiBackendException(Exception):
    pass


class ClaudeBackendException(Exception):
    pass


def get_genner(
    backend: str,
    qwen_config: QwenConfig = QwenConfig(),
    dream_config: DreamConfig = DreamConfig(),
    oai_client: OpenAI | None = None,
) -> Genner:
    """
    Get a genner instance based on the backend.

    Args:
        backend (str): The backend to use.
        deepseek_config (DeepseekConfig, optional): The configuration for the Deepseek backend. Defaults to DeepseekConfig().
        oai_config (OAIConfig, optional): The configuration for the OpenAI backend. Defaults to OAIConfig().
        wizard_config (WizardCoderConfig, optional): The configuration for the WizardCoder backend. Defaults to WizardCoderConfig().
        qwen_config (QwenConfig, optional): The configuration for the Qwen backend. Defaults to QwenConfig().
        claude_config (ClaudeConfig, optional): The configuration for the Claude backend. Defaults to ClaudeConfig().
        oai_client (OpenAI | None, optional): The OpenAI client. Defaults to None.
        claude_client (Anthropic | None, optional): The Anthropic client. Defaults to None.

    Raises:
        BackendException: If the backend is not supported.
        OaiBackendException: If the OpenAI client is required for the OAI backend but not provided.
        ClaudeBackendException: If the Anthropic client is required for the Claude backend but not provided.

    Returns:
        Genner: The genner instance.
    """
    available_backends = ["deepseek", "qwen", "qwen-finetuned", "qwen-cleanup-merged", "wizardcoder", "oai", "claude"]

    if backend == "qwen":
        return QwenGenner(qwen_config)
    elif backend == "qwen-uncensored":
        qwen_config.model = "qwen-uncensored:latest"
        return QwenGenner(qwen_config)
    elif backend == "qwen-0c68278e2d-25-01-22":
        qwen_config.model = "qwen2.5-coder-1.5b-instruct-0c68278e2d-25-01-22:latest"
        return QwenGenner(qwen_config)
    elif backend == "qwen-finetuned":
        qwen_config.model = "qwen-finetuned"
        return QwenGenner(qwen_config)
    elif backend == "qwen-cleanup-merged":
        qwen_config.model = "qwen-cleanup-merged"
        return QwenGenner(qwen_config)
    elif backend == 'vllm':
        qwen_config.model = "Your-Model-Name" # change this based on the model you are using
        if not oai_client:
            raise Exception(
                "Using backend 'oai', OpenAI client is required for OAI backend"
            )
        return QwenVllmGenner(oai_client,qwen_config)
    elif backend == "dream":
        return DreamGenner(dream_config)
    
    raise BackendException(
        f"Unsupported backend: {backend}, available backends: {', '.join(available_backends)}"
    )
