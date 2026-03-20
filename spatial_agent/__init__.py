"""
spatial_agent — VLM provider abstraction for Cherry spatial evaluation.

Exposes:
  VLMProvider    : abstract base class
  VLMResponse    : response dataclass
  ToolCall       : tool-call dataclass
  ClaudeProvider : Anthropic Claude
  OpenAIProvider : OpenAI GPT-4o
  VLLMProvider   : vLLM OpenAI-compatible server (Qwen-VL, LLaVA, …)
  OllamaProvider : Ollama local server
  create_model_provider : factory function
"""

from .model_providers import (
    VLMProvider,
    VLMResponse,
    ToolCall,
    ClaudeProvider,
    OpenAIProvider,
    HuggingFaceProvider,
    VLLMProvider,
    OllamaProvider,
    create_model_provider,
)

__all__ = [
    "VLMProvider",
    "VLMResponse",
    "ToolCall",
    "ClaudeProvider",
    "OpenAIProvider",
    "HuggingFaceProvider",
    "VLLMProvider",
    "OllamaProvider",
    "create_model_provider",
]
