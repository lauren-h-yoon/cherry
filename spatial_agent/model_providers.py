"""
model_providers.py — Unified VLM provider interface for Cherry spatial evaluation.

All providers accept a common tool-schema format (as produced by
run_unity_eval._tools_to_schema) and return a VLMResponse containing
the model's text output and any tool calls it made.

Common tool schema format
-------------------------
{
    "name": "tool_name",
    "description": "...",
    "parameters": {
        "type": "object",
        "properties": {...},
        "required": [...],
    }
}

Supported providers
-------------------
  claude   — Anthropic Claude (default: claude-sonnet-4-6)
  openai   — OpenAI GPT-4o   (default: gpt-4o)
  vllm     — OpenAI-compatible vLLM server (default: Qwen/Qwen2-VL-7B-Instruct)
  qwen     — Alias for vllm
  ollama   — Ollama local server (default: qwen2-vl:7b)

Usage
-----
    from spatial_agent import create_model_provider

    provider = create_model_provider("claude")
    response = provider.generate(
        image_path="photos/kitchen.jpg",
        prompt="Reconstruct the scene layout.",
        system_prompt="You are a spatial reasoning agent...",
        tools=[...],
    )
    print(response.text)
    for tc in response.tool_calls:
        print(tc.name, tc.arguments)
"""

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A single tool call made by the model."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class VLMResponse:
    """Response from a VLM provider."""
    text: str
    tool_calls: List[ToolCall] = field(default_factory=list)


# ─── Base class ───────────────────────────────────────────────────────────────

class VLMProvider(ABC):
    """Abstract base class for all VLM providers."""

    model_name: str

    @abstractmethod
    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        tools: List[Dict],
        snapshot_path: Optional[str] = None,
    ) -> VLMResponse:
        """
        Generate a response, optionally invoking tools.

        Parameters
        ----------
        image_path : str
            Path to the scene image.
        prompt : str
            User-facing prompt (may include prior tool results).
        system_prompt : str
            System / instruction prompt for the model.
        tools : list of dict
            Tool schemas in common format (see module docstring).
        snapshot_path : str, optional
            Path to a Unity scene snapshot PNG. When provided, it is injected
            as a second image in the user message (after the original image).

        Returns
        -------
        VLMResponse
            Text output and any tool calls.
        """
        ...

    def _encode_image(self, image_path: str) -> tuple:
        """Return (base64_data, media_type) for an image file."""
        path = Path(image_path)
        ext = path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(ext, "image/jpeg")
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return data, media_type


# ─── Claude (Anthropic) ───────────────────────────────────────────────────────

class ClaudeProvider(VLMProvider):
    """
    Anthropic Claude provider.

    Requires the ANTHROPIC_API_KEY environment variable (or ~/.anthropic key file).
    Default model: claude-sonnet-4-6
    """

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        try:
            import anthropic as _anthropic
            self._anthropic = _anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self._client = self._anthropic.Anthropic()

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        tools: List[Dict],
        snapshot_path: Optional[str] = None,
    ) -> VLMResponse:
        img_data, media_type = self._encode_image(image_path)

        # Convert to Anthropic tool format (uses "input_schema" not "parameters")
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ] if tools else []

        content: List[Dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data,
                },
            },
        ]
        if snapshot_path:
            snap_data, snap_media = self._encode_image(snapshot_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": snap_media,
                    "data": snap_data,
                },
            })
        content.append({"type": "text", "text": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{"role": "user", "content": content}],
        }
        if anthropic_tools:
            request_kwargs["tools"] = anthropic_tools

        response = self._client.messages.create(**request_kwargs)

        text = ""
        tool_calls: List[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(name=block.name, arguments=block.input))

        return VLMResponse(text=text, tool_calls=tool_calls)


# ─── OpenAI ───────────────────────────────────────────────────────────────────

class OpenAIProvider(VLMProvider):
    """
    OpenAI GPT-4o / GPT-4-turbo provider.

    Requires the OPENAI_API_KEY environment variable.
    Default model: gpt-4o
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self._client = _OpenAI()

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        tools: List[Dict],
        snapshot_path: Optional[str] = None,
    ) -> VLMResponse:
        img_data, media_type = self._encode_image(image_path)

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ] if tools else []

        user_content: List[Dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{img_data}",
                    "detail": "high",
                },
            },
        ]
        if snapshot_path:
            snap_data, snap_media = self._encode_image(snapshot_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{snap_media};base64,{snap_data}"},
            })
        user_content.append({"type": "text", "text": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if openai_tools:
            request_kwargs["tools"] = openai_tools

        response = self._client.chat.completions.create(**request_kwargs)

        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls: List[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return VLMResponse(text=text, tool_calls=tool_calls)


# ─── vLLM (OpenAI-compatible server) ─────────────────────────────────────────

class VLLMProvider(VLMProvider):
    """
    OpenAI-compatible vLLM server provider.

    Works with any model served by vLLM, e.g.:
        vllm serve Qwen/Qwen2-VL-7B-Instruct --port 8000

    Default model: Qwen/Qwen2-VL-7B-Instruct
    Default base_url: http://localhost:8000/v1
    """

    DEFAULT_MODEL = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        **kwargs,
    ):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self.base_url = base_url
        self._client = _OpenAI(base_url=base_url, api_key="EMPTY")

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        tools: List[Dict],
        snapshot_path: Optional[str] = None,
    ) -> VLMResponse:
        img_data, media_type = self._encode_image(image_path)

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ] if tools else []

        user_content: List[Dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
        ]
        if snapshot_path:
            snap_data, snap_media = self._encode_image(snapshot_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{snap_media};base64,{snap_data}"},
            })
        user_content.append({"type": "text", "text": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if openai_tools:
            request_kwargs["tools"] = openai_tools

        response = self._client.chat.completions.create(**request_kwargs)

        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls: List[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return VLMResponse(text=text, tool_calls=tool_calls)


# ─── Ollama ───────────────────────────────────────────────────────────────────

class OllamaProvider(VLMProvider):
    """
    Ollama local server provider.

    Works with any multimodal model available in Ollama, e.g.:
        ollama pull qwen2-vl:7b

    Default model: qwen2-vl:7b
    Default base_url: http://localhost:11434/v1
    """

    DEFAULT_MODEL = "qwen2-vl:7b"

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: str = "http://localhost:11434/v1",
        **kwargs,
    ):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )
        self.model_name = model_name or self.DEFAULT_MODEL
        self.base_url = base_url
        self._client = _OpenAI(base_url=base_url, api_key="ollama")

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        tools: List[Dict],
        snapshot_path: Optional[str] = None,
    ) -> VLMResponse:
        img_data, media_type = self._encode_image(image_path)

        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ] if tools else []

        user_content: List[Dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
        ]
        if snapshot_path:
            snap_data, snap_media = self._encode_image(snapshot_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{snap_media};base64,{snap_data}"},
            })
        user_content.append({"type": "text", "text": prompt})

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if openai_tools:
            request_kwargs["tools"] = openai_tools

        response = self._client.chat.completions.create(**request_kwargs)

        msg = response.choices[0].message
        text = msg.content or ""
        tool_calls: List[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

        return VLMResponse(text=text, tool_calls=tool_calls)


# ─── HuggingFace Inference API ────────────────────────────────────────────────

class HuggingFaceProvider(VLMProvider):
    """
    HuggingFace Inference provider via the HF router (router.huggingface.co/v1).

    Uses the OpenAI-compatible HF router endpoint, which supports open-source
    VLMs on free-tier accounts.  Requires a HuggingFace token.

    Default model: Qwen/Qwen3-VL-8B-Instruct

    Usage:
        export HUGGINGFACE_TOKEN=hf_...
        provider = HuggingFaceProvider()
    """

    DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
    HF_ROUTER_URL = "https://router.huggingface.co/v1"

    def __init__(self, model_name: Optional[str] = None, token: Optional[str] = None, **kwargs):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        import os
        self.model_name = model_name or self.DEFAULT_MODEL
        resolved_token = token or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if not resolved_token:
            raise ValueError(
                "HuggingFace token required. Set HUGGINGFACE_TOKEN env var or pass token= argument."
            )
        self._client = _OpenAI(base_url=self.HF_ROUTER_URL, api_key=resolved_token)

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str,
        tools: List[Dict],
        snapshot_path: Optional[str] = None,
    ) -> VLMResponse:
        img_data, media_type = self._encode_image(image_path)

        hf_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ] if tools else []

        user_content: List[Dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{img_data}"}},
        ]
        if snapshot_path:
            snap_data, snap_media = self._encode_image(snapshot_path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{snap_media};base64,{snap_data}"},
            })
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        request_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 2048,
        }
        if hf_tools:
            request_kwargs["tools"] = hf_tools

        try:
            response = self._client.chat.completions.create(**request_kwargs)
            msg = response.choices[0].message
            text = msg.content or ""
            tool_calls: List[ToolCall] = []

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except (json.JSONDecodeError, AttributeError):
                        args = {}
                    tool_calls.append(ToolCall(name=tc.function.name, arguments=args))

            # If model returned text but no tool calls, try to parse JSON place_object calls
            if not tool_calls and text:
                tool_calls = self._parse_tool_calls_from_text(text)

            return VLMResponse(text=text, tool_calls=tool_calls)

        except Exception as exc:
            # Tool calling not supported — retry with JSON-format prompt
            if "tool" in str(exc).lower() or "500" in str(exc):
                return self._generate_json_fallback(messages, image_path, prompt, system_prompt)
            raise

    def _generate_json_fallback(self, messages, image_path, prompt, system_prompt) -> VLMResponse:
        """
        Fallback: ask the model to output place_object calls as JSON array.
        Parses the response and returns synthetic ToolCall objects.
        """
        json_prompt = (
            prompt + "\n\n"
            "Output ONLY a JSON array of place_object calls, no other text. Format:\n"
            '[{"label":"chair","x":-3,"y":0.5,"z":5}, {"label":"table","x":0,"y":0.5,"z":4}, ...]\n'
            "Include at least 5 objects. Use the coordinate system described above."
        )
        fallback_messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": messages[1]["content"][0]["image_url"]["url"]},
                    },
                    {"type": "text", "text": json_prompt},
                ],
            },
        ]
        response = self._client.chat_completion(messages=fallback_messages, max_tokens=1024)
        text = response.choices[0].message.content or ""
        return VLMResponse(text=text, tool_calls=self._parse_tool_calls_from_text(text))

    def _parse_tool_calls_from_text(self, text: str) -> List[ToolCall]:
        """Extract place_object calls from JSON array or Python-style function calls in text."""
        import re
        tool_calls = []

        # 1. Try JSON array format: [{"label":..,"x":..,"y":..,"z":..}, ...]
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
                for item in items:
                    if isinstance(item, dict) and "label" in item:
                        tool_calls.append(ToolCall(
                            name="place_object",
                            arguments={
                                "label": str(item.get("label", "object")),
                                "x": float(item.get("x", 0)),
                                "y": float(item.get("y", 0.5)),
                                "z": float(item.get("z", 5)),
                            }
                        ))
                if tool_calls:
                    return tool_calls
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Try Python function call format: place_object("label", x=val, y=val, z=val, ...)
        py_pattern = re.compile(
            r'place_object\(\s*["\']([^"\']+)["\']\s*,\s*'
            r'(?:x\s*=\s*)?([-\d.]+)\s*,\s*'
            r'(?:y\s*=\s*)?([-\d.]+)\s*,\s*'
            r'(?:z\s*=\s*)?([-\d.]+)'
            r'(?:\s*,\s*(?:scale\s*=\s*)?([-\d.]+))?'
            r'(?:\s*,\s*(?:shape\s*=\s*)["\']?(\w+)["\']?)?',
            re.IGNORECASE,
        )
        for m in py_pattern.finditer(text):
            label, x, y, z = m.group(1), m.group(2), m.group(3), m.group(4)
            scale = float(m.group(5)) if m.group(5) else 1.0
            shape = m.group(6) if m.group(6) else "sphere"
            tool_calls.append(ToolCall(
                name="place_object",
                arguments={
                    "label": label,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "scale": scale,
                    "shape": shape,
                }
            ))

        return tool_calls


# ─── Factory ──────────────────────────────────────────────────────────────────

_PROVIDER_MAP = {
    "claude": ClaudeProvider,
    "openai": OpenAIProvider,
    "huggingface": HuggingFaceProvider,
    "hf": HuggingFaceProvider,
    "vllm": VLLMProvider,
    "qwen": VLLMProvider,    # Qwen served via vLLM
    "ollama": OllamaProvider,
}


def create_model_provider(
    provider: str,
    model_name: Optional[str] = None,
    **kwargs,
) -> VLMProvider:
    """
    Instantiate a VLMProvider by name.

    Parameters
    ----------
    provider : str
        One of: "claude", "openai", "vllm", "qwen", "ollama".
    model_name : str, optional
        Model name / ID. Uses each provider's default if omitted.
    **kwargs
        Extra keyword arguments forwarded to the provider constructor.
        Useful kwargs:
          - base_url (vllm/ollama): server URL

    Returns
    -------
    VLMProvider
    """
    if provider not in _PROVIDER_MAP:
        valid = list(_PROVIDER_MAP.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. Valid options: {valid}"
        )
    cls = _PROVIDER_MAP[provider]
    return cls(model_name=model_name, **kwargs)
