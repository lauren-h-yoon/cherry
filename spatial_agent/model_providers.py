#!/usr/bin/env python3
"""
model_providers.py - Multi-model VLM provider abstraction

Supports multiple VLM backends for spatial intelligence evaluation:
- Claude (Anthropic API)
- Qwen VL (local via transformers/vLLM)
- OpenAI GPT-4o (OpenAI API)
- Ollama (local deployment)

Usage:
    provider = create_model_provider("qwen", model_name="Qwen/Qwen2.5-VL-7B-Instruct")
    response = provider.generate(image_path, prompt, tools)
"""

import os
import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path


@dataclass
class ToolCall:
    """Represents a tool call from the model."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """Standardized response from any VLM provider."""
    text: str
    tool_calls: List[ToolCall]
    raw_response: Any = None


class VLMProvider(ABC):
    """Abstract base class for VLM providers."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        verbose: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

    @abstractmethod
    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        """
        Generate a response given an image and prompt.

        Args:
            image_path: Path to the image file
            prompt: User prompt/task description
            system_prompt: System instructions
            tools: List of tool definitions

        Returns:
            ModelResponse with text and tool calls
        """
        pass

    @abstractmethod
    def supports_tool_use(self) -> bool:
        """Whether this provider supports native tool use."""
        pass

    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image as base64 string."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _get_image_media_type(self, image_path: str) -> str:
        """Get media type from image path."""
        suffix = Path(image_path).suffix.lower()
        media_types = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return media_types.get(suffix, 'image/png')


class ClaudeProvider(VLMProvider):
    """Claude (Anthropic) VLM provider."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        # Lazy import
        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def supports_tool_use(self) -> bool:
        return True

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        from langchain_core.messages import HumanMessage, SystemMessage

        image_b64 = self._encode_image_base64(image_path)
        media_type = self._get_image_media_type(image_path)

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        ))

        if tools:
            llm_with_tools = self.llm.bind_tools(tools)
            response = llm_with_tools.invoke(messages)
        else:
            response = self.llm.invoke(messages)

        # Extract tool calls
        tool_calls = []
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc['name'],
                    arguments=tc['args']
                ))

        return ModelResponse(
            text=response.content if isinstance(response.content, str) else "",
            tool_calls=tool_calls,
            raw_response=response
        )


class QwenVLProvider(VLMProvider):
    """
    Qwen Vision-Language model provider.

    Supports:
    - Qwen2.5-VL series (7B, 72B)
    - Local inference via transformers
    - vLLM for faster inference
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        use_vllm: bool = False,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.device = device
        self.use_vllm = use_vllm
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Qwen VL model."""
        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self):
        """Load model using transformers."""
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            import torch

            if self.verbose:
                print(f"Loading Qwen VL model: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            # Use AutoModelForVision2Seq to auto-detect correct model architecture
            # Works for both Qwen2-VL and Qwen2.5-VL models
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )

            if self.verbose:
                print(f"Model loaded successfully")

        except ImportError:
            raise ImportError(
                "Qwen VL requires: pip install transformers qwen-vl-utils"
            )

    def _load_vllm(self):
        """Load model using vLLM for faster inference."""
        try:
            from vllm import LLM

            if self.verbose:
                print(f"Loading Qwen VL with vLLM: {self.model_name}")

            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=4096
            )

            if self.verbose:
                print(f"vLLM model loaded successfully")

        except ImportError:
            raise ImportError("vLLM required: pip install vllm")

    def supports_tool_use(self) -> bool:
        # Qwen supports function calling but we'll parse it manually
        return True

    def _build_tool_prompt(self, tools: List[Dict]) -> str:
        """Build tool description prompt for Qwen."""
        if not tools:
            return ""

        tool_desc = "\n\n## Available Tools\n\nYou have access to these tools:\n\n"
        for tool in tools:
            tool_desc += f"### {tool['name']}\n"
            tool_desc += f"{tool.get('description', '')}\n"
            if 'parameters' in tool:
                params = tool['parameters'].get('properties', {})
                required = tool['parameters'].get('required', [])
                tool_desc += "Parameters:\n"
                for param_name, param_info in params.items():
                    req = "(required)" if param_name in required else "(optional)"
                    tool_desc += f"  - {param_name} {req}: {param_info.get('description', '')}\n"
            tool_desc += "\n"

        tool_desc += """
To use a tool, respond with a JSON block like:
```json
{"tool": "tool_name", "arguments": {"param1": value1, "param2": value2}}
```

You can use multiple tools by providing multiple JSON blocks.
After using tools, provide your final answer.
"""
        return tool_desc

    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from model response."""
        import re

        tool_calls = []

        # Find JSON blocks in response
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if 'tool' in data and 'arguments' in data:
                    tool_calls.append(ToolCall(
                        name=data['tool'],
                        arguments=data['arguments']
                    ))
            except json.JSONDecodeError:
                continue

        # Also try to find inline JSON objects
        inline_pattern = r'\{"tool":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}'
        inline_matches = re.findall(inline_pattern, text)

        for tool_name, args_str in inline_matches:
            try:
                args = json.loads(args_str)
                # Avoid duplicates
                if not any(tc.name == tool_name and tc.arguments == args for tc in tool_calls):
                    tool_calls.append(ToolCall(
                        name=tool_name,
                        arguments=args
                    ))
            except json.JSONDecodeError:
                continue

        return tool_calls

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        """Generate response using Qwen VL."""

        if self.use_vllm:
            return self._generate_vllm(image_path, prompt, system_prompt, tools)
        else:
            return self._generate_transformers(image_path, prompt, system_prompt, tools)

    def _generate_transformers(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        """Generate using transformers."""
        from qwen_vl_utils import process_vision_info
        import torch

        # Build full prompt with tools
        tool_prompt = self._build_tool_prompt(tools) if tools else ""
        full_system = system_prompt + tool_prompt

        # Build messages
        messages = [
            {
                "role": "system",
                "content": full_system
            } if full_system else None,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        messages = [m for m in messages if m is not None]

        # Process with Qwen
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
                do_sample=self.temperature > 0
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse tool calls
        tool_calls = self._parse_tool_calls(output_text) if tools else []

        return ModelResponse(
            text=output_text,
            tool_calls=tool_calls,
            raw_response=None
        )

    def _generate_vllm(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        """Generate using vLLM."""
        from vllm import SamplingParams

        # Build full prompt
        tool_prompt = self._build_tool_prompt(tools) if tools else ""
        full_prompt = f"{system_prompt}\n{tool_prompt}\n\n{prompt}" if system_prompt or tool_prompt else prompt

        # vLLM multimodal input
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        outputs = self.model.generate(
            [{
                "prompt": full_prompt,
                "multi_modal_data": {"image": image_path}
            }],
            sampling_params=sampling_params
        )

        output_text = outputs[0].outputs[0].text

        # Parse tool calls
        tool_calls = self._parse_tool_calls(output_text) if tools else []

        return ModelResponse(
            text=output_text,
            tool_calls=tool_calls,
            raw_response=outputs[0]
        )


class OpenAIProvider(VLMProvider):
    """OpenAI GPT-4o VLM provider."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI required: pip install openai")

    def supports_tool_use(self) -> bool:
        return True

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        image_b64 = self._encode_image_base64(image_path)
        media_type = self._get_image_media_type(image_path)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": tool
                }
                for tool in tools
            ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=openai_tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        message = response.choices[0].message

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return ModelResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            raw_response=response
        )


class VLLMServerProvider(VLMProvider):
    """
    vLLM Server provider using OpenAI-compatible API.

    Connects to a vLLM server with native tool calling support:
    vllm serve Qwen/Qwen3-VL-8B-Instruct --enable-auto-tool-choice --tool-call-parser hermes

    This is the recommended way to use Qwen with tool calling.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",  # vLLM doesn't require API key by default
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key

        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key
            )
            if self.verbose:
                print(f"Connected to vLLM server at {base_url}")
                print(f"Model: {model_name}")
        except ImportError:
            raise ImportError("OpenAI client required: pip install openai")

    def supports_tool_use(self) -> bool:
        return True

    def _parse_tool_calls_fallback(self, text: str) -> List[ToolCall]:
        """Fallback parser for tool calls when native parsing fails."""
        import re

        tool_calls = []

        # Pattern for <tool_call> XML format
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        matches = re.findall(tool_call_pattern, text, re.DOTALL)

        for match in matches:
            try:
                # Fix Python tuple syntax -> JSON array syntax
                fixed_json = re.sub(r'\((\d+),\s*(\d+)\)', r'[\1, \2]', match)
                data = json.loads(fixed_json)
                if 'name' in data and 'arguments' in data:
                    tool_calls.append(ToolCall(
                        name=data['name'],
                        arguments=data['arguments']
                    ))
            except json.JSONDecodeError:
                continue

        # Also try JSON code blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, text, re.DOTALL)

        for match in json_matches:
            try:
                fixed_json = re.sub(r'\((\d+),\s*(\d+)\)', r'[\1, \2]', match)
                data = json.loads(fixed_json)
                if 'tool' in data and 'arguments' in data:
                    tool_calls.append(ToolCall(
                        name=data['tool'],
                        arguments=data['arguments']
                    ))
                elif 'name' in data and 'arguments' in data:
                    tool_calls.append(ToolCall(
                        name=data['name'],
                        arguments=data['arguments']
                    ))
            except json.JSONDecodeError:
                continue

        return tool_calls

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        """Generate response using vLLM server with native tool calling."""

        # Encode image
        image_b64 = self._encode_image_base64(image_path)
        media_type = self._get_image_media_type(image_path)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Build user message with image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })

        # Convert tools to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": tool
                }
                for tool in tools
            ]

        # Call vLLM server
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=openai_tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        except Exception as e:
            if self.verbose:
                print(f"Error calling vLLM server: {e}")
            raise

        message = response.choices[0].message

        # Extract tool calls (native support via hermes parser)
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    tool_calls.append(ToolCall(
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))
                except json.JSONDecodeError:
                    # Try to fix Python tuple syntax
                    import re
                    fixed_args = re.sub(r'\((\d+),\s*(\d+)\)', r'[\1, \2]', tc.function.arguments)
                    try:
                        tool_calls.append(ToolCall(
                            name=tc.function.name,
                            arguments=json.loads(fixed_args)
                        ))
                    except json.JSONDecodeError:
                        continue

        # Fallback: parse from text if native parsing didn't find tool calls
        content = message.content or ""
        if not tool_calls and content:
            tool_calls = self._parse_tool_calls_fallback(content)

        return ModelResponse(
            text=content,
            tool_calls=tool_calls,
            raw_response=response
        )


class OllamaProvider(VLMProvider):
    """
    Ollama provider for local VLM inference.

    Supports vision models like llava, bakllava, etc.
    """

    def __init__(
        self,
        model_name: str = "llava",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url

        try:
            import ollama
            self.client = ollama.Client(host=base_url)
        except ImportError:
            raise ImportError("Ollama required: pip install ollama")

    def supports_tool_use(self) -> bool:
        # Ollama doesn't natively support tool use, we'll parse manually
        return True

    def _build_tool_prompt(self, tools: List[Dict]) -> str:
        """Build tool description for prompt injection."""
        if not tools:
            return ""

        tool_desc = "\n\n## Available Tools\n\n"
        for tool in tools:
            tool_desc += f"### {tool['name']}\n"
            tool_desc += f"{tool.get('description', '')}\n"
            if 'parameters' in tool:
                params = tool['parameters'].get('properties', {})
                tool_desc += "Parameters:\n"
                for param_name, param_info in params.items():
                    tool_desc += f"  - {param_name}: {param_info.get('description', '')}\n"
            tool_desc += "\n"

        tool_desc += """
To use a tool, output JSON:
```json
{"tool": "tool_name", "arguments": {"param1": value1}}
```
"""
        return tool_desc

    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from response."""
        import re
        tool_calls = []

        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                if 'tool' in data and 'arguments' in data:
                    tool_calls.append(ToolCall(
                        name=data['tool'],
                        arguments=data['arguments']
                    ))
            except json.JSONDecodeError:
                continue

        return tool_calls

    def generate(
        self,
        image_path: str,
        prompt: str,
        system_prompt: str = "",
        tools: List[Dict] = None
    ) -> ModelResponse:
        import ollama

        # Build prompt with tools
        tool_prompt = self._build_tool_prompt(tools) if tools else ""
        full_prompt = f"{system_prompt}\n{tool_prompt}\n\n{prompt}" if system_prompt or tool_prompt else prompt

        response = self.client.generate(
            model=self.model_name,
            prompt=full_prompt,
            images=[image_path],
            options={
                "temperature": self.temperature
            }
        )

        output_text = response['response']

        # Parse tool calls
        tool_calls = self._parse_tool_calls(output_text) if tools else []

        return ModelResponse(
            text=output_text,
            tool_calls=tool_calls,
            raw_response=response
        )


# ============================================================
# Provider Registry and Factory
# ============================================================

PROVIDER_REGISTRY = {
    # Claude models
    "claude": ClaudeProvider,
    "claude-sonnet": ClaudeProvider,
    "claude-sonnet-4-20250514": ClaudeProvider,
    "claude-opus-4-20250514": ClaudeProvider,

    # Qwen models (local transformers)
    "qwen": QwenVLProvider,
    "qwen-vl": QwenVLProvider,
    "qwen2.5-vl": QwenVLProvider,
    "Qwen/Qwen2.5-VL-7B-Instruct": QwenVLProvider,
    "Qwen/Qwen2.5-VL-72B-Instruct": QwenVLProvider,

    # vLLM server (recommended for Qwen with tool calling)
    "vllm": VLLMServerProvider,
    "vllm-server": VLLMServerProvider,
    "qwen-server": VLLMServerProvider,
    "qwen3-vl": VLLMServerProvider,
    "Qwen/Qwen3-VL-4B-Instruct": VLLMServerProvider,
    "Qwen/Qwen3-VL-8B-Instruct": VLLMServerProvider,
    "Qwen/Qwen3-VL-32B-Instruct": VLLMServerProvider,

    # OpenAI models
    "openai": OpenAIProvider,
    "gpt-4o": OpenAIProvider,
    "gpt-4-vision": OpenAIProvider,

    # Ollama (local)
    "ollama": OllamaProvider,
    "llava": OllamaProvider,
}

# Default model names per provider
DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-20250514",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "vllm": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen-server": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl": "Qwen/Qwen3-VL-8B-Instruct",
    "openai": "gpt-4o",
    "ollama": "llava"
}


def create_model_provider(
    provider: str,
    model_name: Optional[str] = None,
    **kwargs
) -> VLMProvider:
    """
    Create a VLM provider instance.

    Args:
        provider: Provider type ('claude', 'qwen', 'openai', 'ollama')
        model_name: Specific model name (uses default if not specified)
        **kwargs: Additional arguments for the provider

    Returns:
        VLMProvider instance

    Examples:
        # Claude (default)
        provider = create_model_provider("claude")

        # Qwen 7B local
        provider = create_model_provider("qwen", model_name="Qwen/Qwen2.5-VL-7B-Instruct")

        # Qwen with vLLM
        provider = create_model_provider("qwen", use_vllm=True)

        # OpenAI
        provider = create_model_provider("openai", model_name="gpt-4o")

        # Ollama local
        provider = create_model_provider("ollama", model_name="llava")
    """
    provider_lower = provider.lower()

    # Find provider class
    provider_class = None
    for key, cls in PROVIDER_REGISTRY.items():
        if provider_lower == key.lower() or provider_lower in key.lower():
            provider_class = cls
            break

    if provider_class is None:
        available = list(set(PROVIDER_REGISTRY.values()))
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: claude, qwen, openai, ollama"
        )

    # Get default model if not specified
    if model_name is None:
        for key, default in DEFAULT_MODELS.items():
            if key in provider_lower:
                model_name = default
                break

    if model_name is None:
        model_name = DEFAULT_MODELS.get("claude")

    return provider_class(model_name=model_name, **kwargs)


def list_available_providers() -> Dict[str, List[str]]:
    """List available providers and their supported models."""
    return {
        "claude": [
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ],
        "qwen": [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen/Qwen2.5-VL-72B-Instruct"
        ],
        "vllm": [
            "Qwen/Qwen3-VL-8B-Instruct (recommended)",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct",
            "Any VL model served by vLLM with --enable-auto-tool-choice"
        ],
        "openai": [
            "gpt-4o",
            "gpt-4-vision-preview"
        ],
        "ollama": [
            "llava",
            "bakllava",
            "llava-llama3"
        ]
    }
