"""
DeepSeek API client using OpenAI-compatible SDK.
Supports streaming, thinking mode, error handling, and timeout.
"""
from typing import List, Dict, Any, Optional, Generator
import time

from openai import OpenAI
from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError

from .utils import logger
import config


class DeepSeekClient:
    """OpenAI-compatible client for DeepSeek API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.model = model or config.DEEPSEEK_MODEL
        self.base_url = base_url or config.DEEPSEEK_BASE_URL
        self.timeout = config.DEEPSEEK_TIMEOUT
        self.max_retries = config.DEEPSEEK_MAX_RETRIES

        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set — API calls will fail")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def check_connectivity(self) -> Dict[str, Any]:
        """Test API connectivity. Returns status dict."""
        try:
            self.client.models.list()
            return {"status": "ok", "model": self.model, "base_url": self.base_url}
        except Exception as e:
            return {"status": "error", "error": str(e), "model": self.model}

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        """Core API call with error handling."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response
        except APIConnectionError as e:
            logger.error("API connection error: %s", e)
            raise ConnectionError(f"无法连接到 DeepSeek API：{e}") from e
        except APITimeoutError as e:
            logger.error("API timeout: %s", e)
            raise TimeoutError(f"API 请求超时（{self.timeout}s）：{e}") from e
        except RateLimitError as e:
            logger.error("API rate limit: %s", e)
            raise RuntimeError(f"API 频率限制，请稍后重试：{e}") from e
        except APIError as e:
            logger.error("API error: %s", e)
            raise RuntimeError(f"API 错误：{e}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Chat with streaming. Yields content chunks."""
        response = self._call_api(
            messages=messages,
            stream=stream,
            reasoning_effort=reasoning_effort,
        )

        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    def chat_sync(
        self,
        messages: List[Dict[str, str]],
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Chat without streaming. Returns full response."""
        response = self._call_api(
            messages=messages,
            stream=False,
            reasoning_effort=reasoning_effort,
        )
        return response.choices[0].message.content or ""

    def ask(
        self,
        system_prompt: str,
        user_prompt: str,
        mode: str = "qa",
        stream: bool = True,
    ):
        """Convenience method for the three modes.

        Args:
            system_prompt: System-level instruction.
            user_prompt: User message with context and query.
            mode: 'qa', 'teaching', or 'solving'.
            stream: Whether to stream responses.

        Returns:
            Generator yielding text chunks (stream=True) or full string.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        reasoning = None
        if mode == "solving":
            reasoning = "high"

        if stream:
            return self.chat(messages, stream=True, reasoning_effort=reasoning)
        else:
            return self.chat_sync(messages, reasoning_effort=reasoning)
