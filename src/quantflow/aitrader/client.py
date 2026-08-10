"""Configurable language-model client.

The service depends on a protocol, not a vendor. Swapping models must not require touching
the decision loop, and — more importantly — the loop must be fully testable with no
network, no credentials and no spend. `NullClient` is therefore the default: it returns a
deterministic HOLD, which means an unconfigured deployment does nothing rather than
something arbitrary.

Every client returns plain text. Parsing and validation live in `decision.py`, deliberately
apart from transport: a client that also interpreted its own output would make it
impossible to test the parser against the malformed responses that matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from quantflow.core.config import LLMSettings
from quantflow.core.errors import ConfigurationError, QuantFlowError
from quantflow.core.logging import get_logger

logger = get_logger(__name__)


class LLMError(QuantFlowError):
    """A model call failed. Recoverable: the next cycle will try again."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response."""

    text: str
    model: str
    #: Tokens billed, when the provider reports them. Cost is worth watching on a loop
    #: that runs every few minutes forever.
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMClient(Protocol):
    """The only thing the service needs from a model."""

    @property
    def model(self) -> str:
        """Identifier recorded in the journal alongside every decision."""
        ...

    async def complete(self, *, system: str, user: str) -> Completion:
        """Send a prompt and return the raw response text."""
        ...

    async def aclose(self) -> None:
        """Release any transport resources."""
        ...


class NullClient:
    """Returns a deterministic HOLD. No network, no credentials, no spend.

    The default, so that a half-configured deployment declines to trade rather than doing
    something unpredictable — and so the whole decision loop can be exercised in tests
    without a provider.
    """

    def __init__(self, model: str = "null") -> None:
        self._model = model

    @property
    def model(self) -> str:
        """Identifier recorded in the journal."""
        return self._model

    async def complete(self, *, system: str, user: str) -> Completion:
        """Always HOLD, with a reason that says why."""
        del system, user
        return Completion(
            text=(
                '{"action": "HOLD", "symbol": "BTCUSDT", "confidence": 0.0, '
                '"reason": "no language model is configured; QF_LLM__PROVIDER is null"}'
            ),
            model=self._model,
        )

    async def aclose(self) -> None:
        """Nothing to release."""


class AnthropicClient:
    """Anthropic Messages API."""

    _URL = "https://api.anthropic.com/v1/messages"
    _VERSION = "2023-06-01"

    def __init__(self, settings: LLMSettings) -> None:
        if settings.api_key is None:
            raise ConfigurationError("anthropic provider requires QF_LLM__API_KEY")
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            headers={
                "x-api-key": settings.api_key.get_secret_value(),
                "anthropic-version": self._VERSION,
                "content-type": "application/json",
            },
        )

    @property
    def model(self) -> str:
        """The configured model."""
        return self._settings.model

    async def complete(self, *, system: str, user: str) -> Completion:
        """Call the Messages API.

        Raises:
            LLMError: on any transport or protocol failure.

        """
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        body = await self._post(self._URL, payload)

        blocks = body.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=body.get("model", self._settings.model),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST and return the decoded body.

        Raises:
            LLMError: on transport failure, a non-2xx status, or a non-JSON body.

        """
        try:
            response = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"request failed: {exc}") from exc
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:400]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"response was not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise LLMError(f"expected a JSON object, got {type(body).__name__}")
        return body

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._client.aclose()


class OpenAICompatibleClient:
    """Any OpenAI-compatible chat-completions endpoint.

    Covers OpenAI itself and the many local runtimes that copy its shape, which is why
    `base_url` is configurable.
    """

    def __init__(self, settings: LLMSettings) -> None:
        if settings.api_key is None:
            raise ConfigurationError("openai provider requires QF_LLM__API_KEY")
        self._settings = settings
        base = (settings.base_url or "https://api.openai.com/v1").rstrip("/")
        self._url = f"{base}/chat/completions"
        self._client = httpx.AsyncClient(
            timeout=settings.timeout_seconds,
            headers={
                "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
                "content-type": "application/json",
            },
        )

    @property
    def model(self) -> str:
        """The configured model."""
        return self._settings.model

    async def complete(self, *, system: str, user: str) -> Completion:
        """Call the chat-completions endpoint.

        Raises:
            LLMError: on any transport or protocol failure.

        """
        payload: dict[str, Any] = {
            "model": self._settings.model,
            "max_tokens": self._settings.max_tokens,
            "temperature": self._settings.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = await self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"request failed: {exc}") from exc
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise LLMError(f"HTTP {response.status_code}: {response.text[:400]}")
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(f"response was not JSON: {exc}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise LLMError("response contained no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=body.get("model", self._settings.model),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def aclose(self) -> None:
        """Close the HTTP transport."""
        await self._client.aclose()


def build_client(settings: LLMSettings) -> LLMClient:
    """Construct the configured client.

    Raises:
        ConfigurationError: if the provider name is unknown.

    """
    if settings.provider == "null":
        return NullClient(settings.model)
    if settings.provider == "anthropic":
        return AnthropicClient(settings)
    if settings.provider == "openai":
        return OpenAICompatibleClient(settings)
    raise ConfigurationError(f"unknown LLM provider {settings.provider!r}")


__all__ = [
    "AnthropicClient",
    "Completion",
    "LLMClient",
    "LLMError",
    "NullClient",
    "OpenAICompatibleClient",
    "build_client",
]
