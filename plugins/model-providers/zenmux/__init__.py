"""ZenMux provider profile.

ZenMux (https://zenmux.ai) is a multi-provider model router exposing an
OpenAI-compatible Chat Completions endpoint at ``https://zenmux.ai/api/v1``.
Models are addressed with a ``<provider>/<model>`` slug, e.g.
``openai/gpt-5`` or ``anthropic/claude-sonnet-4.5``.

The profile is fully declarative — the standard ``chat_completions``
transport handles everything. ZenMux is OpenAI-compatible with two notable
quirks documented upstream:

* ``max_tokens`` is NOT supported — the API expects ``max_completion_tokens``
  instead. The transport only auto-renames this kwarg for OpenAI-family
  slugs (``gpt-*``/``o1``/``o3``/``o4`` or the OpenAI/Azure/Copilot hosts);
  ``ProviderProfile`` has no hook to rename it declaratively. So a user who
  explicitly sets ``max_tokens`` on a non-OpenAI ZenMux slug (e.g.
  ``anthropic/*``, ``google/*``) will get an HTTP 400. The default is unset,
  so this is latent — leave ``max_tokens`` unset on ZenMux to be safe.
* Reasoning models (e.g. Claude Opus with reasoning enabled) require the
  previous turn's ``reasoning_details`` to be echoed back in full during
  tool-calling. Hermes' history replay already round-trips provider reasoning
  fields, so no override is needed here.
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile

zenmux = ProviderProfile(
    name="zenmux",
    aliases=(),
    display_name="ZenMux",
    description="ZenMux — multi-provider model router (OpenAI-compatible)",
    signup_url="https://zenmux.ai/",
    env_vars=("ZENMUX_API_KEY", "ZENMUX_BASE_URL"),
    base_url="https://zenmux.ai/api/v1",
    # ZenMux routes to many multimodal upstreams via OpenAI Chat Completions,
    # which accepts image content inside tool-result messages natively.
    supports_vision=True,
    # Shown in the /model picker only when the live /api/v1/models fetch
    # fails. Verified present in the live catalog.
    fallback_models=(
        "openai/gpt-5",
        "anthropic/claude-sonnet-4.5",
        "anthropic/claude-opus-4.8",
    ),
)

register_provider(zenmux)
