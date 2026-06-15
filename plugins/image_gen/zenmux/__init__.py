"""ZenMux image generation backend.

Exposes ZenMux's Vertex-AI-style image endpoint as an
:class:`ImageGenProvider`. ZenMux fronts several upstream image models
(OpenAI ``gpt-image``, Google Imagen, Flux, Kling, ...) behind a single
Vertex-compatible ``:predict`` route.

Endpoint:
    POST https://zenmux.ai/api/vertex-ai/v1/publishers/{provider}/models/{model}:predict

The ``model`` is addressed as ``<provider>/<model>`` (e.g.
``openai/gpt-image-2``); the slug is split into the URL path. Auth is a
single ``ZENMUX_API_KEY`` bearer token. The response carries the image
inline as ``predictions[0].bytesBase64Encoded`` which we materialise into
the shared image cache.

Selection precedence (first hit wins):
1. ``ZENMUX_IMAGE_MODEL`` env var
2. ``image_gen.zenmux.model`` in ``config.yaml``
3. :data:`DEFAULT_MODEL`

Base URL: the Vertex media base differs from the chat base, so it has its
own override ``ZENMUX_VERTEX_BASE_URL`` (NOT the chat profile's
``ZENMUX_BASE_URL``, which points at ``/api/v1``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)
from plugins.plugin_utils import SingletonSlot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://zenmux.ai/api/vertex-ai/v1"

_MODELS: Dict[str, Dict[str, Any]] = {
    "openai/gpt-image-2": {
        "display": "OpenAI GPT Image 2",
        "speed": "~10-20s",
        "strengths": "High-quality general-purpose text-to-image.",
    },
    "google/imagen-4.0-generate-001": {
        "display": "Google Imagen 4",
        "speed": "~5-15s",
        "strengths": "Photorealistic; strong prompt adherence.",
    },
}

DEFAULT_MODEL = "openai/gpt-image-2"

# Our unified aspect_ratio vocabulary -> Vertex-native aspect ratios.
_ASPECT_MAP = {
    "landscape": "16:9",
    "square": "1:1",
    "portrait": "9:16",
}

# mimeType -> file extension for the cached image.
_MIME_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/webp": "webp",
    "image/gif": "gif",
}


# ---------------------------------------------------------------------------
# Config / credentials
# ---------------------------------------------------------------------------


def _load_zenmux_config() -> Dict[str, Any]:
    """Read ``image_gen.zenmux`` from config.yaml (best-effort)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        zen = section.get("zenmux") if isinstance(section, dict) else None
        return zen if isinstance(zen, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen.zenmux config: %s", exc)
        return {}


def _resolve_api_key() -> str:
    """Resolve the ZenMux API key from env, falling back to config."""
    key = os.environ.get("ZENMUX_API_KEY", "").strip()
    if key:
        return key
    cfg = _load_zenmux_config()
    candidate = cfg.get("api_key")
    return candidate.strip() if isinstance(candidate, str) else ""


def _resolve_base_url() -> str:
    """Resolve the Vertex media base URL.

    Uses ``ZENMUX_VERTEX_BASE_URL`` — deliberately distinct from the chat
    profile's ``ZENMUX_BASE_URL`` (``/api/v1``), since the Vertex media base
    is ``/api/vertex-ai/v1``.
    """
    base = os.environ.get("ZENMUX_VERTEX_BASE_URL", "").strip()
    if not base:
        cfg = _load_zenmux_config()
        candidate = cfg.get("base_url")
        base = candidate.strip() if isinstance(candidate, str) else ""
    return (base or DEFAULT_BASE_URL).rstrip("/")


def _resolve_model() -> str:
    """Pick the model slug per the documented precedence."""
    env_override = os.environ.get("ZENMUX_IMAGE_MODEL", "").strip()
    if env_override:
        return env_override
    cfg = _load_zenmux_config()
    candidate = cfg.get("model")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return DEFAULT_MODEL


def _split_model(model: str) -> Tuple[str, str]:
    """Split a ``<provider>/<model>`` slug into its URL path components.

    Returns ``(provider, model_name)``. A slug without a provider prefix is
    routed under ``openai`` so callers can pass a bare model name. The live
    catalog's ``name`` field is itself in ``<provider>/<model>`` form
    (verified against the live endpoint), so picker selections split cleanly.
    """
    slug = (model or "").strip()
    if "/" in slug:
        provider, _, name = slug.partition("/")
        return provider or "openai", name or DEFAULT_MODEL.split("/", 1)[1]
    return "openai", slug or DEFAULT_MODEL.split("/", 1)[1]


# ---------------------------------------------------------------------------
# Dynamic model catalog
# ---------------------------------------------------------------------------
#
# ZenMux exposes a live Vertex catalog at ``/api/vertex-ai/v1beta/models``
# (no auth). It returns every model; we filter client-side to the ones whose
# ``outputModalities`` include ``"image"``.
#
# Caching uses a thread-safe SingletonSlot (plugins.plugin_utils): a
# successful fetch is cached for the process *even if it is empty* (the
# catalog genuinely lists no matching models), so we never re-hit the network
# in steady state. A *failed* fetch raises out of the factory, nothing is
# cached, and the next call retries — falling back to :data:`_MODELS`
# meanwhile.

_catalog_slot: SingletonSlot = SingletonSlot()


def _vertex_models_url() -> str:
    """Derive the v1beta models endpoint from the configured base URL."""
    base = _resolve_base_url()  # e.g. https://zenmux.ai/api/vertex-ai/v1
    root = base.rsplit("/", 1)[0]  # https://zenmux.ai/api/vertex-ai
    return f"{root}/v1beta/models"


def _first_price(model: Dict[str, Any], key: str) -> str:
    try:
        arr = (model.get("pricings") or {}).get(key) or []
        if arr:
            p = arr[0]
            return f"${p.get('value')}/{p.get('unit')}"
    except Exception:
        pass
    return ""


def _fetch_image_catalog_raw() -> List[Dict[str, Any]]:
    """Fetch image-output models from the live catalog.

    Raises on any network/HTTP/parse error (so the SingletonSlot does not
    cache a transient failure). Returns a possibly-empty list on success.
    """
    resp = requests.get(_vertex_models_url(), timeout=8.0)
    resp.raise_for_status()
    models = (resp.json() or {}).get("models", [])

    out: List[Dict[str, Any]] = []
    for m in models:
        if "image" not in (m.get("outputModalities") or []):
            continue
        name = m.get("name")
        if not name:
            continue
        entry: Dict[str, Any] = {
            "id": name,
            "display": m.get("displayName", name),
            "strengths": (m.get("description") or "")[:120],
        }
        price = _first_price(m, "image")
        if price:
            entry["price"] = price
        out.append(entry)
    return out


def _image_catalog() -> Optional[List[Dict[str, Any]]]:
    """Return the cached live catalog, or None when the fetch is unavailable."""
    try:
        return _catalog_slot.get(_fetch_image_catalog_raw)
    except Exception as exc:
        logger.debug("ZenMux image catalog unavailable: %s", exc)
        return None


def _static_image_catalog() -> List[Dict[str, Any]]:
    return [
        {
            "id": model_id,
            "display": meta.get("display", model_id),
            "speed": meta.get("speed", ""),
            "strengths": meta.get("strengths", ""),
        }
        for model_id, meta in _MODELS.items()
    ]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ZenMuxImageGenProvider(ImageGenProvider):
    """ZenMux Vertex-style image backend."""

    @property
    def name(self) -> str:
        return "zenmux"

    @property
    def display_name(self) -> str:
        return "ZenMux"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        live = _image_catalog()
        return live if live else _static_image_catalog()

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        ids = {m["id"] for m in models}
        if DEFAULT_MODEL in ids:
            return DEFAULT_MODEL
        return models[0]["id"] if models else DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "ZenMux (image)",
            "badge": "paid",
            "tag": "Vertex-style image gen across OpenAI / Imagen / Flux via one key",
            "env_vars": [
                {
                    "key": "ZENMUX_API_KEY",
                    "prompt": "ZenMux API key",
                    "url": "https://zenmux.ai/",
                },
            ],
        }

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key = _resolve_api_key()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not api_key:
            return error_response(
                error="No ZenMux credentials found. Set ZENMUX_API_KEY (https://zenmux.ai/).",
                error_type="missing_api_key",
                provider="zenmux",
                aspect_ratio=aspect,
            )

        model_slug = _resolve_model()
        provider, model_name = _split_model(model_slug)
        vertex_aspect = _ASPECT_MAP.get(aspect, "16:9")

        url = (
            f"{_resolve_base_url()}/publishers/{provider}/models/{model_name}:predict"
        )
        payload: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": vertex_aspect,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
        except requests.HTTPError as exc:
            resp = exc.response
            status = resp.status_code if resp is not None else 0
            try:
                err_msg = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                err_msg = resp.text[:300] if resp is not None else str(exc)
            logger.error("ZenMux image gen failed (%d): %s", status, err_msg)
            return error_response(
                error=f"ZenMux image generation failed ({status}): {err_msg}",
                error_type="api_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.Timeout:
            return error_response(
                error="ZenMux image generation timed out (120s)",
                error_type="timeout",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.ConnectionError as exc:
            return error_response(
                error=f"ZenMux connection error: {exc}",
                error_type="connection_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )
        except requests.RequestException as exc:
            # Catch-all for the remaining requests errors (SSLError,
            # TooManyRedirects, ChunkedEncodingError, ...) so a transport
            # failure surfaces as a clean error instead of crashing the tool.
            return error_response(
                error=f"ZenMux request failed: {exc}",
                error_type="connection_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        try:
            result = response.json()
        except Exception as exc:
            return error_response(
                error=f"ZenMux returned invalid JSON: {exc}",
                error_type="invalid_response",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        predictions = result.get("predictions") or []
        if not predictions:
            return error_response(
                error="ZenMux returned no image predictions",
                error_type="empty_response",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = predictions[0]
        b64 = first.get("bytesBase64Encoded")
        if not b64:
            return error_response(
                error="ZenMux prediction contained no bytesBase64Encoded image data",
                error_type="empty_response",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        mime = str(first.get("mimeType") or "image/png").split(";", 1)[0].strip().lower()
        extension = _MIME_EXT.get(mime, "png")
        try:
            saved_path = save_b64_image(
                b64, prefix=f"zenmux_{model_name}", extension=extension
            )
        except Exception as exc:
            return error_response(
                error=f"Could not save image to cache: {exc}",
                error_type="io_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        return success_response(
            image=str(saved_path),
            model=model_slug,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="zenmux",
            extra={"mime_type": mime},
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Register this provider with the image gen registry."""
    ctx.register_image_gen_provider(ZenMuxImageGenProvider())
