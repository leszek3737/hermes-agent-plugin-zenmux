"""ZenMux video generation backend.

Surface: text-to-video and image-to-video through ZenMux's Vertex-AI-style
long-running predict route. ZenMux fronts several upstream video models
(Google Veo, ByteDance Seedance, ...) behind one async ``:predictLongRunning``
endpoint.

Workflow (asynchronous):
1. POST .../publishers/{provider}/models/{model}:predictLongRunning
   -> returns ``{"name": "<operation>"}``
2. Poll POST .../publishers/{provider}/models/{model}:fetchPredictOperation
   with body ``{"operationName": "<operation>"}`` until ``done == true``.
3. Read the inline result at ``response.videos[0].bytesBase64Encoded`` and
   materialise it into the shared video cache.

The ``model`` is addressed as ``<provider>/<model>`` (e.g.
``google/veo-3.1-generate-001``); the slug is split into the URL path. Auth
is a single ``ZENMUX_API_KEY`` bearer token.

Selection precedence (first hit wins):
1. explicit tool ``model=`` argument
2. ``ZENMUX_VIDEO_MODEL`` env var
3. ``video_gen.zenmux.model`` in ``config.yaml``
4. :data:`DEFAULT_MODEL`

Base URL: the Vertex media base differs from the chat base, so it has its
own override ``ZENMUX_VERTEX_BASE_URL`` (NOT the chat profile's
``ZENMUX_BASE_URL``, which points at ``/api/v1``).
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent.video_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_RESOLUTION,
    VideoGenProvider,
    error_response,
    save_b64_video,
    success_response,
)
from plugins.plugin_utils import SingletonSlot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://zenmux.ai/api/vertex-ai/v1"
DEFAULT_MODEL = "google/veo-3.1-generate-001"

DEFAULT_DURATION = 8
DEFAULT_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 15

VALID_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}
VALID_RESOLUTIONS = {"720p", "1080p"}

# Curated fallback — the live Vertex catalog (/v1beta/models) currently lists
# no video-output models, but these slugs work against :predictLongRunning
# (verified live). ``max_duration`` / ``supports_audio`` are internal hints
# (see _max_duration_for / _model_supports_audio); they are NOT surfaced in
# picker rows.
_MODELS: Dict[str, Dict[str, Any]] = {
    "google/veo-3.1-generate-001": {
        "display": "Google Veo 3.1",
        "speed": "~30-180s",
        "strengths": "Text-to-video and image-to-video; native audio.",
        "max_duration": 8,
        "modalities": ["text", "image"],
        "supports_audio": True,
    },
    "volcengine/doubao-seedance-1.5-pro": {
        "display": "ByteDance Seedance 1.5 Pro",
        "speed": "~30-180s",
        "strengths": "High-fidelity motion; text and image input.",
        "max_duration": 10,
        "modalities": ["text", "image"],
        "supports_audio": False,
    },
    "volcengine/doubao-seedance-2": {
        "display": "ByteDance Seedance 2",
        "speed": "~30-180s",
        "strengths": "Latest Seedance generation.",
        "max_duration": 10,
        "modalities": ["text", "image"],
        "supports_audio": False,
    },
}


# ---------------------------------------------------------------------------
# Config / credentials
# ---------------------------------------------------------------------------


def _load_zenmux_config() -> Dict[str, Any]:
    """Read ``video_gen.zenmux`` from config.yaml (best-effort)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("video_gen") if isinstance(cfg, dict) else None
        zen = section.get("zenmux") if isinstance(section, dict) else None
        return zen if isinstance(zen, dict) else {}
    except Exception as exc:
        logger.debug("Could not load video_gen.zenmux config: %s", exc)
        return {}


def _resolve_api_key() -> str:
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


def _resolve_model(model: Optional[str], explicit_model: bool) -> str:
    """Pick the model slug per the documented precedence."""
    requested = (model or "").strip()
    if explicit_model and requested:
        return requested
    env_override = os.environ.get("ZENMUX_VIDEO_MODEL", "").strip()
    if env_override:
        return env_override
    cfg = _load_zenmux_config()
    candidate = cfg.get("model")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return requested or DEFAULT_MODEL


def _split_model(model: str) -> Tuple[str, str]:
    """Split ``<provider>/<model>`` into ``(provider, model_name)``."""
    slug = (model or "").strip()
    if "/" in slug:
        provider, _, name = slug.partition("/")
        if provider and name:
            return provider, name
    # No provider prefix — fall back to the default model's path.
    default_provider, default_name = DEFAULT_MODEL.split("/", 1)
    return default_provider, slug or default_name


def _max_duration_for(model_slug: str) -> int:
    meta = _MODELS.get(model_slug)
    return int(meta["max_duration"]) if meta and meta.get("max_duration") else 10


def _model_supports_audio(model_slug: str) -> bool:
    """Whether the model accepts ``generateAudio``.

    Known non-audio models (e.g. Seedance) return False so we don't send a
    rejected/ignored flag. Unknown models default to True (let the API
    decide).
    """
    meta = _MODELS.get(model_slug)
    if meta is None:
        return True
    return bool(meta.get("supports_audio", True))


def _clamp_duration(duration: Optional[int], model_slug: str) -> int:
    value = duration if duration is not None else DEFAULT_DURATION
    value = max(1, min(int(value), _max_duration_for(model_slug)))
    return value


def _extract_http_error(resp: Optional[requests.Response], exc: Exception) -> str:
    """Pull a human-friendly message out of an error response body."""
    if resp is None:
        return str(exc)
    try:
        return resp.json().get("error", {}).get("message", resp.text[:300])
    except Exception:
        return resp.text[:300] or str(exc)


def _image_ref_to_b64(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Convert a local path / data URI / HTTP URL into ``(base64, mime)``.

    Returns ``(None, None)`` when the reference cannot be resolved.
    """
    ref = (value or "").strip()
    if not ref:
        return None, None

    # Already a data URI: "data:image/png;base64,...."
    if ref.lower().startswith("data:image/"):
        try:
            header, _, b64 = ref.partition(",")
            mime = header.split(";", 1)[0][len("data:"):] or "image/png"
            return b64, mime
        except Exception:
            return None, None

    # Remote URL — download the bytes.
    if ref.lower().startswith(("http://", "https://")):
        try:
            resp = requests.get(ref, timeout=60)
            resp.raise_for_status()
            mime = (resp.headers.get("Content-Type") or "image/png").split(";", 1)[0].strip()
            return base64.b64encode(resp.content).decode("ascii"), mime or "image/png"
        except Exception as exc:
            logger.warning("Could not fetch image_url %s: %s", ref, exc)
            return None, None

    # Local file path.
    try:
        path = Path(ref).expanduser()
        if path.is_file():
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            if not mime.startswith("image/"):
                mime = "image/png"
            return base64.b64encode(path.read_bytes()).decode("ascii"), mime
    except OSError as exc:
        # File exists but is unreadable (permissions, I/O error, ...).
        logger.warning("Could not read local image file %s: %s", ref, exc)
        return None, None

    return None, None


# ---------------------------------------------------------------------------
# Dynamic model catalog
# ---------------------------------------------------------------------------
#
# ZenMux exposes a live Vertex catalog at ``/api/vertex-ai/v1beta/models``
# (no auth). We filter client-side to models whose ``outputModalities``
# include ``"video"``. At the time of writing the catalog lists none, so this
# transparently falls back to the curated :data:`_MODELS` list — but it will
# pick up Veo / Seedance automatically once they appear in the catalog.
#
# Caching uses a thread-safe SingletonSlot: a *successful* fetch is cached
# even when empty (so we don't re-hit the network on every picker render
# while the catalog has no video models), while a *failed* fetch raises out
# of the factory and is retried next call.

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


def _fetch_video_catalog_raw() -> List[Dict[str, Any]]:
    """Fetch video-output models from the live catalog.

    Raises on any network/HTTP/parse error (so the SingletonSlot does not
    cache a transient failure). Returns a possibly-empty list on success.
    """
    resp = requests.get(_vertex_models_url(), timeout=8.0)
    resp.raise_for_status()
    models = (resp.json() or {}).get("models", [])

    out: List[Dict[str, Any]] = []
    for m in models:
        if "video" not in (m.get("outputModalities") or []):
            continue
        name = m.get("name")
        if not name:
            continue
        entry: Dict[str, Any] = {
            "id": name,
            "display": m.get("displayName", name),
            "strengths": (m.get("description") or "")[:120],
            "modalities": m.get("inputModalities") or ["text"],
        }
        price = _first_price(m, "video")
        if price:
            entry["price"] = price
        out.append(entry)
    return out


def _video_catalog() -> Optional[List[Dict[str, Any]]]:
    """Return the cached live catalog, or None when the fetch is unavailable."""
    try:
        return _catalog_slot.get(_fetch_video_catalog_raw)
    except Exception as exc:
        logger.debug("ZenMux video catalog unavailable: %s", exc)
        return None


def _static_video_catalog() -> List[Dict[str, Any]]:
    return [
        {
            "id": mid,
            "display": meta.get("display", mid),
            "speed": meta.get("speed", ""),
            "strengths": meta.get("strengths", ""),
            "modalities": meta.get("modalities", ["text"]),
        }
        for mid, meta in _MODELS.items()
    ]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class ZenMuxVideoGenProvider(VideoGenProvider):
    """ZenMux Vertex-style async video backend (text-to-video + image-to-video)."""

    @property
    def name(self) -> str:
        return "zenmux"

    @property
    def display_name(self) -> str:
        return "ZenMux"

    def is_available(self) -> bool:
        return bool(_resolve_api_key())

    def list_models(self) -> List[Dict[str, Any]]:
        live = _video_catalog()
        return live if live else _static_video_catalog()

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        ids = {m["id"] for m in models}
        if DEFAULT_MODEL in ids:
            return DEFAULT_MODEL
        return models[0]["id"] if models else DEFAULT_MODEL

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "ZenMux (video)",
            "badge": "paid",
            "tag": "Text-to-video and image-to-video across Veo / Seedance via one key",
            "env_vars": [
                {
                    "key": "ZENMUX_API_KEY",
                    "prompt": "ZenMux API key",
                    "url": "https://zenmux.ai/",
                },
            ],
        }

    def capabilities(self) -> Dict[str, Any]:
        return {
            "modalities": ["text", "image"],
            "aspect_ratios": sorted(VALID_ASPECT_RATIOS),
            "resolutions": sorted(VALID_RESOLUTIONS),
            "max_duration": 10,
            "min_duration": 1,
            "supports_audio": True,
            "supports_negative_prompt": True,
            "max_reference_images": 0,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        reference_image_urls: Optional[List[str]] = None,
        duration: Optional[int] = None,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        resolution: str = DEFAULT_RESOLUTION,
        negative_prompt: Optional[str] = None,
        audio: Optional[bool] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        api_key = _resolve_api_key()
        prompt = (prompt or "").strip()
        if not api_key:
            return error_response(
                error="No ZenMux credentials found. Set ZENMUX_API_KEY (https://zenmux.ai/).",
                error_type="missing_api_key",
                provider="zenmux",
                prompt=prompt,
            )
        if not prompt:
            return error_response(
                error="prompt is required for ZenMux video generation",
                error_type="missing_prompt",
                provider="zenmux",
                prompt=prompt,
            )

        model_slug = _resolve_model(
            model, explicit_model=bool(kwargs.get("_model_override_explicit"))
        )
        provider, model_name = _split_model(model_slug)
        modality = "image" if image_url else "text"

        normalized_aspect = (aspect_ratio or DEFAULT_ASPECT_RATIO).strip()
        if normalized_aspect not in VALID_ASPECT_RATIOS:
            normalized_aspect = DEFAULT_ASPECT_RATIO
        normalized_resolution = (resolution or DEFAULT_RESOLUTION).strip().lower()
        if normalized_resolution not in VALID_RESOLUTIONS:
            normalized_resolution = DEFAULT_RESOLUTION
        clamped_duration = _clamp_duration(duration, model_slug)

        instance: Dict[str, Any] = {"prompt": prompt}
        if image_url:
            b64, mime = _image_ref_to_b64(image_url)
            if not b64:
                return error_response(
                    error=f"Could not read image_url for image-to-video: {image_url}",
                    error_type="invalid_image",
                    provider="zenmux",
                    model=model_slug,
                    prompt=prompt,
                )
            instance["image"] = {"bytesBase64Encoded": b64, "mimeType": mime}

        parameters: Dict[str, Any] = {
            "aspectRatio": normalized_aspect,
            "durationSeconds": clamped_duration,
            "resolution": normalized_resolution,
        }
        # Only send generateAudio to models that actually support it, so a
        # non-audio backend (Seedance) doesn't reject/ignore the flag.
        if audio is not None and _model_supports_audio(model_slug):
            parameters["generateAudio"] = bool(audio)
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if seed is not None:
            parameters["seed"] = int(seed)

        payload = {"instances": [instance], "parameters": parameters}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        base_url = _resolve_base_url()
        submit_url = f"{base_url}/publishers/{provider}/models/{model_name}:predictLongRunning"
        poll_url = f"{base_url}/publishers/{provider}/models/{model_name}:fetchPredictOperation"

        # ── Submit ───────────────────────────────────────────────
        try:
            resp = requests.post(submit_url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            operation_name = (resp.json() or {}).get("name")
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            detail = _extract_http_error(exc.response, exc)
            return error_response(
                error=f"ZenMux video submit failed ({status}): {detail}",
                error_type="api_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )
        except requests.RequestException as exc:
            return error_response(
                error=f"ZenMux video submit error: {exc}",
                error_type="connection_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        if not operation_name:
            return error_response(
                error="ZenMux submit response did not include an operation name",
                error_type="invalid_response",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        # ── Poll (check once before sleeping) ────────────────────
        deadline = time.monotonic() + DEFAULT_TIMEOUT_SECONDS
        body: Dict[str, Any] = {}
        while True:
            try:
                poll = requests.post(
                    poll_url,
                    headers=headers,
                    json={"operationName": operation_name},
                    timeout=30,
                )
                poll.raise_for_status()
                body = poll.json() or {}
            except requests.HTTPError as exc:
                # 4xx is non-transient (bad operation, expired auth, ...) —
                # abort instead of retrying until the deadline. 5xx falls
                # through to the transient path and keeps polling.
                status = exc.response.status_code if exc.response is not None else 0
                if 400 <= status < 500:
                    return error_response(
                        error=f"ZenMux video poll failed ({status}): {_extract_http_error(exc.response, exc)}",
                        error_type="api_error",
                        provider="zenmux",
                        model=model_slug,
                        prompt=prompt,
                        aspect_ratio=normalized_aspect,
                    )
                logger.debug("ZenMux poll transient HTTP error (%d): %s", status, exc)
                body = {}
            except requests.RequestException as exc:
                logger.debug("ZenMux poll transient error: %s", exc)
                body = {}
            if body.get("done"):
                break
            if time.monotonic() >= deadline:
                return error_response(
                    error=f"Timed out waiting for ZenMux video after {DEFAULT_TIMEOUT_SECONDS}s",
                    error_type="timeout",
                    provider="zenmux",
                    model=model_slug,
                    prompt=prompt,
                    aspect_ratio=normalized_aspect,
                )
            time.sleep(POLL_INTERVAL_SECONDS)

        if body.get("error"):
            err = body["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            return error_response(
                error=f"ZenMux video generation failed: {message}",
                error_type="generation_failed",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        videos = ((body.get("response") or {}).get("videos")) or []
        if not videos:
            return error_response(
                error="ZenMux completed without any video output",
                error_type="empty_response",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        first = videos[0]
        b64 = first.get("bytesBase64Encoded")
        if not b64:
            # Some Vertex backends return only a gs:// URI. The gateway can
            # neither download a gs:// URI nor treat it as a local path, so
            # report a clear error rather than a "success" with an
            # unconsumable reference.
            gcs_uri = first.get("gcsUri")
            hint = f" (got gcsUri {gcs_uri})" if gcs_uri else ""
            return error_response(
                error=f"ZenMux video had no inline bytes{hint}; this backend's output is not retrievable.",
                error_type="unconsumable_output",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        try:
            saved_path = save_b64_video(b64, prefix=f"zenmux_{model_name}")
        except Exception as exc:
            return error_response(
                error=f"Could not save video to cache: {exc}",
                error_type="io_error",
                provider="zenmux",
                model=model_slug,
                prompt=prompt,
                aspect_ratio=normalized_aspect,
            )

        return success_response(
            video=str(saved_path),
            model=model_slug,
            prompt=prompt,
            modality=modality,
            aspect_ratio=normalized_aspect,
            duration=clamped_duration,
            provider="zenmux",
            extra={"resolution": normalized_resolution, "operation": operation_name},
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Plugin entry point — wire ``ZenMuxVideoGenProvider`` into the registry."""
    ctx.register_video_gen_provider(ZenMuxVideoGenProvider())
