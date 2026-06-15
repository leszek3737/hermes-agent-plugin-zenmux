# hermes-agent-plugin-zenmux

> [ZenMux](https://zenmux.ai) provider plugins for [Hermes](https://github.com/NousResearch/hermes) — chat, image, and video, all behind a single API key.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](#)
[![Hermes plugin](https://img.shields.io/badge/Hermes-plugin-7c3aed.svg)](#)

ZenMux is a multi-provider model router. This repository packages **three drop-in
Hermes plugins** that route chat completions, image generation, and video
generation through ZenMux — so one `ZENMUX_API_KEY` unlocks models from OpenAI,
Anthropic, Google, ByteDance, and more.

| Plugin | Kind | What it does |
| --- | --- | --- |
| **Model provider** | `model-provider` | OpenAI-compatible chat/agent inference across every ZenMux model (`<provider>/<model>` slugs). |
| **Image generation** | `backend` | Text-to-image via the Vertex-style `:predict` route (GPT Image, Imagen, Flux, …). |
| **Video generation** | `backend` | Text-to-video **and** image-to-video via the async `:predictLongRunning` route (Veo, Seedance, …). |

---

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available models](#available-models)
- [How it works](#how-it-works)
- [Caveats](#caveats)
- [Repository layout](#repository-layout)
- [License](#license)

---

## Requirements

- A working [Hermes](https://github.com/NousResearch/hermes) installation.
- A ZenMux account and API key — create one at <https://zenmux.ai>.
- Python ≥ 3.10 (matches the host; the plugins ship no extra dependencies — they
  reuse `requests`, already present in Hermes).

---

## Installation

Hermes loads user plugins from `$HERMES_HOME/plugins/` (default: `~/.hermes/plugins/`).
Copy the three plugins in, preserving the directory layout:

```bash
git clone https://github.com/leszek3737/hermes-agent-plugin-zenmux.git
cp -r hermes-agent-plugin-zenmux/plugins/* ~/.hermes/plugins/
```

This places:

```
~/.hermes/plugins/
├── model-providers/zenmux/
├── image_gen/zenmux/
└── video_gen/zenmux/
```

### Enable the plugins

- **Model provider** — discovered automatically (the provider registry scans
  `$HERMES_HOME/plugins/model-providers/`). No further action needed.
- **Image & video backends** — gated by `plugins.enabled` (opt-in). Enable both
  in one shot:

  ```bash
  hermes plugins enable zenmux
  ```

  …or add them to `~/.hermes/config.yaml`:

  ```yaml
  plugins:
    enabled:
      - zenmux
  ```

Set your key once (used by all three plugins) — keep it in `~/.hermes/.env`:

```bash
ZENMUX_API_KEY=sk-...
```

---

## Configuration

### Chat / agent model

Point Hermes at ZenMux and pick any model from the catalog:

```yaml
model:
  provider: zenmux
  default: openai/gpt-5        # any <provider>/<model> ZenMux slug
```

A bare `model: "anthropic/claude-sonnet-4.5"` with `provider: auto` also works
once `ZENMUX_API_KEY` is set — ZenMux is detected from the key.

### Image generation

```yaml
image_gen:
  provider: zenmux
  zenmux:
    model: openai/gpt-image-2  # optional; default is openai/gpt-image-2
```

### Video generation

```yaml
video_gen:
  provider: zenmux
  zenmux:
    model: google/veo-3.1-generate-001   # optional; default Veo 3.1
```

### Environment variables

| Variable | Used by | Purpose |
| --- | --- | --- |
| `ZENMUX_API_KEY` | all | Bearer token. **Required.** |
| `ZENMUX_BASE_URL` | chat | Override the chat base (`https://zenmux.ai/api/v1`). |
| `ZENMUX_VERTEX_BASE_URL` | image, video | Override the Vertex media base (`https://zenmux.ai/api/vertex-ai/v1`). |
| `ZENMUX_IMAGE_MODEL` | image | Force a specific image model. |
| `ZENMUX_VIDEO_MODEL` | video | Force a specific video model. |

> The chat base and the media base are **different endpoints**, which is why
> they have separate override variables. Don't point one at the other.

---

## Usage

Once configured, the plugins are transparent — use Hermes as usual:

- **Chat / agent**: any ZenMux model serves your session and tool calls.
- **Image**: the `image_generate` tool routes to ZenMux. Images are decoded from
  the API's inline base64 and saved under `~/.hermes/cache/images/`.
- **Video**: the `video_generate` tool routes to ZenMux. Provide an `image_url`
  to switch from text-to-video to image-to-video. Videos are saved under
  `~/.hermes/cache/videos/`.

Model selection precedence (image & video): explicit tool `model=` argument →
`ZENMUX_*_MODEL` env var → `config.yaml` → built-in default.

---

## Available models

The plugins fetch the live catalog so you always see what your key can actually
use — no hard-coded lists to go stale:

- **Chat** — `GET /api/v1/models` (135+ models at time of writing).
- **Image** — `GET /api/vertex-ai/v1beta/models`, filtered to image-output models
  (GPT Image, Imagen, Qwen-Image, Seedream, Gemini image, …).
- **Video** — same catalog, filtered to video-output models. The catalog does
  not currently list video models, so the video plugin falls back to a curated
  set (Veo 3.1, Seedance 1.5 Pro, Seedance 2) that is confirmed working against
  the generation endpoint.

A successful-but-empty catalog fetch is cached for the process; a failed fetch
(offline) falls back to the curated list and is retried on the next call.

---

## How it works

Each plugin is a thin adapter onto a Hermes extension point:

- **Model provider** is a declarative `ProviderProfile` — it describes the
  endpoint, auth, and vision support; the standard `chat_completions` transport
  does the rest.
- **Image / video** subclass `ImageGenProvider` / `VideoGenProvider` and
  implement `generate()`, returning the framework's uniform success/error dicts.

| Endpoint | Method | Response shape |
| --- | --- | --- |
| Chat | `POST /api/v1/chat/completions` | OpenAI-compatible |
| Image | `POST /api/vertex-ai/v1/publishers/{provider}/models/{model}:predict` | `predictions[0].bytesBase64Encoded` |
| Video (submit) | `POST .../{model}:predictLongRunning` | `{ "name": "<operation>" }` |
| Video (poll) | `POST .../{model}:fetchPredictOperation` | `{ "done": true, "response": { "videos": [...] } }` |

---

## Caveats

- **`max_tokens` on chat** — ZenMux expects `max_completion_tokens`. Hermes only
  auto-renames this for OpenAI-family slugs (`gpt-*`, `o1`/`o3`/`o4`). On
  `anthropic/*` or `google/*` slugs, an explicit `max_tokens` returns HTTP 400.
  Leave `max_tokens` unset on ZenMux (the default) and you're fine.
- **Video output** — some Vertex backends may return only a `gs://` URI, which
  the gateway cannot fetch. The plugin reports a clear error in that case rather
  than handing back an unusable reference. Models that return inline bytes (Veo,
  Seedance) work end-to-end.
- **Generation latency** — video is asynchronous and typically takes 30 s–3 min;
  the plugin polls every 15 s up to a 5-minute deadline.

---

## Repository layout

```
plugins/
├── model-providers/
│   └── zenmux/
│       ├── __init__.py     # ProviderProfile + register_provider()
│       └── plugin.yaml     # kind: model-provider
├── image_gen/
│   └── zenmux/
│       ├── __init__.py     # ZenMuxImageGenProvider + register(ctx)
│       └── plugin.yaml     # kind: backend
└── video_gen/
    └── zenmux/
        ├── __init__.py     # ZenMuxVideoGenProvider + register(ctx)
        └── plugin.yaml     # kind: backend
```

The layout mirrors Hermes' own `plugins/` tree, so installation is a plain copy.

---

## License

[MIT](LICENSE.md) © 2026 Leszek
