# Arabic YouTube → Book (self-hosted)

Open-source stack to download long YouTube videos, transcribe Arabic audio with Whisper, and chat to turn it into a book. Designed for Easypanel/Coolify on a VPS — no SaaS.

## Stack
- Backend: FastAPI + yt-dlp + faster-whisper
- LLM: Optional Ollama (use your Easypanel Ollama by setting `LLM_API_URL`). The Compose file no longer starts Ollama by default to avoid pulling large models.
- Frontend: static HTML/JS served by FastAPI
- Packaging: Docker Compose

## Run (server)
```bash
docker compose build
docker compose up -d
```
App on port 8000. Data stored in `./data`.

To use your Easypanel Ollama (recommended if you don't want the stack to pull models):
1. Create a `.env` file in the project root or export env vars before starting.

Example `.env`:
```
LLM_API_URL=https://basheer-ollama.x0uyzh.easypanel.host/v1/chat/completions
LLM_MODEL=small-instruct-model-name
```

- If you leave `LLM_API_URL` empty the chat endpoint will return an error and transcription-only workflow still works.
- **Important:** pick a small model already installed on your Easypanel Ollama (check the Ollama UI for installed models) to avoid downloading large weights.

## Env knobs (compose)
- `WHISPER_MODEL` (default `large-v3`)
- `WHISPER_DEVICE` (`cpu`|`cuda`)
- `WHISPER_COMPUTE_TYPE` (`int8_float16`, `float16`, etc.)
- `LLM_API_URL` (set to your Ollama host). If empty, chat is disabled.
- `LLM_MODEL` model name installed on your Ollama (prefer smaller 7B models to avoid big downloads).

## Flow
1) Paste YouTube URL in UI.
2) Backend downloads via yt-dlp → transcribes with faster-whisper → exposes transcript.
3) Chat endpoint sends your prompt + transcript slice to LLM to draft chapters/edits.

## Notes
- For age/region-locked videos, mount a `cookies.txt` into the container and add yt-dlp cookie handling in `backend/app.py`.
- To change model, adjust compose env and ensure it’s pulled in the Ollama container.
- This is a minimal baseline. For production, add persistence (DB), auth, and better chunked retrieval/embeddings.
