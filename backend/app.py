import os
import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import requests
import yt_dlp
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from faster_whisper import WhisperModel

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "../frontend"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YouTube Arabic Transcriber", version="0.1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/upload_cookies")
async def upload_cookies(file: UploadFile = File(...)):
    cookie_path = DATA_DIR / "cookies.txt"
    try:
        content = await file.read()
        cookie_path.write_bytes(content)
        return {"message": "cookies.txt uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))


class CreateJob(BaseModel):
    url: str
    language: str = "ar"
    model_size: str = os.getenv("WHISPER_MODEL", "large-v3")


class ChatRequest(BaseModel):
    job_id: str
    message: str
    model: Optional[str] = None


class Job:
    def __init__(self, job_id: str, url: str, language: str, model_size: str):
        self.id = job_id
        self.url = url
        self.language = language
        self.model_size = model_size
        self.status = "pending"
        self.error: Optional[str] = None
        self.video_path: Optional[Path] = None
        self.transcript_path: Optional[Path] = None


jobs: Dict[str, Job] = {}


@app.post("/api/jobs")
def create_job(body: CreateJob):
    job_id = uuid.uuid4().hex[:12]
    job = Job(job_id, body.url, body.language, body.model_size)
    jobs[job_id] = job
    executor.submit(process_job, job)
    return {"job_id": job_id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "url": job.url,
        "language": job.language,
        "model_size": job.model_size,
        "status": job.status,
        "error": job.error,
    }


@app.get("/api/jobs/{job_id}/transcript")
def get_transcript(job_id: str):
    job = jobs.get(job_id)
    if not job or not job.transcript_path or not job.transcript_path.exists():
        raise HTTPException(status_code=404, detail="transcript not ready")
    return {"transcript": job.transcript_path.read_text(encoding="utf-8")}


@app.post("/api/chat")
def chat(body: ChatRequest):
    job = jobs.get(body.job_id)
    if not job or not job.transcript_path or not job.transcript_path.exists():
        raise HTTPException(status_code=404, detail="transcript not ready")
    transcript = job.transcript_path.read_text(encoding="utf-8")
    context = transcript[-8000:]
    reply = call_llm(body.message, context=context, model=body.model)
    return {"reply": reply}


def process_job(job: Job):
    try:
        job.status = "downloading"
        folder = DATA_DIR / job.id
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

        video_path = folder / "video.mp4"
        download_video(job.url, video_path)
        job.video_path = video_path

        job.status = "transcribing"
        transcript_path = folder / "transcript.txt"
        transcribe_video(video_path, transcript_path, language=job.language, model_size=job.model_size)
        job.transcript_path = transcript_path

        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)


def download_video(url: str, output_path: Path):
    ydl_opts = {
        "outtmpl": str(output_path),
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
    }
    
    # Check for cookies.txt in the data directory to bypass bot detection
    cookie_file = DATA_DIR / "cookies.txt"
    if cookie_file.exists():
        ydl_opts["cookiefile"] = str(cookie_file)
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def transcribe_video(video_path: Path, transcript_path: Path, language: str, model_size: str):
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8_float16")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(str(video_path), language=language)
    lines = []
    for segment in segments:
        lines.append(f"[{segment.start:.2f}-{segment.end:.2f}] {segment.text}")
    transcript_path.write_text("\n".join(lines), encoding="utf-8")


def call_llm(message: str, context: str, model: Optional[str] = None) -> str:
    api_url = os.getenv("LLM_API_URL", "").strip()
    if not api_url:
        raise HTTPException(status_code=503, detail="LLM not configured. Set LLM_API_URL to your Ollama host (e.g. https://your-ollama-host/v1/chat/completions) and LLM_MODEL to a small model available there.")
    api_key = os.getenv("LLM_API_KEY")
    model_name = model or os.getenv("LLM_MODEL", "")

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are a book-writing assistant. Use the provided transcript context to write concise, well-structured Arabic prose."
            },
            {"role": "system", "content": f"Transcript context (may be partial):\n{context}"},
            {"role": "user", "content": message},
        ],
        "stream": False,
        "temperature": 0.4,
    }
    if model_name:
        payload["model"] = model_name

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(api_url, json=payload, headers=headers, timeout=120)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM error: {response.text}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Bad LLM response: {data}") from exc


# Serve static frontend if it exists
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
