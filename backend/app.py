import os
import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Set

import requests
import yt_dlp
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
STATIC_DIR = Path(os.getenv("STATIC_DIR", "../frontend"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# AnythingLLM Configuration
ALLM_BASE_URL = "https://basheer-everythingllm.x0uyzh.easypanel.host/api/v1"
ALLM_API_KEY = "ZYE6SD0-7EDM9Z6-P833MGZ-PPTGX59"
ALLM_WORKSPACE = "voice"

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


@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...), language: str = "ar"):
    """Upload audio file directly for transcription"""
    try:
        job_id = str(uuid.uuid4())
        folder = DATA_DIR / job_id
        folder.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        audio_path = folder / file.filename
        content = await file.read()
        audio_path.write_bytes(content)
        
        # Create job
        transcript_path = folder / "transcript.txt"
        job = Job(
            job_id=job_id,
            url=f"uploaded:{file.filename}",
            language=language,
            model_size="large-v3",
            audio_path=audio_path,
            transcript_path=transcript_path
        )
        jobs[job_id] = job
        
        # Start transcription in background
        executor.submit(process_uploaded_audio, job)
        
        return {"job_id": job_id, "status": "transcribing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload_potoken")
async def upload_potoken(token: str = Form(...)):
    token_path = DATA_DIR / "potoken.txt"
    try:
        token_path.write_text(token.strip(), encoding="utf-8")
        return {"message": "PO Token saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "2")))


class CreateJob(BaseModel):
    url: Optional[str] = None
    language: str = "ar"
    model_size: str = os.getenv("WHISPER_MODEL", "large-v3")


class ChatRequest(BaseModel):
    job_id: str
    message: str
    model: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None


class Job:
    def __init__(self, job_id: str, url: str, language: str, model_size: str, audio_path: Optional[Path] = None, transcript_path: Optional[Path] = None):
        self.id = job_id
        self.url = url
        self.language = language
        self.model_size = model_size
        self.status = "pending"
        self.error: Optional[str] = None
        self.video_path: Optional[Path] = None
        self.audio_path: Optional[Path] = audio_path
        self.transcript_path: Optional[Path] = transcript_path
        self.websocket_clients: Set[WebSocket] = set()  # Track connected WebSocket clients


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
        # Check if the job folder exists (in case of restart)
        folder = DATA_DIR / job_id
        if folder.exists():
            # Reconstruct job from file system
            transcript_path = folder / "transcript.txt"
            if transcript_path.exists():
                status = "done"
            else:
                status = "transcribing"
            return {
                "id": job_id,
                "url": f"uploaded",
                "language": "auto",
                "model_size": "large-v3",
                "status": status,
                "error": None,
            }
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


@app.websocket("/ws/jobs/{job_id}/transcript")
async def websocket_transcript(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for live transcript streaming"""
    job = jobs.get(job_id)
    if not job:
        await websocket.close(code=4004, reason="job not found")
        return
    
    await websocket.accept()
    job.websocket_clients.add(websocket)
    
    try:
        # Keep connection open and handle incoming messages if any
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        job.websocket_clients.discard(websocket)
    except Exception:
        job.websocket_clients.discard(websocket)


@app.post("/api/chat")
def chat(body: ChatRequest):
    job = jobs.get(body.job_id)
    if not job or not job.transcript_path or not job.transcript_path.exists():
        raise HTTPException(status_code=404, detail="transcript not ready")
    transcript = job.transcript_path.read_text(encoding="utf-8")
    context = transcript[-8000:]
    reply = call_llm(
        body.message, 
        context=context, 
        model=body.model,
        api_url=body.api_url,
        api_key=body.api_key
    )
    return {"reply": reply}


def process_job(job: Job):
    try:
        job.status = "downloading"
        folder = DATA_DIR / job.id
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

        video_path = folder / "video.mp4"
        print(f"DEBUG: Starting download to {video_path}")
        download_video(job.url, video_path)
        
        # Verify the file actually exists before transcribing
        if not video_path.exists():
            # Sometimes yt-dlp might name it video.mp4.m4a or similar if merging fails
            # Search for anything starting with 'video.' in the folder
            actual_files = list(folder.glob("video.*"))
            if actual_files:
                video_path = actual_files[0]
                print(f"DEBUG: Found downloaded file at {video_path}")
            else:
                raise Exception(f"Download failed: {video_path} does not exist after yt-dlp run")
        
        job.video_path = video_path

        job.status = "transcribing"
        transcript_path = folder / "transcript.txt"
        anythingllm_transcribe(video_path, transcript_path, job=job)
        job.transcript_path = transcript_path

        job.status = "done"
    except Exception as exc:  # noqa: BLE001
        job.status = "failed"
        job.error = str(exc)
        print(f"DEBUG: Job failed with error: {str(exc)}")


def download_video(url: str, output_path: Path):
    ydl_opts = {
        "outtmpl": str(output_path.with_suffix("")),
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "referer": "https://www.google.com/",
        "nocheckcertificate": True,
        "ignoreerrors": True,
        "no_color": True,
        "geo_bypass": True,
    }
    
    # Search for PO Token
    possible_po_token_paths = [
        DATA_DIR / "po_token.txt",
        Path("/data/po_token.txt")
    ]
    
    po_token = None
    for p in possible_po_token_paths:
        if p.exists():
            po_token = p.read_text().strip()
            print(f"DEBUG: Using PO Token from {p}")
            break
    
    if po_token:
        # Use android_music client - no nsig, works with PO token
        po_token_formatted = f"android_music+{po_token}"
        ydl_opts["extractor_args"] = {
            "youtube": {
                "po_token": [po_token_formatted],
                "player_client": ["android_music"]
            }
        }
        print(f"DEBUG: PO Token configured: {po_token_formatted[:35]}...")
    else:
        print("DEBUG: No PO Token found, using android client")
        ydl_opts["extractor_args"] = {"youtube": {"player_client": ["android"]}}
    
    # Comprehensive cookie file search
    possible_cookie_paths = [
        DATA_DIR / "cookies.txt",
        Path("cookies.txt"),
        Path("../cookies.txt"),
        Path("/app/backend/data/cookies.txt"),
        Path("/data/cookies.txt")
    ]
    
    found_cookies = None
    for p in possible_cookie_paths:
        if p.exists():
            found_cookies = str(p)
            print(f"DEBUG: Using cookies from {found_cookies}")
            break
            
    if found_cookies:
        ydl_opts["cookiefile"] = found_cookies
    else:
        print("DEBUG: No cookies.txt found in any search path.")
    
    # PO Token support - YouTube now requires this
    potoken_path = DATA_DIR / "potoken.txt"
    if potoken_path.exists():
        potoken = potoken_path.read_text(encoding="utf-8").strip()
        if potoken:
            print(f"DEBUG: Using PO Token: {potoken[:20]}...")
            ydl_opts["extractor_args"] = {
                "youtube": {
                    "player_client": ["web"],
                    "po_token": potoken
                }
            }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def anythingllm_transcribe(audio_path: Path, transcript_path: Path, job: Optional[Job] = None):
    """Transcribe audio file using AnythingLLM API"""
    import base64
    
    print(f"DEBUG: Starting AnythingLLM transcription for {audio_path.name}")
    headers = {"Authorization": f"Bearer {ALLM_API_KEY}"}
    
    # Read and Base64 encode the audio file
    print(f"DEBUG: Reading audio file ({audio_path.stat().st_size} bytes)")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    print(f"DEBUG: Base64 encoded ({len(audio_b64)} chars)")
    
    # Detect MIME type from file extension
    ext = audio_path.suffix.lower()
    mime_map = {
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".opus": "audio/opus",
        ".mp4": "video/mp4",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm"
    }
    audio_mime = mime_map.get(ext, "audio/mpeg")
    print(f"DEBUG: Detected MIME type: {audio_mime}")
    
    # Create Data URI
    data_uri = f"data:{audio_mime};base64,{audio_b64}"
    
    # Send chat with document attachment to trigger transcription
    chat_url = f"{ALLM_BASE_URL}/workspace/{ALLM_WORKSPACE}/chat"
    print(f"DEBUG: Sending chat request to {chat_url}")
    print(f"DEBUG: Request payload size: ~{len(data_uri)/1024/1024:.1f} MB")
    
    try:
        chat_resp = requests.post(
            chat_url,
            headers={**headers, "Content-Type": "application/json"},
            json={
                "message": "Please transcribe this audio file.",
                "mode": "chat",
                "attachments": [{
                    "name": audio_path.name,
                    "mime": "application/anythingllm-document",
                    "contentString": data_uri
                }]
            },
            timeout=600  # 10 minute timeout for large files
        )
    except requests.exceptions.Timeout as e:
        print(f"ERROR: Request timed out after 10 minutes: {e}")
        raise Exception(f"Transcription timed out - audio file too large or service busy")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        raise Exception(f"API request failed: {e}")
    
    print(f"DEBUG: AnythingLLM response status: {chat_resp.status_code}")
    
    if chat_resp.status_code != 200:
        print(f"ERROR: AnythingLLM API failed with status {chat_resp.status_code}")
        print(f"ERROR: Response body: {chat_resp.text}")
        raise Exception(f"Chat failed: HTTP {chat_resp.status_code} - {chat_resp.text}")
    
    chat_data = chat_resp.json()
    print(f"DEBUG: Received chat response: {str(chat_data)[:200]}")
    
    transcript_text = chat_data.get("textResponse", "")
    if not transcript_text:
        print(f"WARNING: Empty textResponse from AnythingLLM. Full response: {chat_data}")
    
    # Write transcript to file
    transcript_path.write_text(transcript_text, encoding="utf-8")
    print(f"DEBUG: Wrote transcript to {transcript_path} ({len(transcript_text)} chars)")


def call_llm(message: str, context: str, model: Optional[str] = None, api_url: Optional[str] = None, api_key: Optional[str] = None) -> str:
    final_url = (api_url or os.getenv("LLM_API_URL", "")).strip()
    if not final_url:
        raise HTTPException(status_code=503, detail="LLM not configured. Select a provider in the UI.")
    
    # Handle the specific completions endpoint if it's just a base URL
    if not final_url.endswith("/chat/completions") and not final_url.endswith("/chat/completions#"):
        final_url = final_url.rstrip("/") + "/chat/completions"
    elif final_url.endswith("#"):
        final_url = final_url[:-1]

    final_key = api_key or os.getenv("LLM_API_KEY")
    model_name = model or os.getenv("LLM_MODEL", "")

    payload = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert AI assistant specializing in transcript analysis and content refinement. Help the user understand, edit, summarize, and extract insights from transcribed content. Be clear, concise, and professional."
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
    if final_key:
        headers["Authorization"] = f"Bearer {final_key}"

    response = requests.post(final_url, json=payload, headers=headers, timeout=120)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM error: {response.text}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Bad LLM response: {data}") from exc


def process_uploaded_audio(job: Job):
    """Process audio file that was uploaded directly"""
    try:
        job.status = "transcribing"
        
        if not job.audio_path or not job.audio_path.exists():
            raise Exception(f"Audio file not found: {job.audio_path}")
        
        print(f"DEBUG: Transcribing uploaded audio via AnythingLLM: {job.audio_path}")
        
        # Call AnythingLLM transcription
        try:
            anythingllm_transcribe(job.audio_path, job.transcript_path, job=job)
            job.status = "done"
            print(f"DEBUG: Transcription completed successfully for {job.id}")
        except Exception as transcribe_error:
            print(f"ERROR: AnythingLLM transcription failed: {transcribe_error}")
            import traceback
            traceback.print_exc()
            raise
        
        # Send final completion message to all connected clients
        import asyncio
        async def send_complete():
            for ws in list(job.websocket_clients):
                try:
                    await ws.send_json({"type": "complete", "status": "done"})
                except Exception:
                    job.websocket_clients.discard(ws)
        
        try:
            asyncio.create_task(send_complete())
        except RuntimeError:
            pass
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        print(f"DEBUG: Job failed with error: {str(exc)}")
        
        # Send error message to all connected clients
        import asyncio
        async def send_error():
            for ws in list(job.websocket_clients):
                try:
                    await ws.send_json({"type": "error", "error": str(exc)})
                except Exception:
                    job.websocket_clients.discard(ws)
        
        try:
            asyncio.create_task(send_error())
        except RuntimeError:
            pass


# Mount static files LAST so all API routes are handled first
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
