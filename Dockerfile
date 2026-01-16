# Root Dockerfile for Easypanel: builds the backend service and serves frontend static files
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    STATIC_DIR=/app/frontend

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Copy backend and frontend
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY backend/requirements.txt /app/requirements.txt

# Install python deps
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir --upgrade yt-dlp

# Expose port and run
EXPOSE 8000
ENV DATA_DIR=/data
ENV STATIC_DIR=/app/frontend
WORKDIR /app/backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
