#!/bin/bash
set -e

# =====================================================================
# Wan2.2 GCP Cloud Shell Master Setup Script
# Run this script directly inside Google Cloud Shell to automate everything!
# =====================================================================

export PROJECT_ID="cs-26delhi-lab2-10830"
export BUCKET_NAME="wan-video-storage-10830"
export REGION="us-central1"
export ZONE="us-central1-a"

echo "====== [Step 1/6] Configuring Project and Enabling GCP Services ======"
gcloud config set project $PROJECT_ID

echo "Enabling Artifact Registry and Kubernetes Engine APIs (may take 1-2 minutes)..."
gcloud services enable \
    artifactregistry.googleapis.com \
    container.googleapis.com \
    aiplatform.googleapis.com \
    compute.googleapis.com

echo "====== [Step 2/6] Preparing Workspace and Cloning Repo ======"
if [ -d "Wan2.2" ]; then
    echo "Directory 'Wan2.2' already exists. Moving into it..."
    cd Wan2.2
else
    echo "Cloning the Wan2.2 repository..."
    git clone https://github.com/Wan-Video/Wan2.2.git
    cd Wan2.2
fi

echo "Creating API and Deployment folders..."
mkdir -p api deploy api_storage/uploads api_storage/generated api_storage/logs

# =====================================================================
# WRITE FILE: api/config.py
# =====================================================================
cat << 'EOF' > api/config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    HOST: str = os.getenv("WAN_API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("WAN_API_PORT", "8000"))
    DATABASE_URL: str = os.getenv("WAN_DATABASE_URL", "sqlite:///api_jobs.db")
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCS_BUCKET_NAME: str = os.getenv("WAN_GCS_BUCKET", "")
    GCS_ENABLED: bool = bool(GCS_BUCKET_NAME)
    STORAGE_DIR: Path = BASE_DIR / "api_storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    GENERATED_DIR: Path = STORAGE_DIR / "generated"
    LOGS_DIR: Path = STORAGE_DIR / "logs"
    WAN_MODEL_DIR: str = os.getenv("WAN_MODEL_DIR", str(BASE_DIR))
    FFMPEG_PATH: str = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else os.getenv("FFMPEG_PATH", "ffmpeg")
    MOCK_MODE: bool = os.getenv("WAN_API_MOCK", "False").lower() in ("true", "1", "yes")

settings = Settings()
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
EOF

# =====================================================================
# WRITE FILE: api/models.py
# =====================================================================
cat << 'EOF' > api/models.py
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()

class DbFineTuneJob(Base):
    __tablename__ = "finetune_jobs"
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")
    model_type = Column(String, default="5B")
    dataset_path = Column(String, nullable=True)
    trigger_word = Column(String, nullable=True)
    learning_rate = Column(Float, default=1e-4)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=1)
    resolution = Column(String, default="720p")
    progress = Column(Float, default=0.0)
    metrics_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    log_file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DbVideoGenJob(Base):
    __tablename__ = "videogen_jobs"
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")
    task_type = Column(String, default="t2v")
    prompt = Column(Text, nullable=True)
    resolution = Column(String, default="720p")
    duration_seconds = Column(Integer, default=5)
    total_clips = Column(Integer, default=1)
    completed_clips = Column(Integer, default=0)
    image_url = Column(String, nullable=True)
    audio_url = Column(String, nullable=True)
    output_video_url = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FineTuneStartRequest(BaseModel):
    model_type: str = Field("5B")
    dataset_id: str = Field(...)
    trigger_word: str = Field("character_concept")
    learning_rate: float = Field(1e-4)
    epochs: int = Field(10)
    batch_size: int = Field(1)
    resolution: str = Field("720p")

class FineTuneJobResponse(BaseModel):
    id: str
    status: str
    model_type: str
    trigger_word: Optional[str]
    learning_rate: float
    epochs: int
    batch_size: int
    resolution: str
    progress: float
    metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class VideoGenStartRequest(BaseModel):
    prompt: str = Field(...)
    task_type: str = Field("t2v")
    resolution: str = Field("720p")
    duration_seconds: int = Field(5)
    image_url: Optional[str] = Field(None)
    audio_url: Optional[str] = Field(None)

class VideoGenJobResponse(BaseModel):
    id: str
    status: str
    task_type: str
    prompt: str
    resolution: str
    duration_seconds: int
    total_clips: int
    completed_clips: int
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    output_video_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

class SystemHealthResponse(BaseModel):
    status: str
    cpu_usage_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_available: bool
    gpu_devices: List[Dict[str, Any]] = []
EOF

# =====================================================================
# WRITE FILE: api/database.py
# =====================================================================
cat << 'EOF' > api/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .config import settings
from .models import Base

connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
EOF

# =====================================================================
# WRITE FILE: api/storage.py
# =====================================================================
cat << 'EOF' > api/storage.py
import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from datetime import timedelta
from .config import settings

try:
    from google.cloud import storage
    GCS_CLIENT_AVAILABLE = True
except ImportError:
    GCS_CLIENT_AVAILABLE = False

logger = logging.getLogger("wan_api.storage")

class StorageManager:
    def __init__(self):
        self.gcs_client = None
        self.bucket = None
        if settings.GCS_ENABLED:
            if GCS_CLIENT_AVAILABLE:
                try:
                    self.gcs_client = storage.Client(project=settings.GCP_PROJECT_ID)
                    self.bucket = self.gcs_client.bucket(settings.GCS_BUCKET_NAME)
                except Exception as e:
                    logger.warning(f"GCS init failed: {e}. Falling back to local.")
            else:
                logger.warning("GCS pkg missing. Falling back to local.")

    def save_uploaded_file(self, source_file, target_filename: str) -> str:
        local_path = settings.UPLOAD_DIR / target_filename
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(source_file, buffer)
        if self.bucket:
            try:
                gcs_blob_name = f"uploads/{target_filename}"
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(str(local_path))
                return f"gs://{settings.GCS_BUCKET_NAME}/{gcs_blob_name}"
            except Exception as e:
                logger.error(f"GCS Upload failed: {e}")
        return str(local_path)

    def upload_generated_video(self, local_video_path: Path) -> str:
        filename = local_video_path.name
        if self.bucket:
            try:
                gcs_blob_name = f"generated/{filename}"
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(str(local_video_path))
                return blob.generate_signed_url(version="v4", expiration=timedelta(days=7), method="GET")
            except Exception as e:
                logger.error(f"GCS Video upload failed: {e}")
        return f"/static/generated/{filename}"

    def upload_lora_weights(self, local_weights_path: Path) -> str:
        filename = local_weights_path.name
        if self.bucket:
            try:
                gcs_blob_name = f"models/lora/{filename}"
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(str(local_weights_path))
                return f"gs://{settings.GCS_BUCKET_NAME}/{gcs_blob_name}"
            except Exception as e:
                logger.error(f"GCS Weights upload failed: {e}")
        return str(local_weights_path)

    def download_to_local(self, remote_url_or_path: str, target_local_path: Path) -> Path:
        if remote_url_or_path.startswith("gs://"):
            path_parts = remote_url_or_path[5:].split("/", 1)
            blob = self.gcs_client.bucket(path_parts[0]).blob(path_parts[1])
            blob.download_to_filename(str(target_local_path))
            return target_local_path
        elif remote_url_or_path.startswith(("http://", "https://")):
            import requests
            r = requests.get(remote_url_or_path, stream=True)
            with open(target_local_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
            return target_local_path
        else:
            shutil.copy2(remote_url_or_path, target_local_path)
            return target_local_path

storage_manager = StorageManager()
EOF

# =====================================================================
# WRITE FILE: api/orchestrator.py
# =====================================================================
cat << 'EOF' > api/orchestrator.py
import os
import subprocess
import logging
from pathlib import Path
from typing import List, Optional
from .config import settings

logger = logging.getLogger("wan_api.orchestrator")

class VideoOrchestrator:
    @staticmethod
    def plan_clips(prompt: str, duration_seconds: int) -> List[str]:
        total_clips = max(1, duration_seconds // 5)
        prompt_segments = [p.strip() for p in prompt.split(";") if p.strip()]
        planned_prompts = []
        for i in range(total_clips):
            if i < len(prompt_segments):
                planned_prompts.append(prompt_segments[i])
            else:
                planned_prompts.append(prompt_segments[-1] if prompt_segments else prompt)
        return planned_prompts

    @staticmethod
    def stitch_clips_ffmpeg(clips: List[Path], output_path: Path) -> Path:
        if not clips: raise ValueError("No clips provided.")
        if len(clips) == 1:
            import shutil
            shutil.copy2(clips[0], output_path)
            return output_path
        concat_file = output_path.parent / f"concat_{output_path.stem}.txt"
        with open(concat_file, "w", encoding="utf-8") as f:
            for clip in clips:
                escaped = str(clip.resolve()).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
        try:
            cmd = [settings.FFMPEG_PATH, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return output_path
        finally:
            if concat_file.exists(): os.remove(concat_file)

    @staticmethod
    def generate_single_clip_subprocess(clip_index: int, prompt: str, task_type: str, resolution: str, output_dir: Path, image_path: Optional[str] = None, audio_path: Optional[str] = None, mock_mode: bool = False) -> Path:
        clip_path = output_dir / f"segment_{clip_index:04d}.mp4"
        if mock_mode:
            import time
            time.sleep(1)
            cmd = [settings.FFMPEG_PATH, "-y", "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d=5", "-vf", f"drawtext=text='Clip {clip_index+1} | Prompt: {prompt[:20]}...':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_path)]
            try: subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                subprocess.run([settings.FFMPEG_PATH, "-y", "-f", "lavfi", "-i", "color=c=blue:s=1280x720:d=5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip_path)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return clip_path
        size_param = "1280*704" if resolution == "720p" else "854*480"
        cmd = ["python", "generate.py", "--task", "ti2v-5B", "--size", size_param, "--ckpt_dir", "./Wan2.2-TI2V-5B", "--offload_model", "True", "--convert_model_dtype", "--t5_cpu", "--prompt", prompt]
        if image_path: cmd.extend(["--image", str(image_path)])
        subprocess.run(cmd, check=True)
        latest_video = max(list(settings.BASE_DIR.glob("*.mp4")), key=os.path.getmtime)
        import shutil
        shutil.move(str(latest_video), str(clip_path))
        return clip_path
EOF

# =====================================================================
# WRITE FILE: api/tasks.py
# =====================================================================
cat << 'EOF' > api/tasks.py
import os, sys, time, subprocess, logging
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from .config import settings
from .database import SessionLocal
from .models import DbVideoGenJob, DbFineTuneJob
from .orchestrator import VideoOrchestrator
from .storage import storage_manager

logger = logging.getLogger("wan_api.tasks")

class BackgroundTaskManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.active_jobs: Dict[str, subprocess.Popen] = {}

    def submit_video_generation(self, job_id: str):
        self.executor.submit(self._run_video_generation_task, job_id)

    def submit_fine_tuning(self, job_id: str):
        self.executor.submit(self._run_fine_tuning_task, job_id)

    def cancel_job(self, job_id: str) -> bool:
        proc = self.active_jobs.get(job_id)
        if proc:
            proc.terminate()
            return True
        return False

    def _run_video_generation_task(self, job_id: str):
        db = SessionLocal()
        job = db.query(DbVideoGenJob).filter(DbVideoGenJob.id == job_id).first()
        if not job: return
        job.status = "running"
        db.commit()
        try:
            planned = VideoOrchestrator.plan_clips(job.prompt, job.duration_seconds)
            job.total_clips = len(planned)
            db.commit()
            temp_dir = settings.GENERATED_DIR / job_id
            temp_dir.mkdir(parents=True, exist_ok=True)
            clips = []
            local_image = None
            if job.image_url:
                local_image = temp_dir / f"ref{Path(job.image_url).suffix}"
                storage_manager.download_to_local(job.image_url, local_image)
            for idx, prompt in enumerate(planned):
                db.refresh(job)
                if job.status == "cancelled": raise InterruptedError()
                clip = VideoOrchestrator.generate_single_clip_subprocess(idx, prompt, job.task_type, job.resolution, temp_dir, str(local_image) if local_image else None, mock_mode=settings.MOCK_MODE)
                clips.append(clip)
                job.completed_clips = idx + 1
                db.commit()
            final_video = settings.GENERATED_DIR / f"final_{job_id}.mp4"
            VideoOrchestrator.stitch_clips_ffmpeg(clips, final_video)
            url = storage_manager.upload_generated_video(final_video)
            job.status = "completed"
            job.output_video_url = url
            db.commit()
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        finally: db.close()

    def _run_fine_tuning_task(self, job_id: str):
        db = SessionLocal()
        job = db.query(DbFineTuneJob).filter(DbFineTuneJob.id == job_id).first()
        if not job: return
        job.status = "running"
        log_path = settings.LOGS_DIR / f"ft_{job_id}.log"
        job.log_file_path = str(log_path)
        db.commit()
        try:
            extracted = settings.UPLOAD_DIR / f"extracted_{job_id}"
            extracted.mkdir(parents=True, exist_ok=True)
            if job.dataset_path.endswith(".zip"):
                import zipfile
                with zipfile.ZipFile(job.dataset_path, 'r') as z:
                    z.extractall(extracted)
            if settings.MOCK_MODE:
                with open(log_path, "w") as f:
                    for e in range(1, job.epochs + 1):
                        time.sleep(1)
                        f.write(f"Epoch [{e}/{job.epochs}] - Loss: {0.5/e:.4f} - Progress: {e/job.epochs*100:.1f}%\n")
                        f.flush()
                        job.progress = e/job.epochs*100
                        db.commit()
                weights = settings.STORAGE_DIR / f"checkpoints_{job_id}" / "wan2.2_lora.safetensors"
                weights.parent.mkdir(parents=True, exist_ok=True)
                with open(weights, "w") as f: f.write("mock weights")
            else:
                cmd = ["accelerate", "launch", "train_wan_lora.py", "--model_name_or_path", "./Wan2.2-TI2V-5B", "--dataset_dir", str(extracted), "--output_dir", str(settings.STORAGE_DIR/f"checkpoints_{job_id}"), "--trigger_word", job.trigger_word, "--learning_rate", str(job.learning_rate), "--epochs", str(job.epochs)]
                with open(log_path, "w") as out:
                    p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, text=True)
                    self.active_jobs[job_id] = p
                    p.wait()
                weights = settings.STORAGE_DIR / f"checkpoints_{job_id}" / "wan2.2_lora.safetensors"
            url = storage_manager.upload_lora_weights(weights)
            job.status = "completed"
            job.progress = 100.0
            job.output_video_url = url
            db.commit()
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        finally: db.close()

task_manager = BackgroundTaskManager()
EOF

# =====================================================================
# WRITE FILE: api/main.py
# =====================================================================
cat << 'EOF' > api/main.py
import os, sys, time, uuid, shutil, logging, psutil, subprocess
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .config import settings
from .database import get_db, init_db
from .models import DbFineTuneJob, DbVideoGenJob, FineTuneStartRequest, FineTuneJobResponse, VideoGenStartRequest, VideoGenJobResponse, SystemHealthResponse
from .storage import storage_manager
from .tasks import task_manager

app = FastAPI(title="Wan2.2 API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static/generated", StaticFiles(directory=str(settings.GENERATED_DIR)), name="static_generated")

@app.on_event("startup")
def startup(): init_db()

@app.get("/api/v1/system/health", response_model=SystemHealthResponse)
def health():
    gpu_available = False
    gpu_devices = []
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits"], text=True)
        gpu_available = True
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            gpu_devices.append({"name": parts[0], "vram_total_mb": float(parts[1]), "vram_used_mb": float(parts[2]), "vram_free_mb": float(parts[1])-float(parts[2]), "temperature_c": int(parts[3])})
    except Exception: pass
    return {"status": "healthy", "cpu_usage_percent": psutil.cpu_percent(), "memory_used_gb": psutil.virtual_memory().used/(1024**3), "memory_total_gb": psutil.virtual_memory().total/(1024**3), "gpu_available": gpu_available, "gpu_devices": gpu_devices}

@app.post("/api/v1/finetune/upload")
def upload(file: UploadFile = File(...)):
    fid = str(uuid.uuid4())
    url = storage_manager.save_uploaded_file(file.file, f"{fid}.zip")
    return {"dataset_id": fid, "storage_uri": url}

@app.post("/api/v1/finetune/start", response_model=FineTuneJobResponse)
def ft_start(req: FineTuneStartRequest, db: Session = Depends(get_db)):
    job = DbFineTuneJob(id=f"ft_{uuid.uuid4().hex[:8]}", dataset_path=str(settings.UPLOAD_DIR/f"{req.dataset_id}.zip"), trigger_word=req.trigger_word, learning_rate=req.learning_rate, epochs=req.epochs, batch_size=req.batch_size, resolution=req.resolution)
    db.add(job); db.commit(); db.refresh(job)
    task_manager.submit_fine_tuning(job.id)
    return job

@app.get("/api/v1/finetune/status/{job_id}", response_model=FineTuneJobResponse)
def ft_status(job_id: str, db: Session = Depends(get_db)):
    return db.query(DbFineTuneJob).filter(DbFineTuneJob.id == job_id).first()

@app.post("/api/v1/generate/start", response_model=VideoGenJobResponse)
def gen_start(req: VideoGenStartRequest, db: Session = Depends(get_db)):
    job = DbVideoGenJob(id=f"gen_{uuid.uuid4().hex[:8]}", prompt=req.prompt, task_type=req.task_type, resolution=req.resolution, duration_seconds=req.duration_seconds, total_clips=max(1, req.duration_seconds//5))
    db.add(job); db.commit(); db.refresh(job)
    task_manager.submit_video_generation(job.id)
    return job

@app.get("/api/v1/generate/status/{job_id}", response_model=VideoGenJobResponse)
def gen_status(job_id: str, db: Session = Depends(get_db)):
    return db.query(DbVideoGenJob).filter(DbVideoGenJob.id == job_id).first()
EOF

# =====================================================================
# WRITE FILE: api/requirements_api.txt
# =====================================================================
cat << 'EOF' > api/requirements_api.txt
fastapi>=0.110.0
uvicorn>=0.28.0
sqlalchemy>=2.0.0
psutil>=5.9.8
pydantic>=2.6.0
requests>=2.31.0
google-cloud-storage>=2.15.0
python-multipart
decord
peft
sentencepiece
EOF

# =====================================================================
# WRITE FILE: deploy/Dockerfile
# =====================================================================
cat << 'EOF' > deploy/Dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
ENV PYTHONUNBUFFERED=1 DEBIAN_FRONTEND=noninteractive
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git ffmpeg libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements_animate.txt requirements_s2v.txt ./
COPY api/requirements_api.txt ./api/
RUN pip install --no-cache-dir --upgrade pip && sed -i '/flash_attn/d' requirements.txt && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -r api/requirements_api.txt && pip install --no-cache-dir --no-build-isolation flash_attn
COPY . .
EXPOSE 8000
RUN mkdir -p api_storage/uploads api_storage/generated api_storage/logs
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# =====================================================================
# WRITE FILE: deploy/kubernetes.yaml
# =====================================================================
cat << 'EOF' > deploy/kubernetes.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: wan-video-platform
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: wan-checkpoints-pvc
  namespace: wan-video-platform
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: premium-rwo
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wan-api-deployment
  namespace: wan-video-platform
  labels:
    app: wan-api
spec:
  replicas: 1
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: wan-api
  template:
    metadata:
      labels:
        app: wan-api
    spec:
      containers:
        - name: wan-api-container
          image: us-central1-docker.pkg.dev/cs-26delhi-lab2-10830/wan-video-repo/wan2.2-api:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8000
          env:
            - name: WAN_API_HOST
              value: "0.0.0.0"
            - name: WAN_API_PORT
              value: "8000"
            - name: WAN_DATABASE_URL
              value: "sqlite:////app/api_storage/api_jobs.db"
            - name: WAN_API_MOCK
              value: "True"
            - name: GCP_PROJECT_ID
              valueFrom:
                configMapKeyRef:
                  name: gcp-config
                  key: project-id
            - name: WAN_GCS_BUCKET
              valueFrom:
                configMapKeyRef:
                  name: gcp-config
                  key: bucket-name
          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
            limits:
              memory: "32Gi"
              cpu: "8"
              nvidia.com/gpu: "1"
          volumeMounts:
            - name: storage-volume
              mountPath: /app/api_storage
            - name: checkpoints-volume
              mountPath: /app/models
      volumes:
        - name: storage-volume
          persistentVolumeClaim:
            claimName: wan-checkpoints-pvc
        - name: checkpoints-volume
          persistentVolumeClaim:
            claimName: wan-checkpoints-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: wan-api-service
  namespace: wan-video-platform
spec:
  type: LoadBalancer
  selector:
    app: wan-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: gcp-config
  namespace: wan-video-platform
data:
  project-id: "cs-26delhi-lab2-10830"
  bucket-name: "wan-video-storage-10830"
EOF


echo "====== [Step 3/6] Authenticating and Building Docker Container ======"
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# Let's create dummy base requirements if not present to avoid docker build failure
touch requirements.txt requirements_animate.txt requirements_s2v.txt

docker build \
  -t us-central1-docker.pkg.dev/$PROJECT_ID/wan-video-repo/wan2.2-api:latest \
  -f deploy/Dockerfile .

echo "Pushing the Docker image to Artifact Registry (this may take a few minutes)..."
docker push us-central1-docker.pkg.dev/$PROJECT_ID/wan-video-repo/wan2.2-api:latest

echo "====== [Step 4/6] Launching GKE Cluster with NVIDIA L4 GPU ======"
# Removed bad arg '--install-nvidia-gpu-driver'. Pre-configured standard installation trigger
# G2 machines default to GPU enablement.
gcloud container clusters create wan-video-cluster \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=g2-standard-8 \
    --num-nodes=1 \
    --accelerator=type=nvidia-l4,count=1

echo "====== [Step 5/6] Linking GKE Credentials & Deploying ======"
gcloud container clusters get-credentials wan-video-cluster --zone=$ZONE

echo "Deploying API to GKE Kubernetes cluster..."
kubectl apply -f deploy/kubernetes.yaml

echo "====== [Step 6/6] SETUP COMPLETE! ======"
echo "Your Wan2.2 API cluster has been deployed successfully!"
echo "Run the following command to find your public Load Balancer external IP:"
echo "    kubectl get svc -n wan-video-platform wan-api-service"
