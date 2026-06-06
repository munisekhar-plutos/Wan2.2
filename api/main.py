import os
import sys
import time
import uuid
import shutil
import logging
import psutil
import subprocess
from datetime import datetime
from typing import List, Optional
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

# Local imports
from .config import settings
from .database import get_db, init_db
from .models import (
    DbFineTuneJob, DbVideoGenJob, 
    FineTuneStartRequest, FineTuneJobResponse,
    VideoGenStartRequest, VideoGenJobResponse,
    SystemHealthResponse
)
from .storage import storage_manager
from .tasks import task_manager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)] if hasattr(sys, 'stdout') else []
)
logger = logging.getLogger("wan_api.server")

# Create FastAPI instance with Swagger Metadata
app = FastAPI(
    title="Wan2.2 Production API",
    description="Enterprise-grade REST API for running fine-tuning (LoRA) and high-performance video generation up to 10 minutes on GCP.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web UI clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local generated folder as a static path so that videos can be fetched locally if GCS is disabled
app.mount("/static/generated", StaticFiles(directory=str(settings.GENERATED_DIR)), name="static_generated")

@app.on_event("startup")
def startup_event():
    """Trigger DB initialization on server launch."""
    logger.info("Initializing persistent database...")
    init_db()
    logger.info("Wan2.2 API Server launched and listening.")


# ==========================================
# 1. SYSTEM HEALTH DIAGNOSTICS
# ==========================================

@app.get("/api/v1/system/health", response_model=SystemHealthResponse, tags=["Diagnostics"])
def get_system_health():
    """
    Returns real-time server diagnostics: CPU, RAM, and GPU/VRAM telemetry.
    Excellent for tracking GCP utilization and preventing OOM failures.
    """
    cpu_percent = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    
    gpu_available = False
    gpu_devices = []
    
    # Try parsing nvidia-smi for actual GPU telemetry
    try:
        # Run query: name, memory.total, memory.used, temperature.gpu
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits"]
        output = subprocess.check_output(cmd, text=True)
        
        gpu_available = True
        for line in output.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_devices.append({
                    "name": parts[0],
                    "vram_total_mb": float(parts[1]),
                    "vram_used_mb": float(parts[2]),
                    "vram_free_mb": float(parts[1]) - float(parts[2]),
                    "temperature_c": int(parts[3])
                })
    except Exception:
        # Fallback if non-NVIDIA machine or nvidia-smi not in path
        gpu_available = False
        
    return {
        "status": "healthy",
        "cpu_usage_percent": cpu_percent,
        "memory_used_gb": round(mem.used / (1024**3), 2),
        "memory_total_gb": round(mem.total / (1024**3), 2),
        "gpu_available": gpu_available,
        "gpu_devices": gpu_devices
    }


# ==========================================
# 2. FINE-TUNING (LoRA) ENDPOINTS
# ==========================================

@app.post("/api/v1/finetune/upload", status_code=status.HTTP_201_CREATED, tags=["Fine-Tuning"])
def upload_finetune_dataset(file: UploadFile = File(...)):
    """
    Ingests and uploads fine-tuning assets.
    Accepts a ZIP archive of images/videos with corresponding .txt caption files.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only compressed ZIP files (.zip) are supported for datasets.")
        
    # Generate unique ID for this dataset upload
    dataset_id = str(uuid.uuid4())
    filename = f"{dataset_id}.zip"
    
    try:
        # Save file (Uploads to Cloud Storage GCS if configured, else keeps locally)
        saved_uri = storage_manager.save_uploaded_file(file.file, filename)
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "storage_uri": saved_uri,
            "message": "Dataset uploaded successfully. Use 'dataset_id' to start training."
        }
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process dataset file: {str(e)}")


@app.post("/api/v1/finetune/start", response_model=FineTuneJobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Fine-Tuning"])
def start_finetune_job(req: FineTuneStartRequest, db: Session = Depends(get_db)):
    """
    Registers and triggers an asynchronous LoRA fine-tuning subprocess.
    """
    # 1. Locate dataset file
    dataset_file = settings.UPLOAD_DIR / f"{req.dataset_id}.zip"
    
    # Check if dataset exists locally or in GCS
    if not dataset_file.exists():
        # Check if we can pull from GCS bucket
        if settings.GCS_ENABLED:
            try:
                gcs_uri = f"gs://{settings.GCS_BUCKET_NAME}/uploads/{req.dataset_id}.zip"
                storage_manager.download_to_local(gcs_uri, dataset_file)
            except Exception:
                raise HTTPException(status_code=404, detail=f"Dataset with ID {req.dataset_id} was not found on local disk or GCP Storage.")
        else:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {req.dataset_id} was not found on disk.")

    # 2. Register job in persistent database
    job_id = f"ft_{uuid.uuid4().hex[:8]}"
    db_job = DbFineTuneJob(
        id=job_id,
        status="pending",
        model_type=req.model_type,
        dataset_path=str(dataset_file),
        trigger_word=req.trigger_word,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        batch_size=req.batch_size,
        resolution=req.resolution
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    # 3. Dispatch to background thread queue
    task_manager.submit_fine_tuning(job_id)
    
    return db_job


@app.get("/api/v1/finetune/status/{job_id}", response_model=FineTuneJobResponse, tags=["Fine-Tuning"])
def get_finetune_job_status(job_id: str, db: Session = Depends(get_db)):
    """Retrieves real-time progress, loss history, and weights URL of a fine-tuning job."""
    job = db.query(DbFineTuneJob).filter(DbFineTuneJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found.")
    return job


@app.get("/api/v1/finetune/logs/{job_id}", tags=["Fine-Tuning"])
def stream_finetune_logs(job_id: str, db: Session = Depends(get_db)):
    """
    Streams the live raw terminal outputs of an active fine-tuning subprocess.
    Perfect for debugging and keeping tabs on exact loss rates.
    """
    job = db.query(DbFineTuneJob).filter(DbFineTuneJob.id == job_id).first()
    if not job or not job.log_file_path:
        raise HTTPException(status_code=404, detail=f"Log file not found or training has not started for job {job_id}.")
        
    log_path = Path(job.log_file_path)
    if not log_path.exists():
        return StreamingResponse(iter(["Training started, waiting for log buffer to initialize...\n"]), media_type="text/plain")

    def log_generator():
        # Keep reading lines from the log file
        with open(log_path, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    # If training completed or failed, we exit generator
                    # Otherwise sleep briefly and check for new log writes
                    db.refresh(job)
                    if job.status not in ("pending", "running"):
                        # Read any final trailing lines
                        trailing = f.read()
                        if trailing:
                            yield trailing
                        break
                    time.sleep(0.5)

    return StreamingResponse(log_generator(), media_type="text/plain")


@app.get("/api/v1/finetune/jobs", response_model=List[FineTuneJobResponse], tags=["Fine-Tuning"])
def list_finetune_jobs(db: Session = Depends(get_db)):
    """Lists all registered fine-tuning jobs in descending order."""
    return db.query(DbFineTuneJob).order_by(DbFineTuneJob.created_at.desc()).all()


@app.post("/api/v1/finetune/cancel/{job_id}", tags=["Fine-Tuning"])
def cancel_finetune_job(job_id: str, db: Session = Depends(get_db)):
    """Gracefully terminates an active running fine-tuning subprocess."""
    job = db.query(DbFineTuneJob).filter(DbFineTuneJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job {job_id} not found.")
        
    if job.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in state: {job.status}")
        
    success = task_manager.cancel_job(job_id)
    if success:
        job.status = "cancelled"
        db.commit()
        return {"status": "success", "message": f"Fine-tuning job {job_id} was successfully cancelled."}
    else:
        # Subprocess wasn't running, but we mark it as cancelled/failed anyway
        job.status = "cancelled"
        db.commit()
        return {"status": "success", "message": f"Job {job_id} state was set to cancelled."}


# ==========================================
# 3. VIDEO GENERATION ENDPOINTS
# ==========================================

@app.post("/api/v1/generate/start", response_model=VideoGenJobResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Video Generation"])
def start_video_generation(req: VideoGenStartRequest, db: Session = Depends(get_db)):
    """
    Enqueues an asynchronous request for video generation.
    Supports durations up to 10 minutes (600 seconds) by stitching clips sequentially.
    """
    if req.duration_seconds <= 0:
        raise HTTPException(status_code=400, detail="Duration must be a positive integer greater than zero.")
    if req.duration_seconds > 600:
        raise HTTPException(status_code=400, detail="Max supported duration for single request is 600 seconds (10 minutes).")

    # Standard clips are 5 seconds long
    total_clips = max(1, req.duration_seconds // 5)

    # Register job in DB
    job_id = f"gen_{uuid.uuid4().hex[:8]}"
    db_job = DbVideoGenJob(
        id=job_id,
        status="pending",
        task_type=req.task_type,
        prompt=req.prompt,
        resolution=req.resolution,
        duration_seconds=req.duration_seconds,
        total_clips=total_clips,
        completed_clips=0,
        image_url=req.image_url,
        audio_url=req.audio_url
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    # Submit to serialized task manager
    task_manager.submit_video_generation(job_id)

    return db_job


@app.get("/api/v1/generate/status/{job_id}", response_model=VideoGenJobResponse, tags=["Video Generation"])
def get_generation_job_status(job_id: str, db: Session = Depends(get_db)):
    """Retrieves current execution status, progress, segments generated, and output URL."""
    job = db.query(DbVideoGenJob).filter(DbVideoGenJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Generation job {job_id} not found.")
    return job


@app.get("/api/v1/generate/jobs", response_model=List[VideoGenJobResponse], tags=["Video Generation"])
def list_generation_jobs(db: Session = Depends(get_db)):
    """Lists all registered video generation jobs."""
    return db.query(DbVideoGenJob).order_by(DbVideoGenJob.created_at.desc()).all()


@app.post("/api/v1/generate/cancel/{job_id}", tags=["Video Generation"])
def cancel_generation_job(job_id: str, db: Session = Depends(get_db)):
    """Cancels video generation and gracefully stops active clip assembly."""
    job = db.query(DbVideoGenJob).filter(DbVideoGenJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Generation job {job_id} not found.")
        
    if job.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in state: {job.status}")
        
    # Cancel active execution
    task_manager.cancel_job(job_id)
    job.status = "cancelled"
    db.commit()
    return {"status": "success", "message": f"Generation job {job_id} cancelled."}
