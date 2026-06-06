from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()

# ==========================================
# 1. DATABASE MODELS (SQLAlchemy)
# ==========================================

class DbFineTuneJob(Base):
    __tablename__ = "finetune_jobs"
    
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed, cancelled
    model_type = Column(String, default="5B")  # 5B, 14B
    dataset_path = Column(String, nullable=True) # local path or GCS URI of zip file
    trigger_word = Column(String, nullable=True)
    learning_rate = Column(Float, default=1e-4)
    epochs = Column(Integer, default=10)
    batch_size = Column(Integer, default=1)
    resolution = Column(String, default="720p") # 480p, 720p
    progress = Column(Float, default=0.0) # 0.0 to 100.0
    metrics_json = Column(JSON, nullable=True) # {"loss_history": [...]}
    error_message = Column(Text, nullable=True)
    log_file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DbVideoGenJob(Base):
    __tablename__ = "videogen_jobs"
    
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="pending")  # pending, running, completed, failed, cancelled
    task_type = Column(String, default="t2v")  # t2v, i2v, ti2v, s2v
    prompt = Column(Text, nullable=True)
    resolution = Column(String, default="720p") # 480p, 720p
    duration_seconds = Column(Integer, default=5)
    total_clips = Column(Integer, default=1)
    completed_clips = Column(Integer, default=0)
    image_url = Column(String, nullable=True) # Input reference image for i2v/ti2v
    audio_url = Column(String, nullable=True) # Input audio for s2v
    output_video_url = Column(String, nullable=True) # Local path or GCS Signed URL
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ==========================================
# 2. PYDANTIC SCHEMAS (API Validation)
# ==========================================

# Fine-Tuning Schemas
class FineTuneStartRequest(BaseModel):
    model_type: str = Field("5B", description="Model variant to fine-tune (5B or 14B)")
    dataset_id: str = Field(..., description="ID of the uploaded dataset zip file")
    trigger_word: str = Field("character_concept", description="The trigger word representing your subject/style")
    learning_rate: float = Field(1e-4, description="Learning rate for LoRA training")
    epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(1, description="Training batch size")
    resolution: str = Field("720p", description="Target training resolution: 480p, 720p")

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

# Video Generation Schemas
class VideoGenStartRequest(BaseModel):
    prompt: str = Field(..., description="The main text prompt for generating video")
    task_type: str = Field("t2v", description="Task type: t2v (Text-to-Video), i2v (Image-to-Video), ti2v (Text-Image-to-Video), s2v (Speech-to-Video)")
    resolution: str = Field("720p", description="Output video resolution: 480p, 720p")
    duration_seconds: int = Field(5, description="Desired video length in seconds (supports up to 600 seconds/10 minutes)")
    image_url: Optional[str] = Field(None, description="Optional image URL for image-to-video or text-image-to-video tasks")
    audio_url: Optional[str] = Field(None, description="Optional audio URL for speech-to-video tasks")

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

# System Health Schema
class SystemHealthResponse(BaseModel):
    status: str
    cpu_usage_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_available: bool
    gpu_devices: List[Dict[str, Any]] = []
