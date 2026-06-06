import os
from pathlib import Path

# Base Directory of the Wan2.2 Project
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings:
    # API Configurations
    HOST: str = os.getenv("WAN_API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("WAN_API_PORT", "8000"))
    
    # DB Configuration
    DATABASE_URL: str = os.getenv("WAN_DATABASE_URL", "sqlite:///api_jobs.db")
    
    # GCP / Google Cloud Storage Configurations
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
    GCS_BUCKET_NAME: str = os.getenv("WAN_GCS_BUCKET", "")
    GCS_ENABLED: bool = bool(GCS_BUCKET_NAME) # Enabled only if bucket name is set
    
    # Local Storage fallbacks (within the repository workspace)
    STORAGE_DIR: Path = BASE_DIR / "api_storage"
    UPLOAD_DIR: Path = STORAGE_DIR / "uploads"
    GENERATED_DIR: Path = STORAGE_DIR / "generated"
    LOGS_DIR: Path = STORAGE_DIR / "logs"
    
    # Wan2.2 Weights & CLI Configurations
    WAN_MODEL_DIR: str = os.getenv("WAN_MODEL_DIR", str(BASE_DIR))
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg") # Path to ffmpeg binary
    
    # Diagnostic / Mock Mode
    # Set to 'True' to run fully mock video generation & fine-tuning for testing on local non-GPU machines
    MOCK_MODE: bool = os.getenv("WAN_API_MOCK", "False").lower() in ("true", "1", "yes")

# Instantiate settings
settings = Settings()

# Ensure all local storage directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
