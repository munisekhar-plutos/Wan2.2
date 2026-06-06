import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from datetime import timedelta
from .config import settings

# Attempt to import google cloud storage client.
# Fall back gracefully if not installed.
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
                    logger.info(f"Connected successfully to GCS bucket: {settings.GCS_BUCKET_NAME}")
                except Exception as e:
                    logger.warning(f"Failed to initialize GCS client: {e}. Falling back to local storage.")
            else:
                logger.warning("WAN_GCS_BUCKET is configured, but 'google-cloud-storage' package is not installed. Falling back to local storage.")

    def save_uploaded_file(self, source_file, target_filename: str) -> str:
        """
        Saves an uploaded multipart file to local uploads directory first.
        If GCS is enabled, uploads it to GCS.
        Returns the path or GCS URI.
        """
        local_path = settings.UPLOAD_DIR / target_filename
        
        # Save locally first
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(source_file, buffer)
            
        if self.bucket:
            try:
                gcs_blob_name = f"uploads/{target_filename}"
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(str(local_path))
                logger.info(f"Uploaded dataset {target_filename} to GCS: gs://{settings.GCS_BUCKET_NAME}/{gcs_blob_name}")
                return f"gs://{settings.GCS_BUCKET_NAME}/{gcs_blob_name}"
            except Exception as e:
                logger.error(f"Failed to upload to GCS: {e}. Returning local path.")
                return str(local_path)
                
        return str(local_path)

    def upload_generated_video(self, local_video_path: Path) -> str:
        """
        Uploads a generated video file to Cloud Storage (if enabled).
        Returns a download link (either a GCS signed URL or a local static URL).
        """
        if not local_video_path.exists():
            raise FileNotFoundError(f"Local video file not found at: {local_video_path}")
            
        filename = local_video_path.name
        
        if self.bucket:
            try:
                gcs_blob_name = f"generated/{filename}"
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(str(local_video_path))
                logger.info(f"Uploaded video {filename} to GCS bucket: {settings.GCS_BUCKET_NAME}")
                
                # Generate a signed URL valid for 7 days
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(days=7),
                    method="GET"
                )
                return signed_url
            except Exception as e:
                logger.error(f"Failed to upload video to GCS: {e}. Falling back to local URL.")
                
        # Return fallback local relative path which can be served by FastAPI static mounting
        return f"/static/generated/{filename}"

    def upload_lora_weights(self, local_weights_path: Path) -> str:
        """
        Uploads a trained LoRA .safetensors model to Cloud Storage (if enabled).
        Returns either a GCS URI or a local path.
        """
        if not local_weights_path.exists():
            raise FileNotFoundError(f"Local weight file not found at: {local_weights_path}")
            
        filename = local_weights_path.name
        
        if self.bucket:
            try:
                gcs_blob_name = f"models/lora/{filename}"
                blob = self.bucket.blob(gcs_blob_name)
                blob.upload_from_filename(str(local_weights_path))
                logger.info(f"Uploaded LoRA weights {filename} to GCS bucket: {settings.GCS_BUCKET_NAME}")
                return f"gs://{settings.GCS_BUCKET_NAME}/{gcs_blob_name}"
            except Exception as e:
                logger.error(f"Failed to upload weights to GCS: {e}. Falling back to local.")
                
        return str(local_weights_path)

    def download_to_local(self, remote_url_or_path: str, target_local_path: Path) -> Path:
        """
        Helper to pull external files (e.g. references or weights) down to local disk.
        Supports standard HTTP/S links, gs:// URIs, and local copies.
        """
        if remote_url_or_path.startswith("gs://"):
            if not self.bucket:
                raise ValueError("Cannot download gs:// URI because GCS is not configured.")
            # Parse gs:// URI
            path_parts = remote_url_or_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_name = path_parts[1] if len(path_parts) > 1 else ""
            
            blob = self.gcs_client.bucket(bucket_name).blob(blob_name)
            target_local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target_local_path))
            return target_local_path
            
        elif remote_url_or_path.startswith(("http://", "https://")):
            # Stream import requests inside method to avoid global dependency issues
            import requests
            response = requests.get(remote_url_or_path, stream=True)
            response.raise_for_status()
            target_local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_local_path, "wb") as out_file:
                shutil.copyfileobj(response.raw, out_file)
            return target_local_path
            
        else:
            # Assume it's a local path
            src_path = Path(remote_url_or_path)
            if src_path.exists() and src_path != target_local_path:
                target_local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, target_local_path)
            return target_local_path

# Export a single global instance
storage_manager = StorageManager()
