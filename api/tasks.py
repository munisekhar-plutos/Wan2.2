import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.orm import Session
from .config import settings
from .database import SessionLocal
from .models import DbVideoGenJob, DbFineTuneJob
from .orchestrator import VideoOrchestrator
from .storage import storage_manager

logger = logging.getLogger("wan_api.tasks")

class BackgroundTaskManager:
    def __init__(self):
        # We enforce max_workers=1 because training & video generation are GPU intensive.
        # Serializing jobs prevents GPU VRAM Out-Of-Memory (OOM) crashes!
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.active_jobs: Dict[str, subprocess.Popen] = {} # Tracks active subprocesses for cancellation

    def submit_video_generation(self, job_id: str):
        """Submit a video generation job to the serialized worker pool."""
        self.executor.submit(self._run_video_generation_task, job_id)

    def submit_fine_tuning(self, job_id: str):
        """Submit a fine-tuning job to the serialized worker pool."""
        self.executor.submit(self._run_fine_tuning_task, job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Gracefully terminates an active running subprocess (inference or training)."""
        proc = self.active_jobs.get(job_id)
        if proc:
            try:
                proc.terminate()
                # Wait briefly for process to die
                proc.wait(timeout=5)
                logger.info(f"Successfully cancelled active subprocess for job: {job_id}")
                return True
            except subprocess.TimeoutExpired:
                proc.kill()
                logger.warning(f"Forced kill active subprocess for job: {job_id}")
                return True
            except Exception as e:
                logger.error(f"Error terminating process for job {job_id}: {e}")
                return False
        return False

    def _run_video_generation_task(self, job_id: str):
        """Processes video generation sequentially, compiling segments and uploading output."""
        db: Session = SessionLocal()
        job = db.query(DbVideoGenJob).filter(DbVideoGenJob.id == job_id).first()
        if not job:
            logger.error(f"Job ID {job_id} not found in database.")
            db.close()
            return

        logger.info(f"Starting Video Generation Job: {job_id}")
        job.status = "running"
        db.commit()

        try:
            # Plan prompts & clips
            planned_prompts = VideoOrchestrator.plan_clips(job.prompt, job.duration_seconds)
            job.total_clips = len(planned_prompts)
            db.commit()

            # Create a dedicated temp folder for this job's clips
            job_temp_dir = settings.GENERATED_DIR / job_id
            job_temp_dir.mkdir(parents=True, exist_ok=True)

            clips_paths = []
            
            # Download input image/audio if present
            local_image_path = None
            local_audio_path = None
            if job.image_url:
                local_image_path = job_temp_dir / f"input_ref{Path(job.image_url).suffix}"
                storage_manager.download_to_local(job.image_url, local_image_path)
            if job.audio_url:
                local_audio_path = job_temp_dir / f"input_audio{Path(job.audio_url).suffix}"
                storage_manager.download_to_local(job.audio_url, local_audio_path)

            # Generate individual clips
            for idx, prompt in enumerate(planned_prompts):
                # Ensure the job hasn't been cancelled in the meantime
                db.refresh(job)
                if job.status == "cancelled":
                    logger.info(f"Job {job_id} detected as cancelled during clip generation.")
                    raise InterruptedError("Job was cancelled by user.")

                clip_path = VideoOrchestrator.generate_single_clip_subprocess(
                    clip_index=idx,
                    prompt=prompt,
                    task_type=job.task_type,
                    resolution=job.resolution,
                    output_dir=job_temp_dir,
                    image_path=str(local_image_path) if local_image_path else None,
                    audio_path=str(local_audio_path) if local_audio_path else None,
                    mock_mode=settings.MOCK_MODE
                )
                clips_paths.append(clip_path)
                
                job.completed_clips = idx + 1
                db.commit()

            # Stitch completed clips using ffmpeg
            db.refresh(job)
            if job.status == "cancelled":
                raise InterruptedError("Job was cancelled prior to stitching.")

            final_local_video = settings.GENERATED_DIR / f"final_{job_id}.mp4"
            logger.info(f"Stitching {len(clips_paths)} segments for job {job_id}...")
            VideoOrchestrator.stitch_clips_ffmpeg(clips_paths, final_local_video)

            # Upload stitched video to storage
            logger.info(f"Uploading stitched video to production storage...")
            uploaded_url = storage_manager.upload_generated_video(final_local_video)

            # Complete Job
            job.status = "completed"
            job.output_video_url = uploaded_url
            db.commit()
            logger.info(f"Completed Video Generation Job successfully: {job_id}")

            # Cleanup temp directory containing segmented clips
            import shutil
            shutil.rmtree(job_temp_dir, ignore_errors=True)

        except InterruptedError:
            job.status = "cancelled"
            db.commit()
        except Exception as e:
            logger.exception(f"Error during video generation job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        finally:
            db.close()

    def _run_fine_tuning_task(self, job_id: str):
        """Spawns and manages the subprocess executing the LoRA training script."""
        db: Session = SessionLocal()
        job = db.query(DbFineTuneJob).filter(DbFineTuneJob.id == job_id).first()
        if not job:
            logger.error(f"FineTune Job ID {job_id} not found in database.")
            db.close()
            return

        logger.info(f"Starting Fine-Tuning Job: {job_id}")
        job.status = "running"
        
        # Create log file for tracking stdout
        log_file_path = settings.LOGS_DIR / f"finetune_{job_id}.log"
        job.log_file_path = str(log_file_path)
        db.commit()

        try:
            # 1. Unpack dataset zip
            local_zip_path = Path(job.dataset_path)
            dataset_extract_dir = settings.UPLOAD_DIR / f"extracted_{job_id}"
            dataset_extract_dir.mkdir(parents=True, exist_ok=True)
            
            if local_zip_path.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_extract_dir)
                logger.info(f"Dataset extracted successfully to: {dataset_extract_dir}")
            else:
                # If it wasn't a zip, assume it's a directory or folder
                dataset_extract_dir = local_zip_path

            # 2. Setup training command
            # For production-grade execution, we spawn a python subprocess that runs DiffSynth or AI-toolkit training.
            # Here we structure a robust subprocess wrapper.
            
            if settings.MOCK_MODE:
                # In Mock Mode, we simulate a standard LoRA training loop inside a Python subprocess
                mock_train_script = settings.STORAGE_DIR / f"mock_train_{job_id}.py"
                with open(mock_train_script, "w") as f:
                    f.write(f"""
import time, sys
print("Starting Mock Fine-Tuning for Job: {job_id}...", flush=True)
print("Trigger Word: {job.trigger_word}", flush=True)
print("Epochs: {job.epochs}", flush=True)
for epoch in range(1, {job.epochs} + 1):
    time.sleep(1.5)
    loss = 0.5 / epoch + 0.05
    print(f"Epoch [{{epoch}}/{job.epochs}] - Loss: {{loss:.4f}} - Progress: {{epoch / {job.epochs} * 100:.1f}%", flush=True)
print("Training Complete! Saving checkpoints...", flush=True)
""")
                cmd = ["python", str(mock_train_script)]
            else:
                # REAL Training Command
                # This launches the standard accelerate/diffusers LoRA script.
                # In custom environments, we invoke PyTorch accelerate launch
                cmd = [
                    "accelerate", "launch", "train_wan_lora.py", # Custom script path or package binary
                    "--model_name_or_path", str(settings.BASE_DIR / "Wan2.2-TI2V-5B" if job.model_type == "5B" else settings.BASE_DIR / "Wan2.2-T2V-A14B"),
                    "--dataset_dir", str(dataset_extract_dir),
                    "--output_dir", str(settings.STORAGE_DIR / f"checkpoints_{job_id}"),
                    "--trigger_word", job.trigger_word,
                    "--learning_rate", str(job.learning_rate),
                    "--epochs", str(job.epochs),
                    "--batch_size", str(job.batch_size),
                    "--resolution", "720" if job.resolution == "720p" else "480",
                    "--mixed_precision", "bf16"
                ]

            logger.info(f"Launching Fine-Tuning Subprocess: {' '.join(cmd)}")
            
            # Start process and stream logs to file
            with open(log_file_path, "w", encoding="utf-8") as log_file:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(settings.BASE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, # Merge stderr and stdout
                    text=True,
                    bufsize=1
                )
                
                # Save process reference for cancellation
                self.active_jobs[job_id] = proc
                
                # Monitor and parse stdout line by line for metrics
                loss_history = []
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    
                    # Write to log file
                    log_file.write(line)
                    log_file.flush()
                    
                    # Optional: Parse line for progress metrics
                    # e.g., "Epoch [2/10] - Loss: 0.1234 - Progress: 20.0%"
                    if "Progress:" in line:
                        try:
                            parts = line.split("Progress:")
                            prog_str = parts[1].split("%")[0].strip()
                            job.progress = float(prog_str)
                            db.commit()
                        except Exception:
                            pass
                    if "Loss:" in line:
                        try:
                            parts = line.split("Loss:")
                            loss_str = parts[1].split()[0].strip()
                            loss_val = float(loss_str)
                            loss_history.append(loss_val)
                            job.metrics_json = {"loss_history": loss_history}
                            db.commit()
                        except Exception:
                            pass
                            
                proc.wait()
                
            # Cleanup process tracking
            self.active_jobs.pop(job_id, None)

            if proc.returncode == 0:
                # Save & upload output LoRA weights
                lora_output_file = settings.STORAGE_DIR / f"checkpoints_{job_id}" / "wan2.2_lora.safetensors"
                
                # Create mock weight file if in Mock Mode
                if settings.MOCK_MODE:
                    lora_output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(lora_output_file, "w") as f:
                        f.write("mock lora safetensors content")

                if lora_output_file.exists():
                    logger.info("Fine-tuning finished. Uploading trained weights...")
                    gcs_weights_url = storage_manager.upload_lora_weights(lora_output_file)
                    job.status = "completed"
                    job.progress = 100.0
                    job.output_video_url = gcs_weights_url # Reusing field for simplicity as final artifact URL
                else:
                    raise FileNotFoundError(f"Trained weights not found at target: {lora_output_file}")
            elif proc.returncode < 0 or job.status == "cancelled":
                # Process was terminated (cancelled)
                job.status = "cancelled"
            else:
                raise RuntimeError(f"Training subprocess exited with non-zero code: {proc.returncode}")
                
            db.commit()

        except Exception as e:
            logger.exception(f"Error during fine-tuning job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
        finally:
            # Cleanup process reference
            self.active_jobs.pop(job_id, None)
            
            # Delete extracted dataset to conserve local disk space
            if settings.MOCK_MODE and os.path.exists(settings.STORAGE_DIR / f"mock_train_{job_id}.py"):
                os.remove(settings.STORAGE_DIR / f"mock_train_{job_id}.py")
            
            # Clean up checkpoints folder locally if uploaded successfully
            checkpoints_folder = settings.STORAGE_DIR / f"checkpoints_{job_id}"
            if checkpoints_folder.exists() and job.status == "completed":
                import shutil
                shutil.rmtree(checkpoints_folder, ignore_errors=True)
                
            db.close()

# Export single global instance
task_manager = BackgroundTaskManager()
