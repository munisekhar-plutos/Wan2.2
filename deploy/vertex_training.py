import os
import logging
from typing import Optional

try:
    from google.cloud import aiplatform
    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False

logger = logging.getLogger("wan_api.vertex_deploy")

def submit_vertex_lora_training(
    project_id: str,
    location: str,
    bucket_name: str,
    dataset_id: str,
    trigger_word: str,
    learning_rate: float = 1e-4,
    epochs: int = 10,
    batch_size: int = 1,
    resolution: str = "720p",
    gpu_type: str = "NVIDIA_TESLA_A100", # Can be NVIDIA_L4 or NVIDIA_TESLA_A100 (80GB)
    gpu_count: int = 1
) -> Optional[str]:
    """
    Submits a fully-managed custom training job to Vertex AI Custom Training on GCP.
    This offloads heavy model fine-tuning from our Web VMs to a managed cloud cluster,
    autoscaling compute down to zero as soon as training is completed.
    """
    if not VERTEX_AVAILABLE:
        logger.error("The 'google-cloud-aiplatform' package is not installed. Run 'pip install google-cloud-aiplatform' to use Vertex AI.")
        return None

    # Initialize the Vertex AI SDK
    aiplatform.init(project=project_id, location=location, staging_bucket=f"gs://{bucket_name}")

    # Docker image used to execute training (contains our codebase + accelerate config)
    # This should be your custom container registry path
    container_uri = f"gcr.io/{project_id}/wan2.2-api:latest"

    # Define the arguments that will be passed into the container
    # Since our container CLI accepts options, we can run training directly
    args = [
        "accelerate", "launch", "train_wan_lora.py",
        "--model_name_or_path", f"gs://{bucket_name}/base_models/Wan2.2-TI2V-5B",
        "--dataset_dir", f"gs://{bucket_name}/uploads/{dataset_id}.zip",
        "--output_dir", f"gs://{bucket_name}/models/checkpoints_{dataset_id}",
        "--trigger_word", trigger_word,
        "--learning_rate", str(learning_rate),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--resolution", "704" if resolution == "720p" else "480",
        "--mixed_precision", "bf16"
    ]

    # Map the friendly GPU string to Vertex machine specs
    # For A100: use standard a2-highgpu-1g machine (equipped with A100 40GB/80GB)
    # For L4: use g2-standard-8 machine (equipped with L4 24GB)
    if "A100" in gpu_type:
        machine_type = f"a2-highgpu-{gpu_count}g" if gpu_count <= 8 else "a2-highgpu-8g"
    else:
        machine_type = f"g2-standard-{8 * gpu_count}"

    logger.info(f"Submitting Custom Job to Vertex AI on machine: {machine_type} with {gpu_count}x {gpu_type}...")

    try:
        # Create and submit the Custom Container Training Job
        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"wan2.2_lora_{dataset_id[:8]}",
            container_uri=container_uri,
            command=["/bin/bash", "-c", " ".join(args)],
        )

        model = job.run(
            model_display_name=f"wan2.2_lora_model_{dataset_id[:8]}",
            replica_count=1,
            machine_type=machine_type,
            accelerator_type=gpu_type,
            accelerator_count=gpu_count,
            sync=False # Run asynchronously so the API worker doesn't block!
        )
        
        # Return the job resource ID
        logger.info(f"Vertex AI Custom Training Job submitted successfully: {job.resource_name}")
        return job.resource_name
        
    except Exception as e:
        logger.exception(f"Failed to submit training job to Vertex AI: {e}")
        raise e
