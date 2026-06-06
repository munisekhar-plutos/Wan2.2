# 🚀 Wan2.2 GCP Production Deployment Guide

This directory contains the configurations and scripts required to containerize, test, and deploy the **Wan2.2 FastAPI Service** on **Google Cloud Platform (GCP)** at scale.

---

## 📂 Configuration Files Summary

* **`Dockerfile`**: A CUDA 12.1-enabled runtime image compiling all system drivers, FFmpeg (for video stitching), PyTorch 2.4.0, and API scripts.
* **`docker-compose.yaml`**: Local container testing setup supporting NVIDIA GPU pass-through.
* **`kubernetes.yaml`**: GKE production manifests (Namespace, PVC, ConfigMap, LoadBalancer Service, Deployment with `nvidia.com/gpu` requests).
* **`vertex_training.py`**: A python manager to trigger fully managed fine-tuning on **GCP Vertex AI Custom Jobs** utilizing autoscaling training nodes.

---

## 🛠️ Phase 1: Containerizing the Application

To run the API on GKE, you must build the Docker container and push it to GCP **Artifact Registry** or **Google Container Registry (GCR)**.

### 1. Enable Artifact Registry on GCP
Ensure the API is enabled in your cloud project:
```bash
gcloud services enable artifactregistry.googleapis.com
```

### 2. Create a Docker Repository
Create a repository in your desired region (e.g. `us-central1`):
```bash
gcloud artifacts repositories create wan-video-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for Wan2.2 API"
```

### 3. Build & Tag the Image
From the root of the `Wan2.2` repository:
```bash
# Authenticate your local Docker client to GCP
gcloud auth configure-docker us-central1-docker.pkg.dev

# Build and tag the image
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/wan-video-repo/wan2.2-api:latest -f deploy/Dockerfile .
```

### 4. Push the Image to GCP
```bash
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/wan-video-repo/wan2.2-api:latest
```

---

## 🐳 Phase 2: Local GPU Testing (Docker Compose)

Before deploying to GKE, you can test the containerized API on a local GPU rig or a raw GCP VM instance.

### Prerequisites:
Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your system.

### Spin up the service:
```bash
# Run in background
docker compose -f deploy/docker-compose.yaml up -d
```
*Navigating to `http://localhost:8000/docs` will show the fully functional, containerized API.*

---

## ☸️ Phase 3: Deploying on Google Kubernetes Engine (GKE)

Deploying on GKE enables automatic load balancing, multi-node scaling, and node auto-provisioning.

### 1. Create a GKE Autopilot or Standard Cluster with GPUs
Create a cluster with **NVIDIA L4** (24GB) or **A100** (80GB) GPUs:
```bash
gcloud container clusters create wan-video-cluster \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --num-nodes=1 \
    --accelerator=type=nvidia-l4,count=1
```

### 2. Configure credentials
```bash
gcloud container clusters get-credentials wan-video-cluster --zone=us-central1-a
```

### 3. Update Manifest variables
Open `deploy/kubernetes.yaml`:
1. Replace `gcr.io/YOUR_GCP_PROJECT/wan2.2-api:latest` with your Artifact Registry image URL.
2. Under ConfigMap (`gcp-config`), set your actual `project-id` and `bucket-name`.

### 4. Apply manifests to GKE
```bash
kubectl apply -f deploy/kubernetes.yaml
```

### 5. Obtain external IP
Wait for the external load balancer to allocate an IP:
```bash
kubectl get svc -n wan-video-platform wan-api-service
```
*Your production endpoints are now live at `http://<EXTERNAL_IP>/docs`!*

---

## 🧠 Phase 4: Scaling Fine-Tuning via Google Vertex AI

Instead of executing fine-tuning inside a standard GKE Pod (which wastes GKE pool resources and risk container preemption), use **Vertex AI Custom Training**:

1. Submit training jobs asynchronously via `deploy/vertex_training.py` using `submit_vertex_lora_training()`.
2. This provisions a dedicated single-node GPU server (e.g. an `a2-highgpu-1g` with 1x A100-80GB) only for the duration of the training.
3. Once fine-tuning is completed, the node spins down to **zero**, costing you nothing in idle time.
4. The final LoRA weights (`.safetensors`) are output directly to your GCP bucket (`gs://your-bucket-name/models/`).
