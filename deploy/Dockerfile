# Use official PyTorch GPU runtime image with CUDA 12.1
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set system-level environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Set workspace directory
WORKDIR /app

# Install system dependencies (FFmpeg is critical for lossless video stitching)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first to optimize Docker layer caching
COPY requirements.txt requirements_animate.txt requirements_s2v.txt ./
COPY api/requirements_api.txt ./api/

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    sed -i '/flash_attn/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r api/requirements_api.txt && \
    pip install --no-cache-dir --no-build-isolation flash_attn

# Copy the rest of the application codebase
COPY . .

# Expose the default FastAPI port
EXPOSE 8000

# Create directories for persistent and temporary storage
RUN mkdir -p api_storage/uploads api_storage/generated api_storage/logs

# Command to launch the API server under Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
