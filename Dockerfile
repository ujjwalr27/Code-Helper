# Multi-stage Dockerfile for AI Code Remediation Service

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY model_handler.py .
COPY rag_retriever.py .

# Copy recipes directory if exists
COPY recipes/ recipes/ 2>/dev/null || true

# Create directories for logs and metrics
RUN mkdir -p /app/logs /app/data

# Environment variables
ENV MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
ENV USE_GPU="false"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# GPU-enabled version (optional)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as gpu

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir faiss-gpu

# Copy application
COPY app.py model_handler.py rag_retriever.py ./
COPY recipes/ recipes/ 2>/dev/null || true

ENV MODEL_NAME="Qwen/Qwen2.5-Coder-1.5B-Instruct"
ENV USE_GPU="true"
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]