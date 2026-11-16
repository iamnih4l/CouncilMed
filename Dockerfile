# CouncilMed Cloud Backend - Hugging Face Spaces Container
# Optimized for free 16 GB RAM CPU tier
FROM python:3.11-slim

# Prevent interactive prompts & bufferless python logs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install system dependencies needed for imaging libraries (libgl, libgomp, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires running as non-root user 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# Install lightweight PyTorch CPU wheel (avoids 4GB CUDA download)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install backend dependencies
COPY --chown=user:user backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application source code, model registry, and evaluation data
COPY --chown=user:user app.py /app/app.py
COPY --chown=user:user backend /app/backend
COPY --chown=user:user model_registry /app/model_registry
COPY --chown=user:user eva-data /app/eva-data

# Create working data & output directories with full user permissions
RUN mkdir -p /app/data/uploads \
             /app/output/previews \
             /app/output/explainability \
             /app/output/reports \
    && chown -R user:user /app

# Switch to non-root user
USER user

# Hugging Face Spaces listens on port 7860
EXPOSE 7860

# Launch Uvicorn server on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
