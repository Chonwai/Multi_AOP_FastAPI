# Multi-stage Dockerfile for Multi-AOP FastAPI Application
# Using conda for RDKit installation (recommended approach)

# ============================================
# Stage 1: Build stage - Install all dependencies
# ============================================

FROM continuumio/miniconda3:latest AS builder

# Set working directory
WORKDIR /build

# Install system dependencies required for RDKit and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set conda environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Create conda environment for the application
RUN conda create -n app python=3.10 -y && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "app", "/bin/bash", "-c"]

# Install RDKit via conda-forge (recommended way for production)
RUN conda install -c conda-forge rdkit -y && \
    conda clean -afy

# Copy requirements file
COPY requirements.txt /build/requirements.txt

# Install Python dependencies via pip (excluding rdkit-pypi since we use conda RDKit)
# Create a temporary requirements file without rdkit-pypi and xlstm
# xlstm will be installed separately to handle platform-specific mlstm_kernels dependency
RUN grep -v "rdkit-pypi" /build/requirements.txt | grep -v "^# xLSTM" | grep -v "xlstm" > /build/requirements_base.txt || true

# Install base Python dependencies (excluding xlstm)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /build/requirements_base.txt

# Install xlstm dependencies first (these are needed by xlstm)
# Based on xlstm package dependencies from PyPI
RUN pip install --no-cache-dir \
    einops \
    omegaconf \
    transformers \
    dacite \
    ftfy \
    ninja \
    huggingface-hub \
    rich \
    tokenizers \
    seaborn \
    joypy \
    ipykernel || true

# Try to install mlstm_kernels (optional, may fail on ARM64/aarch64)
# If this fails, xlstm will automatically use native PyTorch kernels instead
# This is a soft failure - we continue even if mlstm_kernels cannot be installed
RUN pip install --no-cache-dir mlstm_kernels 2>&1 | tee /tmp/mlstm_install.log || \
    echo "INFO: mlstm_kernels not available for this platform (ARM64), xlstm will use native PyTorch kernels"

# Install xlstm
# Strategy: Try normal installation first, if it fails due to mlstm_kernels dependency,
# install with --no-deps and rely on already installed dependencies
# xlstm will work with native PyTorch kernels if mlstm_kernels is not available
RUN pip install --no-cache-dir "xlstm>=2.0.2,<3.0.0" || \
    (echo "WARNING: xlstm installation failed (likely due to mlstm_kernels), retrying without dependency check" && \
     pip install --no-cache-dir --no-deps "xlstm>=2.0.2,<3.0.0") && \
    python -c "import xlstm; print('xlstm installed successfully')" && \
    echo "xlstm installation completed"

# ============================================
# Stage 2: Runtime stage - Minimal image
# ============================================

FROM continuumio/miniconda3:latest AS runtime

# Set working directory
WORKDIR /app

# Install minimal system dependencies for runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set conda environment variables
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy conda environment from builder stage
COPY --from=builder /opt/conda/envs/app /opt/conda/envs/app

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Copy application code
COPY app/ /app/app/
COPY predict/ /app/predict/

# Copy final_model directory if it exists (optional, for training artifacts)
# Note: This will fail if directory doesn't exist. Remove this line if not needed.
# COPY final_model/ /app/final_model/

# Note: Model files are included in the image for Render deployment
# For local development with docker-compose, you can use volume mount instead
COPY predict/model/ /app/predict/model/

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check (using curl if available, otherwise python)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD conda run -n app python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Default command (using uvicorn for production)
# For simple testing, you can override with:
#   docker run ... python run_simple.py
# Or set CMD to use Python directly:
#   CMD ["conda", "run", "-n", "app", "python", "run_simple.py"]
# 
# Note: Use ${PORT:-8000} to support Render's PORT environment variable (default: 10000)
# This allows the app to work on both local (8000) and Render (10000) environments
CMD ["conda", "run", "-n", "app", "sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
