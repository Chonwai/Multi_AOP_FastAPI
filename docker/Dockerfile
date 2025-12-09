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
# Create a temporary requirements file without rdkit-pypi
RUN grep -v "rdkit-pypi" /build/requirements.txt > /build/requirements_no_rdkit.txt || true

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /build/requirements_no_rdkit.txt

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
COPY final_model/ /app/final_model/ 2>/dev/null || true

# Copy model file (if exists)
COPY predict/model/ /app/predict/model/ 2>/dev/null || true

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check (using curl if available, otherwise python)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD conda run -n app python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Default command
CMD ["conda", "run", "-n", "app", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
