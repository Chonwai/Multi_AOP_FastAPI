# Docker Testing Guide

This guide provides instructions for testing the Dockerized Multi-AOP FastAPI application.

## Prerequisites

- Docker installed and running
- Docker Compose installed (optional, for docker-compose testing)
- Model file available at `predict/model/best_model_Oct13.pth`

## Quick Start

### 1. Build the Docker Image

```bash
# Using conda-based Dockerfile (recommended)
docker build -f docker/Dockerfile -t multi-aop-api:latest ..

# Or using pip-based Dockerfile (alternative)
docker build -f docker/Dockerfile.pip -t multi-aop-api:latest ..
```

### 2. Run the Container

```bash
docker run -d \
  --name multi-aop-api \
  -p 8000:8000 \
  -e MODEL_PATH=predict/model/best_model_Oct13.pth \
  -e DEVICE=cpu \
  -v $(pwd)/predict/model:/app/predict/model:ro \
  multi-aop-api:latest
```

### 3. Test Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-19T...",
  "environment": "production",
  "message": ""
}
```

## Testing Checklist

### ✅ Build Tests

- [ ] Docker image builds successfully
- [ ] No build errors or warnings
- [ ] Image size is reasonable (< 5GB for conda-based, < 2GB for pip-based)

### ✅ Container Startup Tests

- [ ] Container starts without errors
- [ ] Container logs show successful startup
- [ ] No import errors for RDKit, PyTorch, etc.
- [ ] Model loads successfully on startup (if pre-loading enabled)

### ✅ Health Check Tests

- [ ] `/health` endpoint returns 200 OK
- [ ] Response includes `model_loaded: true` after model loads
- [ ] Health check passes Docker healthcheck

### ✅ API Endpoint Tests

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKLLVVVFCLVLAAP"}'
```

Expected response:
```json
{
  "sequence": "MKLLVVVFCLVLAAP",
  "prediction": 1,
  "probability": 0.85,
  "confidence": "high",
  "is_aop": true,
  "message": "Prediction completed successfully"
}
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKLLVVVFCLVLAAP", "ACDEFGHIKLMNPQRSTVWY"]}'
```

Expected response:
```json
{
  "total": 2,
  "results": [
    {
      "sequence": "MKLLVVVFCLVLAAP",
      "prediction": 1,
      "probability": 0.85,
      "confidence": "high",
      "is_aop": true
    },
    {
      "sequence": "ACDEFGHIKLMNPQRSTVWY",
      "prediction": 0,
      "probability": 0.23,
      "confidence": "low",
      "is_aop": false
    }
  ],
  "processing_time_seconds": 2.5
}
```

#### Model Info

```bash
curl http://localhost:8000/api/v1/model/info
```

Expected response:
```json
{
  "model_version": "1.0.0",
  "model_path": "predict/model/best_model_Oct13.pth",
  "device": "cpu",
  "seq_length": 50,
  "loaded_at": "2024-12-19T...",
  "is_loaded": true
}
```

### ✅ Error Handling Tests

#### Invalid Sequence

```bash
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "X"}'
```

Expected: 422 Unprocessable Entity with validation error

#### Empty Batch

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"sequences": []}'
```

Expected: 422 Unprocessable Entity with validation error

### ✅ Performance Tests

- [ ] Single prediction completes in < 5 seconds
- [ ] Batch prediction (10 sequences) completes in < 30 seconds
- [ ] Memory usage is reasonable (< 4GB)
- [ ] CPU usage is reasonable during prediction

### ✅ Logging Tests

- [ ] Logs are output to stdout/stderr
- [ ] Log level can be configured via environment variable
- [ ] Logs include request information
- [ ] Error logs include stack traces

## Using Docker Compose

### Start Services

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### View Logs

```bash
docker-compose -f docker/docker-compose.yml logs -f
```

### Stop Services

```bash
docker-compose -f docker/docker-compose.yml down
```

### Using Makefile

```bash
# Build image
make -C docker build

# Start services
make -C docker up

# View logs
make -C docker logs

# Health check
make -C docker health

# Stop services
make -C docker down
```

## Troubleshooting

### RDKit Import Error

**Problem**: `ModuleNotFoundError: No module named 'rdkit'`

**Solution**:
1. Ensure you're using the conda-based Dockerfile
2. Check conda environment: `docker exec multi-aop-api conda list | grep rdkit`
3. Verify RDKit installation: `docker exec multi-aop-api python -c "from rdkit import Chem; print('OK')"`

### Model Not Found

**Problem**: `ModelLoadError: Model file not found`

**Solution**:
1. Check model file exists: `ls -la predict/model/best_model_Oct13.pth`
2. Verify volume mount: `docker exec multi-aop-api ls -la /app/predict/model/`
3. Check MODEL_PATH environment variable

### Port Already in Use

**Problem**: `Error: bind: address already in use`

**Solution**:
1. Change port in `.env` file: `API_PORT=8001`
2. Update docker run command: `-p 8001:8000`
3. Or stop existing container: `docker stop multi-aop-api`

### Container Exits Immediately

**Problem**: Container starts then exits

**Solution**:
1. Check logs: `docker logs multi-aop-api`
2. Verify model file exists
3. Check environment variables
4. Verify all dependencies are installed

## Performance Benchmarks

Expected performance on CPU:

- Single prediction: 1-3 seconds
- Batch prediction (10 sequences): 5-15 seconds
- Batch prediction (100 sequences): 30-60 seconds
- Memory usage: 2-4 GB
- CPU usage: 50-100% during prediction

## Security Checklist

- [ ] Container runs as non-root user
- [ ] Only necessary ports exposed
- [ ] No sensitive data in image
- [ ] Environment variables used for configuration
- [ ] Health check configured
- [ ] Logs don't contain sensitive information

