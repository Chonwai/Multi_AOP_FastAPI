# Docker Configuration for Multi-AOP FastAPI

This directory contains Docker configuration files for containerizing the Multi-AOP FastAPI application.

## Files

- `Dockerfile`: Main Dockerfile using conda for RDKit installation (recommended for production)
- `Dockerfile.pip`: Alternative Dockerfile using pip/rdkit-pypi (simpler but may have compatibility issues)
- `docker-compose.yml`: Docker Compose configuration for local development and deployment

## Building the Image

### Using conda-based Dockerfile (Recommended)

```bash
docker build -f docker/Dockerfile -t multi-aop-api:latest ..
```

### Using pip-based Dockerfile (Alternative)

```bash
docker build -f docker/Dockerfile.pip -t multi-aop-api:latest ..
```

## Running with Docker Compose

1. Copy `.env.example` to `.env` and configure environment variables:
```bash
cp .env.example .env
```

2. Start the service:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

3. Check logs:
```bash
docker-compose -f docker/docker-compose.yml logs -f
```

4. Stop the service:
```bash
docker-compose -f docker/docker-compose.yml down
```

## Running with Docker

```bash
docker run -d \
  --name multi-aop-api \
  -p 8000:8000 \
  -e MODEL_PATH=predict/model/best_model_Oct13.pth \
  -e DEVICE=cpu \
  -v $(pwd)/predict/model:/app/predict/model:ro \
  multi-aop-api:latest
```

## Environment Variables

See `.env.example` for all available environment variables.

Key variables:
- `API_HOST`: API host address (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `MODEL_PATH`: Path to model file (default: predict/model/best_model_Oct13.pth)
- `DEVICE`: Device for inference (cpu/cuda, default: cpu)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR, default: INFO)
- `ENVIRONMENT`: Environment (development/production, default: production)

## Health Check

The container includes a health check that monitors `/health` endpoint:
- Interval: 30 seconds
- Timeout: 10 seconds
- Start period: 40 seconds
- Retries: 3

## Security

- Container runs as non-root user (`appuser`)
- Minimal base image (miniconda3)
- Multi-stage build reduces image size
- Only necessary ports exposed

## Troubleshooting

### RDKit Import Error

If you encounter RDKit import errors:
1. Ensure you're using the conda-based Dockerfile
2. Check that conda environment is properly activated
3. Verify RDKit installation: `conda list | grep rdkit`

### Model Not Found

If the model file is not found:
1. Ensure model file exists in `predict/model/` directory
2. Check `MODEL_PATH` environment variable
3. Verify volume mount if using Docker Compose

### Port Already in Use

If port 8000 is already in use:
1. Change `API_PORT` in `.env` file
2. Update port mapping in `docker-compose.yml`
3. Update `EXPOSE` in Dockerfile if needed

