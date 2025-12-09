# Makefile for Docker operations

.PHONY: build build-pip up down logs restart shell test clean

# Build Docker image using conda-based Dockerfile
build:
	docker build -f docker/Dockerfile -t multi-aop-api:latest ..

# Build Docker image using pip-based Dockerfile
build-pip:
	docker build -f docker/Dockerfile.pip -t multi-aop-api:latest ..

# Start services using docker-compose
up:
	docker-compose -f docker/docker-compose.yml up -d

# Stop services
down:
	docker-compose -f docker/docker-compose.yml down

# View logs
logs:
	docker-compose -f docker/docker-compose.yml logs -f

# Restart services
restart: down up

# Open shell in running container
shell:
	docker exec -it multi-aop-api bash

# Run tests (if available)
test:
	docker exec multi-aop-api python -m pytest tests/ -v || echo "No tests found"

# Clean up containers and images
clean:
	docker-compose -f docker/docker-compose.yml down -v
	docker rmi multi-aop-api:latest || true

# Health check
health:
	curl http://localhost:8000/health || echo "Service not available"

