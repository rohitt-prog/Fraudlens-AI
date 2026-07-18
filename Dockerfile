# ==============================================================================
# FraudLens AI - Dockerfile Placeholder
# ==============================================================================
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Set application directory
WORKDIR /app

# Install standard compiler dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy directories into image
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY README.md .

# Expose API service port (later use)
EXPOSE 8000

# Simple placeholder command for now. Can be updated when API is built.
CMD ["python", "-c", "print('FraudLens AI Docker container environment is ready.')"]
