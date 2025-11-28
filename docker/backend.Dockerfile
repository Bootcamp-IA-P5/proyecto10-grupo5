# --------------------------------------------------------------------------------
# Stage 1: Build Stage (Builder)
# Installs dependencies and downloads NLTK data
# --------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Define a specific, predictable path for NLTK data.
ENV NLTK_DATA=/app/nltk_data

# 1. Install System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 2. Python Dependencies and NLTK Data
COPY backend/backend_requirements.txt ./requirements.txt

# Install packages and then use the NLTK_DATA environment variable
# to download resources into the defined directory (/app/nltk_data).
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader -d ${NLTK_DATA} stopwords && \
    python -m nltk.downloader -d ${NLTK_DATA} wordnet

# --------------------------------------------------------------------------------
# Stage 2: Production Stage (Final Image)
# --------------------------------------------------------------------------------
FROM python:3.11-slim

# Re-set the environment variable so NLTK knows where to look at RUNTIME
ENV NLTK_DATA=/app/nltk_data
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the core dependencies
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/nltk_data /app/nltk_data

# Copy the application source code
COPY backend /app/backend
COPY resources/models /app/resources/models
COPY utils /app/utils

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]