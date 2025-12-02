# --------------------------------------------------------------------------------
# Stage 1: Build Stage (Builder)
# Installs dependencies
# --------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Set the working directory in the container
WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY frontend/frontend_requirements.txt ./requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------------------
# Stage 2: Production Stage (Final Image)
# --------------------------------------------------------------------------------
FROM python:3.11-slim

# Set the environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy the core dependencies
COPY --from=builder /usr/local /usr/local

# Copy the rest of the application's code
# Copy the application source code
COPY frontend /app/frontend
COPY utils /app/utils

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["python", "-m", "streamlit", "run", "frontend/app.py"]