# Use an official lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Create a working directory
WORKDIR /app

# Copy requirement files
COPY requirements.txt .

# Install system dependencies and Python dependencies
RUN apt-get update && \
    apt-get install -y git ffmpeg libgl1 libglib2.0-0 && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy entire project into the container
COPY . .

# Expose port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "store.main:app", "--host", "0.0.0.0", "--port", "8000"]
