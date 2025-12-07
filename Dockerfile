# Use a slim Python base image
FROM python:3.10-slim

# System deps (for Pillow and image handling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
# For CPU-only torch/torchvision, use extra index URL
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY app.py /app/app.py
COPY blip.py /app/blip.py
# Copy your trained weights into the image (optional)
# Alternatively, mount them at runtime
COPY miniblip_epoch9.pt /app/miniblip_epoch9.pt

# Expose port
EXPOSE 8000

# Environment variables
ENV MODEL_PATH=/app/miniblip_epoch9.pt
ENV PYTHONUNBUFFERED=1

# Start the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
