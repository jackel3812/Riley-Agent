# Use the latest GPU-enabled PyTorch image with Hugging Face support
FROM FROM python:3.10-slim

# Set working directory
WORKDIR /venv

# Install required system libraries for audio + TTS
RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Environment variables to avoid caching issues
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy all app files
COPY . .

# Expose internal port for Hugging Face
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
