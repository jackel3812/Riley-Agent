# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies for audio processing
RUN apt update && apt install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all files from your current folder to the container
COPY . .

# Upgrade pip and install the TTS wheel first
RUN pip install --upgrade pip && \
    pip install TTS-0.22.0-cp311-cp311-manylinux1_x86_64.whl

# Install remaining project dependencies
RUN pip install -r requirements.txt
RUN pip install gtts 
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install keras
RUN pip install flask
Cmd docker build -t riley-ai .
CMD docker run -p 7860:7860 riley-ai

# Expose app port (optional, based on your app.py)
EXPOSE 7860

# Start the application
CMD ["python", "app.py"]

CMD import os

CMD os.environ["OMP_NUM_THREADS"] = "5"          # or number of CPU cores you have
CMD os.environ["MKL_NUM_THREADS"] = "5"
CMD os.environ["TF_NUM_INTRAOP_THREADS"] = "3"
CMD os.environ["TF_NUM_INTEROP_THREADS"] = "3"


