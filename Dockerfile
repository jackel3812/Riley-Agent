# Use the official Hugging Face Python base image
FROM huggingface/transformers-pytorch-gpu:latest

# Set working directory
WORKDIR /code

# Copy all files to the container
COPY . .

# Install system packages if needed
RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg && \
    apt-get clean

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
