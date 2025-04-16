
FROM huggingface/transformers-pytorch-gpu:latest


WORKDIR Main




# Install system packages needed by TTS/librosa
RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg && \
    apt-get clean

# Set env variables to prevent caching errors in Hugging Face
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --root-user-action=ignore -r requirements.txt


EXPOSE 9000

CMD ["python", "app.py"]

