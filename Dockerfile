FROM python:3.10

WORKDIR /app.py
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.py
CMD ["python", "app.py"]


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

