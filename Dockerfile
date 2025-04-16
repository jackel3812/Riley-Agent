
# create and configure a new process
Run process = multiprocessing.Process(target=python:3.10)

# start the new process
Run process.start(python:3.10)
FROM python:3.10



WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

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

