# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.9-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install pip and upgrade it
RUN pip install --upgrade pip

# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the application code
COPY --chown=user . /app

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet

# Command to run the server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]