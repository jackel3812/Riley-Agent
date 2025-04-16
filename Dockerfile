# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.10
RUN useradd -m -u 1000 user


WORKDIR /app



# Copy requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the application code
COPY --chown=user . /app

# Download NLTK data
RUN python -m nltk.downloader punkt wordnet

