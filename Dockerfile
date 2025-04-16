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
FROM python:3.9


# Install necessary packages
RUN pip install --no-cache-dir some-package

# Add your application code
COPY . /app

# Set resource limits for the container
# Note: This is typically done in the Docker run command or docker-compose.yml, not in the Dockerfile
# Example of setting limits in docker-compose.yml:
# services:
#   your_service:
#     deploy:
#       resources:
#         limits:
#           cpus: '0.1'  # 100mb CPU equivalent
#           memory: 100M

# Command to run your application