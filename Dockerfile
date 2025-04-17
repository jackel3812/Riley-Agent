# Use official lightweight Python image
FROM python:3.11-slim

# Set environment variables for cache and config directories
CMD 
ENV XDG_CACHE_HOME=/tmp/.cache \
    XDG_CONFIG_HOME=/tmp/.config \
    XDG_DATA_HOME=/tmp/.local/share \
    MPLCONFIGDIR=/tmp/mplconfig \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    FONTCONFIG_PATH=/usr/share/fonts

# Create necessary directories with appropriate permissions
CMD mkdir -p /tmp/.cache /tmp/.config /tmp/.local/share /tmp/mplconfig /tmp/numba_cache && \ 
CMD chmod -R 777 /tmp/.cache /tmp/.config /tmp/.local/share /tmp/mplconfig /tmp/numba_cache

# Set working directory inside the container
WORKDIR /app


# Copy all files from your current folder to the container
COPY . .

# Upgrade pip and install the TTS wheel first
RUN pip install TTS
 CMD CMD  lask" did not complete successfully: exit code: 
# Install remaining project dependencies
RUN pip install requirements
CMD “Fix the GPU crash” or
CMD “Build my local Riley engine now.”
# Install additional Python packages
RUN pip install gtts
RUN pip install flask 


# Expose application port
EXPOSE 7860

# Define the default command to run your application
CMD ["python", "app.py"]
