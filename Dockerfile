FROM python
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
RUN pip install gtts 
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860

# Install NVIDIA CUDA Toolkit
RUN pip apt update
RUN pip apt install nvidia-cuda-toolkit nvidia-driver-535 -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118




Run sudo reboot

