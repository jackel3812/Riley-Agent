FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
RUN pip install TTS==0.22.0
CMD ["python", "app.py"]
Run # Use CPU version of PyTorch
torch==2.1.0+cpu
torchaudio==2.1.0+cpu
torchvision==0.16.0+cpu
--extra-index-url https://download.pytorch.org/whl/cpu



