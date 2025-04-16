FROM python:3.10-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir gradio
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

RUN pip install -r TTS==0.22.0
CMD ["python", "app.py"]