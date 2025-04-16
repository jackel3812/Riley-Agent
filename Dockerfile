FROM python
CMD pipe.to("cpu")  # Ensure model runs on CPU and python
WORKDIR /app
COPY . .
Run pip install sudo 
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
Run pip install TTS 3.9
RUN pip install gtts 
RUN pip install flask
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN 
RUN pip install keras
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860


