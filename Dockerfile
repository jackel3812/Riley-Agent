FROM python
CMD pip install TTS 0.22.0
CMD pipe python.to("cpu")  # Ensure model runs on CPU and python
WORKDIR /app
COPY . .
Run pip install sudo 
RUN sudo install TTS
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
CMD pip install tts
RUN pip install gtts 
RUN pip install flask
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install keras
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860


