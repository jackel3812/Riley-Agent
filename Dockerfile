FROM python 3.12
CMD pipe python.to("cpu")  # Ensure model runs on CPU and python 3.12 
WORKDIR /app
COPY . .
CMD cpu(5)
CMD ["python", "app.py"]
RUN pip install --no-cache-dir gradio
CMD pip install tts
RUN pip install gtts 
RUN pip install flask
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install keras
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860


