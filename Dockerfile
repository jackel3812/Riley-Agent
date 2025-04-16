FROM pytho
WORKDIR /app
COPY . .
RUN pip install gtts 
RUN pip install flask
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install keras
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860


