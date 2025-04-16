FROM python
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"
RUN pip install TTS==0.22.0
CMD ["python", "app.py"]
EXPOSE 7860


