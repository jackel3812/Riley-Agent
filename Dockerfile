 Run pip install sudo 
FROM python
WORKDIR /app
COPY . .
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
RUN pip install gtts 
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860
