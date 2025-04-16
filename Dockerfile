FROM python
WORKDIR /app
COPY . .
Run pip install sudo 
RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir gradio
RUN pip install gtts 
RUN pip install flask 
RUN pip install keras
ENV GRADIO_SERVER_NAME="0.0.0.0"
CMD ["python", "app.py"]
EXPOSE 7860
