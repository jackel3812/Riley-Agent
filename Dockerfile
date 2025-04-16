FROM python
RUN pip install TTS-0.22.0-cp311-cp311-manylinux1_x86_64.whl
CMD pipe python.to("cpu")  # Ensure model runs on CPU and python
WORKDIR /app
COPY . .
Run pip install sudo 
RUN sudo install TTS
RUN pip install Git 
RUN python3 -m pip install --upgrade pip
RUN Upgrade pip and install your TTS wheel first
RUN git clone https://github.com/coqui-ai/TTS
RUN pip install -e .[all,dev,notebooks]  # Select the relevant extras


# Install remaining project dependencies
RUN pip install -r requirements.txt

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


