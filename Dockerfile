# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

RUN pip install -r requirements.txt
RUN run: pip install --upgrade pip
RUN pip install -r requirements.txt python app.py
Run pip install flask numpy nltk keras tensorflow gradio TTS
Run python -m nltk.downloader punkt wordnet
COPY --chown=user . /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
