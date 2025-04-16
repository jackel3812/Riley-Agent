FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles

ARG PYTHON_VERSION=3.10.13
ARG USE_PERSISTENT_DATA

RUN apt-get update && apt-get install -y \
    git git-lfs \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

RUN curl https://pyenv.run | bash
ENV PATH=$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH

RUN pyenv install $PYTHON_VERSION && \
    pyenv global $PYTHON_VERSION && \
    pyenv rehash && \
    python -m venv $HOME/.venv

ENV PATH="$HOME/.venv/bin:$PATH"
RUN echo 'source $HOME/.venv/bin/activate' >> $HOME/.bashrc

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /tmp
COPY --chown=user:user ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

RUN pip install InvokeAI --use-pep517 --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR $HOME/app

ENV INVOKEAI_ROOT=${USE_PERSISTENT_DATA:+/data/invokeai}
ENV INVOKEAI_ROOT=${INVOKEAI_ROOT:-$HOME/invokeai}

COPY --chown=user:user ./start.sh $HOME/start.sh
COPY --chown=user:user invokeai.yaml $HOME/invokeai.yaml

RUN chmod +x $HOME/start.sh

EXPOSE 9090

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/home/user/start.sh"]