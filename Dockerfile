



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