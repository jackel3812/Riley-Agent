from TTS.api import TTS
import os
import torch
import multiprocessing

import tempfile

# Set environment variables to redirect cache and config directories
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["XDG_CONFIG_HOME"] = "/tmp/.config"
os.environ["XDG_DATA_HOME"] = "/tmp/.local/share"
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

# Ensure directories exist
os.makedirs("/tmp/.cache", exist_ok=True)
os.makedirs("/tmp/.config", exist_ok=True)
os.makedirs("/tmp/.local/share", exist_ok=True)
os.makedirs("/tmp/mplconfig", exist_ok=True)

# Set the number of threads for PyTorch
torch.set_num_threads(multiprocessing.cpu_count())

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
      