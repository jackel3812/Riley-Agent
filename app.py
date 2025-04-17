import os
import torch
import multiprocessing
import gradio as gr
from TTS.api import TTS
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

# Define your Gradio interface here
def greet(name):
    return f"Hello {name}!"

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()
