import os
import re
import tempfile
import torch
import multiprocessing
import gradio as gr
from models import ask_riley
from riley_genesis import RileyCore
from TTS.api import TTS

# Fix font and cache permissions for Hugging Face
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["FONTCONFIG_PATH"] = "/usr/share/fonts"
os.makedirs("/tmp/mplconfig", exist_ok=True)
os.makedirs("/tmp/.cache", exist_ok=True)

# Set writable cache directory
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp"
os.environ["XDG_CONFIG_HOME"] = "/tmp"


tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# Max out CPU threads safely
torch.set_num_threads(max(1, multiprocessing.cpu_count()))
torch.set_num_interop_threads(max(1, multiprocessing.cpu_count() // 2))

# Load Riley core + TTS voice
riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def chat_interface(history, user_input):
    if user_input.startswith("!mode"):
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            mode = parts[1]
            return history + [{"role": "system", "content": riley.set_mode(mode)}], "", None, history

    if user_input.startswith("!personality"):
        parts = user_input.split(maxsplit=1)
        if len(parts) > 1:
            profile = parts[1]
            return history + [{"role": "system", "content": riley.set_personality(profile)}], "", None, history

    # Generate response from Riley
    context_prompt = riley.think(user_input)
    response_raw = ask_riley(context_prompt)

    # Clean up response
    response = re.sub(r'[\n\r\\]', ' ', response_raw).strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()

    riley.remember(f"Riley: {response}")

    # Append to history, limiting memory bloat
    history = history[-30:]  # keep only last 30 turns
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    # Synthesize voice
    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path)

    return history, "", audio_path, history

# CSS
css = """
body { background: #0b0f1e; color: #00ffff; font-family: 'Orbitron', sans-serif; }
.gradio-container {
    border: 2px solid #ffaa00; background: linear-gradient(145deg, #000000, #0c1440);
    box-shadow: 0 0 25px #ffaa00; padding: 25px; border-radius: 20px;
}
button {
    background-color: #0c1440; color: #ffaa00; border: 2px solid #ffaa00; border-radius: 8px;
}
button:hover { background-color: #ffaa00; color: black; }
.chatbox {
    background-color: #111; color: #00ffff; border: 1px solid #00ffff; padding: 10px; height: 450px;
}
"""

# Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🧬 RILEY-AI: Genesis Core (Phi-2 Patched)")
    gr.Markdown("### Fixed Memory | No Looping | Voice Enabled")

    chatbot = gr.Chatbot(label="Riley Terminal", elem_classes="chatbox", type='messages')
    msg = gr.Textbox(label="Ask or command Riley...")
    audio = gr.Audio(label="Riley’s Voice", interactive=False)
    clear = gr.Button("Clear Chat")
    state = gr.State([])

   msg.submit(chat_interface, [state, msg], [chatbot, msg, audio])
   clear.click(lambda: ([], "", None, []), None, [chatbot, msg, audio, state])


if __name__ == "__main__":
   demo.launch(share=True, show_api=False)

