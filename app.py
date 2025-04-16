# app.py
import torch
torch.set_num_threads(27)  # Try 2–4, experiment with what runs faster

import gradio as gr
from models import ask_riley
from riley_genesis import RileyCore
from TTS.api import TTS
import tempfile

# Initialize core engine and TTS
riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# app.p
def chat_interface(history, user_input):
    if user_input.startswith("!mode"):
        _, mode = user_input.split()
        return history + [{"role": "system", "content": riley.set_mode(mode)}], "", None, history

    if user_input.startswith("!personality"):
        _, profile = user_input.split()
        return history + [{"role": "system", "content": riley.set_personality(profile)}], "", None, history

    context_prompt = riley.think(user_input)
    response_raw = ask_riley(context_prompt)
    response = response_raw.replace('\n', ' ').replace('\r', '').strip()

    if "User:" in response:
        response = response.split("User:")[0].strip()

    riley.remember(f"Riley: {response}")
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path)

    return history, "", audio_path, history

# UI style
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

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🧪 RILEY-AI: Genesis Core")
    gr.Markdown("### Fixed Memory | Voice Output Enabled")

    chatbot = gr.Chatbot(label="Riley Terminal", elem_classes="chatbox", type='messages')
    msg = gr.Textbox(label="Ask or command Riley...")
    audio = gr.Audio(label="Riley's Voice", interactive=False)
    clear = gr.Button("Clear Chat")
    state = gr.State([])

    msg.submit(chat_interface, [state, msg], [chatbot, msg, audio, state])
    clear.click(lambda: ([], "", None, []), None, [chatbot, msg, audio, state])

if __name__ == "__main__":
    demo.launch()