import os
import tempfile

# Environment configuration to avoid permission errors
os.environ["NUMBA_DISABLE_CACHE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = tempfile.gettempdir()
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()
os.environ["TTS_CACHE_DIR"] = tempfile.gettempdir()
os.environ["HOME"] = tempfile.gettempdir()

import gradio as gr
from models import ask_riley
from riley_genesis import RileyCore
from TTS.api import TTS

# Initialize Riley and TTS once
riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

# Chat logic
def chat_interface(history, user_input):
    if user_input.startswith("!mode"):
        _, mode = user_input.split()
        return history + [{"role": "system", "content": riley.set_mode(mode)}], "", None, history

    if user_input.startswith("!personality"):
        _, profile = user_input.split()
        return history + [{"role": "system", "content": riley.set_personality(profile)}], "", None, history

    context_prompt = riley.think(user_input)
    response_raw = ask_riley(context_prompt)
    response = response_raw.replace('\n', ' ').replace('\r', '').replace('\\', '').strip()

    if "User:" in response:
        response = response.split("User:")[0].strip()

    riley.remember(f"Riley: {response}")
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    tts.tts_to_file(text=response, file_path=audio_path)

    return history, "", audio_path, history

# Log reader
def read_logs(keyword="", num_lines=100):
    log_path = "./logs.txt"
    if not os.path.exists(log_path):
        return "⚠️ Log file not found."

    with open(log_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    if keyword:
        lines = [line for line in lines if keyword.lower() in line.lower()]

    return "".join(lines[-num_lines:])

# Gradio UI
def build_interface():
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
        gr.Markdown("# 🧬 RILEY-AI: Genesis Core (Phi-2 Patched)")
        gr.Markdown("### Fixed Memory | No Looping | Voice Enabled")

        chatbot = gr.Chatbot(label="Riley Terminal", elem_classes="chatbox", type='messages')
        msg = gr.Textbox(label="Ask or command Riley...")
        audio = gr.Audio(label="Riley’s Voice", interactive=False)
        clear = gr.Button("Clear Chat")
        state = gr.State([])

        msg.submit(chat_interface, [state, msg], [chatbot, msg, audio, state])
        clear.click(lambda: ([], "", None, []), None, [chatbot, msg, audio, state])

        with gr.Tab("📜 Logs"):
            keyword_input = gr.Textbox(label="Search Logs (e.g., error, warning, fail)")
            line_slider = gr.Slider(10, 1000, value=100, label="How many lines to display")
            log_output = gr.Textbox(label="Logs", lines=20, interactive=False)

            refresh_btn = gr.Button("🔄 Refresh Logs")
            refresh_btn.click(fn=read_logs, inputs=[keyword_input, line_slider], outputs=log_output)

    return demo

# Launch the interface
demo = build_interface()
demo.queue().launch(debug=True)
