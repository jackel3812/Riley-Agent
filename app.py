import gradio as gr
import os
import tempfile
 
# Environment configuration to avoid permission errors
os.environ["NUMBA_DISABLE_CACHE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = tempfile.gettempdir()
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()
os.environ["TTS_CACHE_DIR"] = tempfile.gettempdir()
os.environ["HOME"] = tempfile.gettempdir()
 
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
        background-color: #111; color: #00ff
    }
    """
 
    with gr.Blocks(css=css) as demo:
        gr.Markdown("## Chat with Riley")
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(placeholder="Type your message here...")
        submit_btn = gr.Button("Send")
 
        history = []
 
        def submit_message(user_input):
            nonlocal history
            history, _, audio_path, _ = chat_interface(history, user_input)
            return history, audio_path
 
        submit_btn.click(submit_message, inputs=user_input, outputs=[chatbot, "audio"])
 
    return demo
 
# Main entry point to run the server
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()