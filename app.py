import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


import gradio as gr
from transformers import pipeline
from riley_api import get_riley_response as ask_riley
from riley_genesis import RileyCore
import tempfile
from TTS.api import TTS

riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

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

    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path)

    return history, "", audio_path, history

css = "static/style.css"

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

if __name__ == "__main__":
    demo.launch()
