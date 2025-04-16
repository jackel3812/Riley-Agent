import gradio as gr
from gtts import gTTS
import os

# ⬇️ Define the function FIRST so it's available when needed
def submit_message(message):
    response = f"Riley says: {message}"
    
    # Convert response to speech
    tts = gTTS(text=response, lang='en', slow=False)
    audio_path = "/tmp/riley_response.mp3"
    tts.save(audio_path)
    
    # Return structured chat message and audio
    return [{"role": "user", "content": message}, {"role": "assistant", "content": response}], audio_path

# ⬇️ Then define the interface
def build_interface():
    with gr.Blocks(css="static/style.css") as demo:
        gr.Markdown("# 🧬 Riley-AI Voice Assistant")

        chatbot = gr.Chatbot(type="messages", label="Chat with Riley")
        user_input = gr.Textbox(label="Type your message here")
        audio_output = gr.Audio(label="Riley's Voice Output", interactive=False)
        submit_btn = gr.Button("Send")

        submit_btn.click(
            submit_message,
            inputs=user_input,
            outputs=[chatbot, audio_output]
        )

    return demo

# ⬇️ Run the app
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
