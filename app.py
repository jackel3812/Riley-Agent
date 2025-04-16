import gradio as gr
from TTS.api import TTS

# Initialize Text-to-Speech
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def submit_message(message):
    # Generate a reply (you can make this smarter)
    response = f"Riley says: {message}"

    # Synthesize audio
    audio_path = "output.wav"
    tts.tts_to_file(text=response, file_path=audio_path)

    # Return the message and audio
    return [{"role": "user", "content": message}, {"role": "assistant", "content": response}], audio_path

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

if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
