import gradio as gr
import os

from riley_api import get_riley_response  # 👈 Make sure this file exists

# Riley AI voice response
def submit_message(message):
    # Get response from Riley core logic
    response = get_riley_response(message)


    # Return structured chat and audio
    return [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response}
    ], audio_path

# Build the Gradio interface
def build_interface():
    with gr.Blocks(css="static/style.css") as demo:
        gr.Markdown("# 🧬 Riley-AI Voice Interface")

        chatbot = gr.Chatbot(type="messages", label="Chat with Riley")
        user_input = gr.Textbox(label="Type your message here")
        audio_output = gr.Audio(label="Riley’s Voice", interactive=False)
        submit_btn = gr.Button("Send")

        submit_btn.click(
            submit_message,
            inputs=user_input,
            outputs=[chatbot, audio_output]
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()


