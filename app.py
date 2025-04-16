import gradio as gr
import os


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
