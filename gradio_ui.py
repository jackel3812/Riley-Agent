import gradio as gr
from models import ask_riley

def respond(message, history):
    response = ask_riley(message)
    history.append((message, response))
    return history, ""

with gr.Blocks(css="static/style.css") as demo:
    gr.Markdown("# 🧬 Riley-AI Chat Interface")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    state = gr.State([])

    msg.submit(respond, [msg, state], [chatbot, msg])
    gr.ClearButton([chatbot, msg])

if __name__ == "__main__":
    demo.launch()
