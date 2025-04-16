import gradio as gr

# Dummy response function (replace with your AI later)
def submit_message(user_input, chat_history):
    response = f"Riley says: {user_input[::-1]}"  # <-- just reverses input for demo
    chat_history.append((user_input, response))
    return chat_history, ""

# Build the Gradio interface
def build_interface():
    with gr.Blocks(css="static/style.css") as demo:
        gr.Markdown("# 🧬 Riley-AI Chat Interface")

        chatbot = gr.Chatbot(label="Chat with Riley")
        user_input = gr.Textbox(label="Type your message here")
        state = gr.State([])  # Holds chat history
        submit_btn = gr.Button("Send")

        submit_btn.click(
            fn=submit_message,
            inputs=[user_input, state],
            outputs=[chatbot, user_input],
        )

    return demo

# Launch the interface
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
