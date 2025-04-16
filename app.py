import gradio as gr

# Proper OpenAI-style message format
def submit_message(user_input, chat_history):
    if not chat_history:
        chat_history = []
    chat_history.append({"role": "user", "content": user_input})

    # Dummy AI response — replace with real logic
    ai_response = f"Riley says: {user_input[::-1]}"
    chat_history.append({"role": "assistant", "content": ai_response})

    return chat_history, ""

# Build the Gradio interface
def build_interface():
    with gr.Blocks(css="static/style.css") as demo:
        gr.Markdown("# 🧬 Riley-AI Chat Interface")

        chatbot = gr.Chatbot(label="Chat with Riley", type="messages")
        user_input = gr.Textbox(label="Type your message here")
        state = gr.State([])  # Chat history state
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
