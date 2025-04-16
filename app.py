import gradio as gr

# Simple test function for Riley’s first response
def riley_respond(message):
    return "🧠 Riley says: " + message[::-1]  # Just reverses the message for now

# Set up Gradio interface
demo = gr.Interface(fn=riley_respond, inputs="text", outputs="text", title="Riley AI", description="Your evolving AI companion.")

# Launch app on Hugging Face Spaces
demo.launch(server_name="0.0.0.0", server_port=7860)

