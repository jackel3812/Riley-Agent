import gradio as gr
import os
# Force Gradio to run properly on Hugging Face Spaces
def riley_respond(message):
    return "🧠 Riley says: " + message[::-1]

demo = gr.Interface(
    fn=riley_respond,
    inputs="text",
    outputs="text",
    title="Riley AI",
    description="Your evolving AI companion."
)

if __name__ == "__main__":
    print("🚀 Launching Riley AI...")
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    os.environ["GRADIO_SERVER_PORT"] = "7860"
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=True, share=False, debug=True)
else:
    demo.launch(server_name="0.0.0.0", server_port=7860)
