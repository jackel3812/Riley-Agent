import gradio as gr
import json
import requests

# ✅ Set your Hugging Face Space URL here:
RILEY_URL = "https://huggingface.co/spaces/Zelgodiz/Riley"

def ask_riley(user_input, chat_history):
    try:
        r = requests.post(RILEY_URL, json={"message": user_input})
        r.raise_for_status()
        r_data = r.json()
        bot_reply = r_data.get("response", "⚠️ No reply.")
    except requests.exceptions.HTTPError as http_err:
        bot_reply = f"HTTP error: {http_err}"
    except json.JSONDecodeError:
        bot_reply = "⚠️ Invalid JSON response."
    except Exception as e:
        bot_reply = f"Error: {e}"

    chat_history.append((user_input, bot_reply))
    return chat_history, ""

# ✅ Launch the UI
with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
    gr.Markdown("## 🧬 Riley Genesis Core\nWelcome to your personal AI terminal.")
    chatbot = gr.Chatbot(label="Riley")
    txt = gr.Textbox(label="Ask or command Riley")
    state = gr.State([])

    txt.submit(ask_riley, [txt, state], [chatbot, txt])
    gr.Button("Clear").click(lambda: ([], ""), None, [chatbot, txt])

ui.launch()
