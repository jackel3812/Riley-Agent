# Import required libraries
from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import datetime
import logging
import gradio as gr


# /app/riley_api.py

def get_riley_response(message):
    # You can make this as advanced as you want
    return f"Hello, I’m Riley. You said: {message}"


# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and data with error handling
try:
    logger.debug("🔄 Loading model...")
    model = load_model("mymodel.h5")
    logger.debug("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise SystemExit("Model loading failed.")

try:
    logger.debug("🔄 Loading intents, words, and classes...")
    intents = json.loads(open("intents.json").read())
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    logger.debug("✅ Data files loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading data files: {e}")
    raise SystemExit("Failed to load intent/data files.")

# Initialize Flask app
app = Flask(__name__)

# Preprocessing user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

# Get response based on intent
def get_response(ints, intents_json):
    tag = ints[0]['intent'] if ints else 'noanswer'
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand. Can you try asking another way?"

# Define API endpoints
@app.route("/", methods=["GET"])
def home():
    return "🧠 Riley API is up. POST to /chat with JSON { 'message': 'Hi' }"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    logger.debug(f"📩 Received message: {user_input}")
    ints = predict_class(user_input)
    response = get_response(ints, intents)
    logger.debug(f"🤖 Responding with: {response}")
    return jsonify({"response": response})

# Helper function for chatbot logic
def get_riley_response(message):
    ints = predict_class(message)
    response = get_response(ints, intents)
    return response

# Gradio interface wrapper
def gradio_chat(message):
    return get_riley_response(message)

demo = gr.Interface(
    fn=gradio_chat,
    inputs=gr.Textbox(label="Ask Riley something..."),
    outputs=gr.Textbox(label="Riley's response"),
    title="🧠 Riley AI Chatbot",
    description="Riley is a smart AI assistant. Ask her anything."
)

# Run Gradio on Hugging Face
if __name__ != "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

# Run Flask if local
if __name__ == "__main__":
    logger.info("🚀 Starting Riley API server...")
    # app.run(host="0.0.0.0", port=5000)  # Optional local use
    
    
    # riley_api.py

    def get_riley_response(message):
    """
    This is the brain of Riley. You can later replace this with GPT, Mistral, or your custom AI.
    """
    # For now, a simple echo response:
    return f"Hello, I'm Riley. You said: '{message}'"