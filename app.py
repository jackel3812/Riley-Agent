import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import pickle
import json
import logging
import random

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setup lemmatizer and model
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

# Load model and data
try:
    model = load_model("mymodel.h5")
    intents = json.loads(open("intents.json").read())
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    logger.info("✅ Model and data loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model or data: {e}")
    raise SystemExit("Cannot start without model and data.")

# App init
app = Flask(__name__, template_folder="templates", static_folder="static")

# NLP functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent'] if ints else 'noanswer'
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm not sure I understand."

def get_riley_response(message):
    ints = predict_class(message)
    return get_response(ints, intents)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    response = get_riley_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
