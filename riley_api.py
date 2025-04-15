import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


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

# Initialize lemmatizer and load chatbot components
lemmatizer = WordNetLemmatizer()
model = load_model("mymodel.h5")  # Load trained model
intents = json.loads(open("intents.json").read())  # Load intents
words = pickle.load(open("words.pkl", "rb"))  # Vocabulary file
classes = pickle.load(open("classes.pkl", "rb"))  # Classes file

# Initialize Flask app
app = Flask(__name__)

# Preprocessing user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize words
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  # Initialize empty bag
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:  # Match word against vocabulary
                bag[i] = 1
    return np.array(bag)

# Predict intent
def predict_class(sentence):
    p = bow(sentence, words)  # Convert sentence to BoW
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

# Get response based on intent
def get_response(ints, intents_json):
    tag = ints[0]['intent'] if ints else 'noanswer'  # Default response
    for i in intents_json['intents']:
        if i['tag'] == tag:  # Match tag to response
            return random.choice(i['responses'])
    return "I'm not sure I understand. Can you try asking another way?"

# Define API endpoints
@app.route("/", methods=["GET"])
def home():
    return "🧠 Riley API is up. POST to /chat with JSON { 'message': 'Hi' }"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")  # Get user input from JSON
    ints = predict_class(user_input)  # Predict intent
    response = get_response(ints, intents)  # Get response
    return jsonify({"response": response})

# Running the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

# Helper function for chatbot response
def get_riley_response(message):
    ints = predict_class(message)  # Predict intent
    response = get_response(ints, intents)  # Get response
    return response
