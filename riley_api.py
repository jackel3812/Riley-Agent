from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import datetime
import webbrowser

lemmatizer = WordNetLemmatizer()
model = load_model("mymodel.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)

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
    return "I'm not sure I understand. Can you try asking another way?"

@app.route("/", methods=["GET"])
def home():
    return "🧠 Riley API is up. POST to /chat with JSON { 'message': 'Hi' }"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    ints = predict_class(user_input)
    response = get_response(ints, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

def get_riley_response(message):
    ints = predict_class(message)
    response = get_response(ints, intents)
    return response


