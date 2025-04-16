import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import gradio as gr
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import pickle
import json

# Prepare
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()

model = load_model("mymodel.h5")
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# NLP logic
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

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
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints):
    tag = ints[0]['intent'] if ints else 'noanswer'
    for i in intents['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])
    return "I'm not sure I understand."

def riley_chat(message, history=[]):
    ints = predict_class(message)
    response = get_response(ints)
    history.append((message, response))
    return history, ""

# Gradio 4.x interface
demo = gr.ChatInterface(
    fn=riley_chat,
    title="🧠 Riley AI - Intent Chatbot",
    description="Ask Riley anything. She's trained and ready!",
    retry_btn="🔁 Retry",
    undo_btn="↩️ Undo",
    theme="soft"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
