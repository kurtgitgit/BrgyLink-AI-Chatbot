"""Test the locally trained BrgyLink intent classifier without Mission17 integration."""

import json
import os
import pickle
import re

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "data", "intents_brgylink_curated.json")
MODEL_FILE = os.path.join(BASE_DIR, "chatbot_model.h5")
WORDS_FILE = os.path.join(BASE_DIR, "words.pkl")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.pkl")
CONFIDENCE_THRESHOLD = 0.55

# A closed-domain classifier must not force unrelated questions into a barangay
# service intent. These terms are intentionally narrow and based on Mission17's
# documented services and app navigation.
SERVICE_SIGNAL_PATTERN = (
    r"\b(barangay|brgy|brgylink|document|request|clearance|indigency|residency|"
    r"business|permit|good\s+moral|certificate|office|hall|hour|oras|bukas|fee|"
    r"bayad|blotter|incident|insidente|complaint|reklamo|report|magrereport|ulat|ingay|emergency|police|pulis|"
    r"ambulance|sunog|fire|garbage|basura|waste|collection|hakot|schedule|"
    r"hello|hi|kamusta|magandang|thank|thanks|salamat|goodbye|bye|paalam)\b"
)

lemmatizer = WordNetLemmatizer()


def tokenize(message):
    return [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(message)]


def make_bag(message, words):
    tokens = set(tokenize(message))
    return np.array([1 if word in tokens else 0 for word in words])


def load_classifier():
    required_files = (MODEL_FILE, WORDS_FILE, CLASSES_FILE, INTENTS_FILE)
    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        names = ", ".join(os.path.basename(path) for path in missing)
        raise FileNotFoundError(f"Missing {names}. Run: python train.py")

    with open(WORDS_FILE, "rb") as file:
        words = pickle.load(file)
    with open(CLASSES_FILE, "rb") as file:
        classes = pickle.load(file)
    with open(INTENTS_FILE, encoding="utf-8") as file:
        intents = json.load(file)["intents"]

    responses = {item["tag"]: item["responses"][0] for item in intents}
    return load_model(MODEL_FILE), words, classes, responses


def predict(message, model, words, classes, responses):
    if not re.search(SERVICE_SIGNAL_PATTERN, message, re.IGNORECASE):
        return {
            "intent": "fallback",
            "confidence": 0.0,
            "response": "I can help with BrgyLink and Barangay Bagong Pag-asa services. Please check Announcements or contact the official barangay office for other concerns.",
        }

    bag = make_bag(message, words)
    scores = model.predict(np.array([bag]), verbose=0)[0]
    best_index = int(np.argmax(scores))
    confidence = float(scores[best_index])
    intent = classes[best_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "intent": "fallback",
            "confidence": confidence,
            "response": "I am not certain about that. Please check BrgyLink Announcements or contact the official barangay office.",
        }

    return {
        "intent": intent,
        "confidence": confidence,
        "response": responses[intent],
    }


def main():
    model, words, classes, responses = load_classifier()
    print("BrgyLink local classifier test")
    print("Type a question, or type 'quit' to exit.\n")

    while True:
        message = input("You: ").strip()
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue

        result = predict(message, model, words, classes, responses)
        print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
        print(f"BrgyLink AI: {result['response']}\n")


if __name__ == "__main__":
    main()
