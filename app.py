"""
BrgyLink AI - Multilingual Intelligent Virtual Assistant Server
Barangay Bagong Pag-asa, San Jacinto, Pangasinan.
Fluent in: Pangasinan (Kabaleyan), Ilocano (Ti Samtoy), Tagalog (Filipino), and English.
"""

import json
import os
import time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from smart_classifier import (
    MODEL_VERSION,
    detect_language,
    handle_message,
    load_model,
    new_session,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)

# Load Smart Classifier model at startup
print(f"Loading BrgyLink Smart Classifier v{MODEL_VERSION}...")
classifier_model = load_model()
print("🎉 BrgyLink AI Multilingual Engine loaded and ready for chat!")

# User session cache
sessions: dict[str, dict] = {}

LANG_DISPLAY = {
    "pangasinan": "Pangasinan",
    "pag": "Pangasinan",
    "ilocano": "Ilocano",
    "ilo": "Ilocano",
    "tagalog": "Tagalog / Filipino",
    "fil": "Tagalog / Filipino",
    "english": "English",
    "en": "English",
}


@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/health', methods=['GET'])
def health():
    intents_count = len(classifier_model.get("responses", {}))
    return jsonify({
        "status": "healthy",
        "engine": "smart_classifier_v5",
        "model_version": MODEL_VERSION,
        "supported_languages": [
            {"code": "pag", "name": "Pangasinan"},
            {"code": "ilo", "name": "Ilocano"},
            {"code": "fil", "name": "Tagalog"},
            {"code": "en", "name": "English"},
        ],
        "intents_count": intents_count,
        "patterns_count": 982,
        "jurisdiction": "Barangay Bagong Pag-asa, San Jacinto, Pangasinan",
    })


@app.route('/api/languages', methods=['GET'])
def get_languages():
    return jsonify({
        "languages": [
            {
                "code": "pag",
                "name": "Pangasinan",
                "label": "Pangasinan (Kabaleyan)",
                "sample_query": "Mano so bayad na clearance tan permit?",
                "greeting": "Maabig ya agew kabaleyan! Panon ta kan natulongan ed saray serbisyo na barangay hall?",
            },
            {
                "code": "ilo",
                "name": "Ilocano",
                "label": "Ilocano (Ti Samtoy)",
                "sample_query": "Kasano ti agala ti barangay clearance para iti trabaho?",
                "greeting": "Naimbag nga aldaw kailian! Kasanu ka a matulongan kadagiti serbisyo ti barangay hall?",
            },
            {
                "code": "fil",
                "name": "Tagalog",
                "label": "Tagalog (Filipino)",
                "sample_query": "Paano kumuha ng Barangay Clearance at ano ang mga requirements?",
                "greeting": "Magandang araw po! Paano kita matutulungan sa mga serbisyo ng ating barangay hall?",
            },
            {
                "code": "en",
                "name": "English",
                "label": "English",
                "sample_query": "What are the office hours and requirements for certificates?",
                "greeting": "Hello! How can I assist you with barangay certificates, permits, and services today?",
            },
        ]
    })


@app.route('/api/services', methods=['GET'])
def get_services():
    kb_path = os.path.join(BASE_DIR, 'knowledge_base.json')
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            return jsonify(kb.get('services', []))
    return jsonify([])


@app.route('/chat', methods=['POST'])
def chat():
    t0 = time.time()
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get('message', '').strip()
        session_id = str(data.get('session_id') or request.remote_addr or 'default')
        forced_lang = data.get('language')  # Optional: "pag", "ilo", "fil", "en"

        if not user_message:
            return jsonify({'error': 'Please enter a question or message.'}), 400

        # Retrieve or initialize user session
        if session_id not in sessions:
            sessions[session_id] = new_session()
        session = sessions[session_id]

        # Apply forced language preference if provided
        if forced_lang:
            norm = forced_lang.lower().strip()
            if norm in {"pag", "pangasinan"}:
                session["preferred_language"] = "pangasinan"
            elif norm in {"ilo", "ilocano"}:
                session["preferred_language"] = "ilocano"
            elif norm in {"fil", "tagalog", "tl"}:
                session["preferred_language"] = "tagalog"
            elif norm in {"en", "english"}:
                session["preferred_language"] = "english"

        # Process message via Smart Classifier
        result = handle_message(user_message, session=session, model=classifier_model)
        elapsed_ms = round((time.time() - t0) * 1000, 2)

        lang_code = result.get('language', 'english')
        lang_name = LANG_DISPLAY.get(lang_code, lang_code.capitalize())

        return jsonify({
            'response': result['response'],
            'intent': result.get('intent', 'unknown'),
            'confidence': result.get('similarity', 1.0),
            'language': lang_code,
            'language_name': lang_name,
            'processing_time_ms': elapsed_ms,
        })

    except Exception as e:
        print(f"Error handling chat: {e}", flush=True)
        return jsonify({'error': 'Failed to process chat message', 'details': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting BrgyLink AI Multilingual Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
