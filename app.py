"""
BrgyLink AI - Multilingual Intelligent Virtual Assistant Server
Barangay Bagong Pag-asa, San Jacinto, Pangasinan.
Supports: Pangasinan (Kabaleyan), Ilocano (Ti Samtoy), Tagalog (Filipino), and English.
"""

import json
import os
import time
import tempfile
import fcntl
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory, abort, session, redirect
from smart_classifier import (
    MODEL_VERSION,
    detect_language,
    handle_message,
    load_model,
    new_session,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
if os.environ.get("FLASK_ENV") == "production":
    app.config['SESSION_COOKIE_SECURE'] = True
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise ValueError("FLASK_SECRET_KEY is required in production")


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
    import smart_classifier
    kb_index = smart_classifier.load_knowledge_base()
    intents_count = len(kb_index)
    trained_intents_count = len(classifier_model.get("centroids", {}))
    return jsonify({
        "status": "healthy",
        "engine": f"smart_classifier_v{MODEL_VERSION}",
        "model_version": MODEL_VERSION,
        "supported_languages": [
            {"code": "pag", "name": "Pangasinan"},
            {"code": "ilo", "name": "Ilocano"},
            {"code": "fil", "name": "Tagalog"},
            {"code": "en", "name": "English"},
        ],
        "intents_count": intents_count,
        "trained_intents_count": trained_intents_count,
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
            # Do not expose unverified details.
            safe_services = []
            for svc in kb.get('services', []):
                safe_services.append({
                    "intent": svc.get("intent"),
                    "title": svc.get("title"),
                    "verified": svc.get("verified", False),
                })
            return jsonify(safe_services)
    return jsonify([])

def check_admin_access():
    if os.environ.get("BRGYLINK_ENABLE_ADMIN") != "true":
        abort(403, description="Admin dashboard is disabled by default for security. Development-only.")
    if 'admin_user' not in session:
        abort(401, description="Unauthorized")

def check_csrf():
    token = request.headers.get('X-CSRF-Token')
    if not token or token != session.get('csrf_token'):
        abort(403, description="CSRF token missing or invalid")

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    if os.environ.get("BRGYLINK_ENABLE_ADMIN") != "true":
        abort(403, description="Admin tools are development-only.")
    data = request.get_json() or {}
    expected_password = os.environ.get('ADMIN_PASSWORD')
    if not expected_password:
        abort(403, description="Admin login disabled (ADMIN_PASSWORD not set).")

    if data.get('username') == 'admin' and data.get('password') == expected_password:
        session['admin_user'] = 'admin'
        session['actor'] = 'System Administrator (Authenticated)'
        session['csrf_token'] = os.urandom(24).hex()
        return jsonify({"status": "ok", "csrf_token": session['csrf_token']})

    abort(401, description="Invalid credentials")

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    check_csrf()
    session.clear()
    return jsonify({"status": "ok"})

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if os.environ.get("BRGYLINK_ENABLE_ADMIN") != "true":
        abort(403, description="Admin dashboard is disabled by default for security. Development-only.")
    if request.method == 'POST':
        password = request.form.get('password')
        expected_password = os.environ.get('ADMIN_PASSWORD')
        if expected_password and password == expected_password:
            session['admin_user'] = 'admin'
            session['actor'] = 'System Administrator (Authenticated)'
            session['csrf_token'] = os.urandom(24).hex()
            return redirect('/admin')
        return "Invalid credentials", 401
    return '''
    <form method="POST">
        <input type="password" name="password" placeholder="Admin Password">
        <button type="submit">Login</button>
    </form>
    '''

@app.route('/admin')
def admin():
    if os.environ.get("BRGYLINK_ENABLE_ADMIN") != "true":
        abort(403, description="Admin dashboard is disabled by default for security. Development-only.")
    if 'admin_user' not in session:
        return redirect('/login')
    return send_from_directory(BASE_DIR, 'admin.html')

@app.route('/api/admin/csrf', methods=['GET'])
def get_csrf():
    check_admin_access()
    return jsonify({"csrf_token": session.get('csrf_token')})


@app.route('/api/admin/knowledge_base', methods=['GET'])
def api_get_kb():
    check_admin_access()
    kb_path = os.path.join(BASE_DIR, 'knowledge_base.json')
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb = json.load(f)
            return jsonify(kb)
    return jsonify({})


@app.route('/api/admin/knowledge_base/<intent>', methods=['POST'])
def api_update_kb(intent):
    check_admin_access()
    check_csrf()
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    kb_path = os.path.join(BASE_DIR, 'knowledge_base.json')
    # Apply file lock
    lock_path = os.path.join(BASE_DIR, 'knowledge_base.lock')
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)

            if "base_version" not in data:
                return jsonify({"error": "Missing base_version"}), 400

            current_version = kb.get("_version", 1)
            if current_version != data["base_version"]:
                return jsonify({"error": "Concurrent modification detected. Refresh and try again."}), 409

            kb["_version"] = current_version + 1

            found_svc = None
            # find the intent
            for svc in kb["services"]:
                if svc["intent"] == intent:
                    found_svc = svc
                    break

            if not found_svc:
                return jsonify({"error": "Intent not found"}), 404

            audit_event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "actor": session.get('actor', 'Unknown'),
                "intent": intent,
                "changes": {}
            }

            # update fields safely
            if "title" in data:
                if not isinstance(data["title"], str) or len(data["title"]) > 200:
                    return jsonify({"error": "Invalid title"}), 400
                audit_event["changes"]["title"] = {"old": found_svc.get("title"), "new": data["title"]}
                found_svc["title"] = data["title"]

            if "answer" in data:
                ans = data["answer"]
                if not isinstance(ans, dict):
                    return jsonify({"error": "Invalid answer format"}), 400
                for lang in ["english", "tagalog", "ilocano", "pangasinan"]:
                    if lang not in ans or not isinstance(ans[lang], str) or not ans[lang].strip():
                        return jsonify({"error": f"Missing required answer for {lang}"}), 400
                    if len(ans[lang]) > 2000:
                        return jsonify({"error": "Answer too long"}), 400
                audit_event["changes"]["answer"] = "Answer updated"
                found_svc["answer"] = data["answer"]

            # handle verification
            is_verified = data.get("verified")
            if is_verified is not None:
                if is_verified:
                    source_val = data.get("source")
                    is_valid_url = False
                    is_doc_ref = False

                    if isinstance(source_val, str):
                        source = source_val.strip()
                        from urllib.parse import urlparse
                        parsed = urlparse(source)
                        is_valid_url = parsed.scheme in ["http", "https"] and bool(parsed.netloc)
                    elif isinstance(source_val, dict):
                        source = source_val
                        if "type" in source and "title" in source and "identifier" in source:
                            is_doc_ref = True
                    else:
                        return jsonify({"error": "source must be a URL string or a structured dictionary"}), 400

                    if not is_valid_url and not is_doc_ref:
                        return jsonify({"error": "source must be a valid absolute http/https URL or a structured document reference"}), 400

                    expires_at = data.get("expires_at")
                    if not isinstance(expires_at, str):
                        return jsonify({"error": "expires_at must be a string"}), 400

                    try:
                        dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            return jsonify({"error": "expires_at must be timezone-aware"}), 400
                        if dt <= datetime.now(timezone.utc):
                            return jsonify({"error": "expires_at must be strictly in the future"}), 400
                    except ValueError:
                        return jsonify({"error": "expires_at must be ISO-8601 format"}), 400

                    # Only record who verified if it wasn't already verified
                    if not found_svc.get("verified") or found_svc.get("source") != source or found_svc.get("expires_at") != expires_at:
                        found_svc["verified_by"] = session.get('actor', 'Unknown')
                        found_svc["verified_at"] = datetime.now(timezone.utc).isoformat()

                    found_svc["verified"] = True
                    found_svc["source"] = source
                    found_svc["expires_at"] = expires_at
                    found_svc["last_reviewed_at"] = datetime.now(timezone.utc).isoformat()
                    found_svc["review_notes"] = data.get("review_notes", "")

                    audit_event["changes"]["verification"] = "Marked as verified"

                elif not is_verified:
                    found_svc["verified"] = False
                    audit_event["changes"]["verification"] = "Marked as unverified"

            # Clear smart_classifier cache so it reloads KB next request
            import smart_classifier
            smart_classifier._kb_cache = None

            # Append audit log
            with open(os.path.join(BASE_DIR, "audit.log"), "a", encoding="utf-8") as af:
                af.write(json.dumps(audit_event) + "\n")

            # Backup mechanism
            import shutil
            shutil.copy2(kb_path, kb_path + ".bak")

            # Atomic write
            fd, temp_path = tempfile.mkstemp(dir=BASE_DIR, text=True)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(kb, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, kb_path)

        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return jsonify({"status": "success", "message": "Updated successfully"})


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
