"""
BrgyLink AI - Multilingual Intelligent Virtual Assistant Server
Barangay Bagong Pag-asa, San Jacinto, Pangasinan.
Supports: Pangasinan (Kabaleyan), Ilocano (Ti Samtoy), Tagalog (Filipino), and English.
"""

import json
import hmac
import os
import re
import time
import tempfile
import threading
from contextlib import contextmanager
from collections import OrderedDict
from datetime import datetime, timezone
from flask import Flask, jsonify, request, send_from_directory, abort, session, redirect
from smart_classifier import (
    MODEL_VERSION,
    detect_language,
    handle_message,
    load_model,
    new_session,
)

try:
    import fcntl  # POSIX (the production Linux deployment)
except ImportError:  # Windows development and local regression tests
    fcntl = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
IS_PRODUCTION = os.environ.get("BRGYLINK_ENV") == "production"
if IS_PRODUCTION:
    app.config['SESSION_COOKIE_SECURE'] = True
    if not os.environ.get("FLASK_SECRET_KEY"):
        raise ValueError("FLASK_SECRET_KEY is required in production")
    if os.environ.get("BRGYLINK_ENABLE_ADMIN", "").lower() == "true":
        raise ValueError("The development-only admin dashboard cannot be enabled in production")
    if not os.environ.get("AI_SERVICE_TOKEN"):
        raise ValueError("AI_SERVICE_TOKEN is required in production")


# Load Smart Classifier model at startup
print(f"Loading BrgyLink Smart Classifier v{MODEL_VERSION}...")
classifier_model = load_model()
print("🎉 BrgyLink AI Multilingual Engine loaded and ready for chat!")

# User session cache
sessions: OrderedDict[str, dict] = OrderedDict()
session_last_seen: dict[str, float] = {}
CHAT_SESSION_LOCK = threading.Lock()
CHAT_SESSION_TTL_SECONDS = 3600
MAX_CHAT_SESSIONS = 1000
MAX_MESSAGE_LENGTH = 2000
WINDOWS_KB_WRITE_LOCK = threading.Lock()


@contextmanager
def knowledge_base_write_lock(lock_path: str):
    """Serialize development-only KB admin writes on POSIX and Windows.

    Production runs on Linux and uses an OS file lock. Windows is explicitly
    development-only, so a process lock gives deterministic local behavior
    without changing the production locking path.
    """
    if fcntl is None:
        with WINDOWS_KB_WRITE_LOCK:
            yield
        return

    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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
LANG_ALIASES = {
    "en": "english", "english": "english", "fil": "tagalog", "tl": "tagalog", "tagalog": "tagalog",
    "ilo": "ilocano", "ilocano": "ilocano", "pag": "pangasinan", "pangasinan": "pangasinan",
}


def service_token_required() -> bool:
    """Keep local development simple while making production server-to-server only."""
    configured = os.environ.get("BRGYLINK_REQUIRE_SERVICE_TOKEN", "").lower() == "true"
    return IS_PRODUCTION or configured


def has_valid_service_token() -> bool:
    """Accept a dedicated header so a private HF gateway can use Authorization."""
    if not service_token_required():
        return True
    expected = os.environ.get("AI_SERVICE_TOKEN", "")
    candidate = request.headers.get("X-AI-Service-Token", "")
    if not candidate:
        authorization = request.headers.get("Authorization", "")
        prefix = "Bearer "
        if authorization.startswith(prefix):
            candidate = authorization[len(prefix):]
    return bool(expected and candidate and hmac.compare_digest(expected, candidate))


def prune_chat_sessions(now: float) -> None:
    """Called under CHAT_SESSION_LOCK; keep only bounded, recently used state."""
    expired = [key for key, seen in session_last_seen.items() if now - seen >= CHAT_SESSION_TTL_SECONDS]
    for key in expired:
        sessions.pop(key, None)
        session_last_seen.pop(key, None)
    while len(sessions) >= MAX_CHAT_SESSIONS:
        key, _ = sessions.popitem(last=False)
        session_last_seen.pop(key, None)


@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({'error': 'Request body is too large.'}), 413


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
    lock_path = os.path.join(BASE_DIR, 'knowledge_base.lock')
    with knowledge_base_write_lock(lock_path):
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


    return jsonify({"status": "success", "message": "Updated successfully"})


@app.route('/chat', methods=['POST'])
def chat():
    t0 = time.time()
    try:
        if not has_valid_service_token():
            return jsonify({'error': 'Unauthorized chatbot service request.'}), 401
        if request.content_length is not None and request.content_length > app.config['MAX_CONTENT_LENGTH']:
            return payload_too_large(None)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({'error': 'A JSON object is required.'}), 400
        user_message = data.get('message')
        if not isinstance(user_message, str) or not user_message.strip():
            return jsonify({'error': 'Please enter a question or message.'}), 400
        user_message = user_message.strip()
        if len(user_message) > MAX_MESSAGE_LENGTH:
            return jsonify({'error': f'Messages must be at most {MAX_MESSAGE_LENGTH} characters.'}), 400
        session_id = data.get('session_id')
        if session_id is not None and (not isinstance(session_id, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', session_id)):
            return jsonify({'error': 'Use a non-personal session_id of 1 to 128 letters, digits, dots, underscores or hyphens.'}), 400
        forced_lang = data.get('language')
        if forced_lang is not None:
            if not isinstance(forced_lang, str) or forced_lang.lower().strip() not in LANG_ALIASES:
                return jsonify({'error': 'Unsupported language.'}), 400
            forced_lang = LANG_ALIASES[forced_lang.lower().strip()]

        # No explicit ID means no persisted state. A shared proxy/Wi-Fi IP must
        # never mix one resident's language or pending document with another's.
        # The future authenticated backend adapter owns trusted ID generation.
        with CHAT_SESSION_LOCK:
            now = time.monotonic()
            if session_id and session_id not in sessions:
                prune_chat_sessions(now)
                sessions[session_id] = new_session()
            elif session_id:
                expired = now - session_last_seen.get(session_id, 0) >= CHAT_SESSION_TTL_SECONDS
                if expired:
                    sessions[session_id] = new_session()
            chat_session = sessions[session_id] if session_id else new_session()
            if session_id:
                sessions.move_to_end(session_id)
                session_last_seen[session_id] = now
            if forced_lang:
                chat_session['preferred_language'] = forced_lang
            result = handle_message(user_message, session=chat_session, model=classifier_model)
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
        # The Flask body-size exception must retain its 413 status.
        from werkzeug.exceptions import RequestEntityTooLarge
        if isinstance(e, RequestEntityTooLarge):
            return payload_too_large(e)
        app.logger.exception('Chat processing failed')
        return jsonify({'error': 'Failed to process chat message. Please try again.'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting BrgyLink AI Multilingual Server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
