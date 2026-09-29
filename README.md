# BrgyLink AI Chatbot

An offline, development-only barangay-assistant prototype for Barangay Bagong Pag-asa, San Jacinto. It uses a local character n-gram classifier trained from curated intents; it does not call a hosted AI API.

The chatbot deliberately does not state unverified barangay facts as confirmed. Its runtime answers come only from `knowledge_base.json`; unverified barangay facts receive a cautionary notice in the selected language. This project is not an official barangay information service or a production deployment.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Run the full local regression suite with:

```bash
python test_security.py
python test_freshness.py
python test_architecture.py
python test_holdout.py
python test_multilingual_all.py
```

The local tester supports `reset` to clear its language preference and pending document selection.

## Knowledge and verification policy

`knowledge_base.json` is the only runtime answer source. Each entry is one of:

- `system_copy`: non-factual UI text such as greetings and fallback messages; no verification notice is needed.
- `barangay_fact`: an operational or public-service answer. It is treated as unverified unless an authorized reviewer records a real source, review identity, verification date, and expiry date.

The chatbot currently directs users to the BrgyLink Officials section for the roster; it does not yet read that roster dynamically. Do not mark an entry verified merely because it was migrated, generated, or tested.

## Admin dashboard

The admin dashboard is disabled by default and is intended only for local development. Enabling it requires `BRGYLINK_ENABLE_ADMIN=true`, `ADMIN_PASSWORD`, and, in production environments, `FLASK_SECRET_KEY`. It is a single-admin tool, not a production staff-access system: it does not provide staff accounts, roles, rate limiting, or password hashing.

## Included data

`data/intents_brgylink_curated.json` contains the chatbot's curated training examples. Large general-language datasets, local model adapters, and generated model files are intentionally not committed.
