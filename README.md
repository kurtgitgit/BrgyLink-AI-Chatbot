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
python -B -X utf8 run_checks.py
```

The local tester supports `reset` to clear its language preference and pending document selection.

## Knowledge and verification policy

`knowledge_base.json` is the only runtime answer source. Each entry is one of:

- `system_copy`: non-factual UI text such as greetings and fallback messages; no verification notice is needed.
- `barangay_fact`: an operational or public-service answer. It is treated as unverified unless an authorized reviewer records a real source, review identity, verification date, and expiry date.

The chatbot does not yet read the Officials roster dynamically and must not name current officials or invent contact details. Its safe fallback directs users to the barangay office or Announcements. The source for a future approved connection is Mission17's public `/api/officials` active roster (`Official.js`, excluding archived records). Do not mark an entry verified merely because it was migrated, generated, or tested.

Password, OTP and registration-ID help may select a four-language `answer_variants` template from the same knowledge-base entry. The entry's original verification requirements and disclaimer still apply. The classifier never uses the training dataset's response fields as runtime answers.

## Current local readiness

See `READINESS_REPORT.md` for evidence and limits, and `INTEGRATION_HANDOFF.md` for the future backend adapter. The current engine is v7, with 34 trained intents and 1,107 training examples. A training-data fingerprint invalidates stale local models; interrupted writes do not replace a working model with a partial file.

Emergency and threat routing takes priority over language-switch requests. Missing/expired/invalid signup codes select recovery guidance, not the generic signup flow. This remains a curated FAQ classifier, not a general conversational LLM.

The 72-query diagnostic and 30-query expansion set are regression gates. They have no normalized exact duplicates in training, but they were consulted during remediation. Passing them is not an unbiased accuracy estimate and does not certify language fluency or emergency coverage.

## Local HTTP contract

`POST /chat` accepts a JSON object containing `message`, optional `session_id`, and optional `language`. Messages are limited to 2,000 characters and bodies to 16 KiB. A session ID is 1–128 letters, digits, dots, underscores or hyphens; do not use an email or other personal identifier. Use language codes `en`, `fil`, `ilo`, `pag` (full names and `tl` are also accepted).

Requests without a session ID are stateless; shared IP addresses never share chat context. Explicit sessions are process-local, expire after one hour of inactivity, and have a capacity of 1,000. They are not identity authentication. Repository/model files are not public static downloads. The service still needs a protected hosting boundary and an authenticated, rate-limited Mission17 adapter before external use.

To reproduce the reviewed data and model locally:

```powershell
python -B -X utf8 prepare_integration_training.py
python -B -X utf8 smart_classifier.py
python -B -X utf8 run_checks.py
```

The preparation script refuses exact evaluation/training overlap and does not create verification records. It does update curated app guidance; review its diff before using it after any staff-authored KB changes. Only load locally generated, trusted model files.

## Admin dashboard

The admin dashboard is disabled by default and is intended only for local development. Enabling it requires `BRGYLINK_ENABLE_ADMIN=true`, `ADMIN_PASSWORD`, and, in production environments, `FLASK_SECRET_KEY`. It is a single-admin tool, not a production staff-access system: it does not provide staff accounts, roles, rate limiting, or password hashing.

## Included data

`data/intents_brgylink_curated.json` contains the chatbot's curated training examples. Large general-language datasets, local model adapters, and generated model files are intentionally not committed.
