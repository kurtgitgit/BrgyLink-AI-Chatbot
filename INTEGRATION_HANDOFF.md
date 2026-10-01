# Mission17 integration handoff

The local remediation gates in `READINESS_REPORT.md` pass. This supports a
controlled backend-mediated integration test once separately approved. It is **not**
ready to be exposed directly from the mobile application or to publish as an
independent public service.

## Stable local service contract

- `GET /health` reports the model and loaded intent counts.
- `POST /chat` accepts `{ "message": string, "session_id"?: string, "language"?: "en"|"fil"|"ilo"|"pag" }`.
- It returns `{ response, intent, confidence, language, language_name, processing_time_ms }`.
- The confidence value is a routing score (keyword rules may exceed 1), not a calibrated probability of correctness.
- Invalid types/languages/IDs return `400`; oversized bodies return `413`;
  processing failures return a generic `500` without internal exception details.
- Anonymous calls are stateless. Explicit sessions are isolated, bounded and
  process-local; multiple workers do not share conversation state.

## Mission17 adapter contract

The mobile app currently sends `{ message, history }` to Mission17's
`POST /api/chatbot` endpoint and expects `{ reply }`. Keep that mobile contract
unchanged. The Mission17 backend should call this service server-to-server,
then return only `{ reply: chatbotResponse.response }` to the app.

- Keep any service URL and service token in the Mission17 backend environment;
  never put it in the APK or a mobile OTA update.
- The adapter must create a stable, non-personal `session_id` from the
  authenticated BrgyLink user. Do not rely on a shared IP address, which can
  mix language or document context between residents.
- Apply the Mission17 backend's existing authentication, request-size limit,
  and rate limit before proxying.
- On service timeout or failure, return the existing safe BrgyLink fallback;
  never expose Python exception details to residents.
- Before hosting externally, protect the Python service with private networking
  or authenticated service-to-service access. It currently has no service-token
  enforcement or production rate limiter. Never treat `session_id` as login auth.
- Audit/pin the minimal runtime dependencies for the selected hosting platform;
  the legacy `requirements.txt` also includes unused experimental ML packages.

## Safety invariants

- Keep `smart_classifier.py` safety routing ahead of normal classification:
  emergency, threats/abuse/self-harm, medical symptoms, legal questions, and
  out-of-scope topics.
- Keep `knowledge_base.json` as the only chatbot answer source. New barangay
  facts require a real source, reviewer, verification date, and expiry date.
- The chatbot must not access resident records, document requests, IDs,
  blotter reports, account data, or admin APIs.
- The Officials answer remains non-dynamic for now. When that work is approved,
  the only eligible source is the active roster behind Mission17's public
  `GET /api/officials` route and `models/Official.js`. Never use a route or
  query that includes archived officials (for example `status=all`).

## Verified local checks

Run from this repository on Windows:

```powershell
python -B -X utf8 run_checks.py
```

These cover current signup guidance, age 15, OTP recovery, civic-task and
event proof navigation, roster safety, emergency routing, factual safety,
multilingual responses, and the development-only admin safeguards.
They also cover two frozen resident query sets, mixed-language safety precedence,
stale/corrupt model recovery, HTTP input limits, repository-file exposure,
session isolation/expiry/capacity and generic error responses. No deployment,
APK/OTA or real-device integration has been performed as part of this work.
