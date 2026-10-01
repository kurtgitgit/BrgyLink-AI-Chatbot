# Hugging Face Space deployment

This service is designed for a separate **private Docker Space** named
`Kurtgitgit/brgylink-ai-chatbot`. It is a backend dependency, not a public
resident chat website.

## Required Space secrets

Add these in the Space **Settings → Variables and secrets** page. Never commit
them, paste them into a chat, or put them in a mobile app.

| Secret | Purpose |
| --- | --- |
| `AI_SERVICE_TOKEN` | Shared token that Mission17 sends in `X-AI-Service-Token` |
| `FLASK_SECRET_KEY` | Random value for Flask's signed local session cookie |

The Dockerfile sets `BRGYLINK_ENV=production`, which refuses to start if either
secret is missing. It also forcibly keeps `BRGYLINK_ENABLE_ADMIN=false`.

## Space settings

- SDK: Docker
- Visibility: Private
- Hardware: CPU Basic is sufficient for this small offline classifier.
- Port: 7860 (defined in the README front matter).

Private Spaces require the Mission17 backend to authenticate with a Hugging
Face read token at the Space gateway. Keep that token only in the backend
environment. The backend then sends the distinct `X-AI-Service-Token` header
to this app. Do not put either token in Expo, an APK, a website bundle, or
client-side JavaScript.

## Deployment checks

1. Confirm the Space build reaches `Running`.
2. With the Hugging Face access token and `X-AI-Service-Token`, request
   `GET /health` and `POST /chat` from the server side.
3. Confirm a missing or incorrect service token receives `401` on `/chat`.
4. Only then add the Mission17 backend proxy and test the existing mobile
   chatbot screen. The app still sends `{message, history}` and expects
   `{reply}`; no APK or OTA is needed for that backend-only change.

The local command `python -B -X utf8 run_checks.py` remains the pre-push gate.
