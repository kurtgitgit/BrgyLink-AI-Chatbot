# Local chatbot remediation verdict

**PASS for the local remediation gates; suitable for a separately approved,
controlled backend integration test. NOT a production or flawless-system claim.**

Evaluated on Windows / Python 3.11. Local Flask and Werkzeug were 2.2.2;
these existing development dependencies were not installed or upgraded here.
Audit a current, pinned hosting runtime before deploying.

## Scope and state

- Repository: `C:\Users\Kurt Perez\BrgyLink-AI-Chatbot`, separate from Mission17.
- Starting Git checkpoint: `main`, `d1f3660`; changes remain local/uncommitted.
- Engine v7: 34 trained intents, 1,107 examples, 37 KB services.
- No commit, push, service hosting, Lightsail change, APK build or OTA publication.
- Mission17 checkout at `264f5e8` was inspected read-only. Its pre-existing
  documentation and other worktree changes were not modified by this remediation.

## Requirements and authoritative evidence

| Requirement | Evidence | Result |
| --- | --- | --- |
| Safety wins over language requests | `smart_classifier.py` routes emergency/threat/medical concerns before language-only acknowledgement; 48 mixed-language safety subcases | Pass |
| Common urgent wording is recognized | Frozen query sets exercise cannot breathe, stopped breathing, choking, struggling to breathe, self-harm, plus Tagalog/Ilocano/Pangasinan phrases | Pass for tested phrases |
| OTP recovery differs from signup timing | Missing/expired/incorrect-code questions use the KB `otp` variant; timing questions still explain Step 1 | Pass |
| Password and account-review help is actionable | KB variants, all four languages; Forgot Password button, resend timer, pending/rejected review without claiming personal status | Pass |
| Signup and ID-photo copy matches current app | Step 1 verification, final-step password/ID, minimum 15, front/back or Passport information page; source inspected in Mission17 | Pass, source-verified navigation |
| Civic/event wording is useful and honest | Profile / Community Mission Submissions for civic status, Events tab for live details, no invented approval, schedule or rewards | Pass |
| Officials safety is preserved | Confirmed-roster fallback retained; `Official.js` and public active-only `/api/officials` inspected; no dynamic connection or names inserted | Pass |
| Runtime answers remain KB-controlled | `get_kb_answer` resolves base or validated four-language KB variants; verification requirements/disclaimers remain; zero entries marked verified | Pass |
| Broader diagnostics are reproducible | Frozen 72-query set and 30-query expansion, no normalized exact training duplicates; tests assert routing, selected language and key content | Pass, regression evidence only |
| API avoids preventable exposure/failures | Request types/languages/IDs/size bounds, static repository downloads blocked, generic failures, no IP-shared state; 8 API tests | Pass locally |
| Model can recover from stale/partial data | Training SHA-256 checked on load; stale/corrupt models rebuilt; writes use atomic replacement; isolated lifecycle tests | Pass locally |
| Existing behavior is preserved | All original supported safety, multilingual, architecture/freshness and development-admin suites rerun | Pass |

Emergency templates direct users to the Philippine national emergency number,
without implying that the chatbot can dispatch help. The number was checked
against [PH Emergency Hotlines](https://ehotlines.e.gov.ph/). The expanded breathing
phrases are escalation triggers, not diagnoses or treatment instructions;
the seriousness of breathing difficulty in young children was cross-checked
with [NHS urgent-help guidance](https://www.nhs.uk/baby/health/when-to-get-urgent-medical-help-for-babies-and-children-under-5/).
UK contact numbers were not imported into the Philippine bot. Barangay-specific
verification fields were not fabricated or marked confirmed.

## Executed local gates

Command: `python -B -X utf8 run_checks.py` — exit code 0.

| Suite | Observed result |
| --- | --- |
| Resident readiness | 11 test methods pass, including 102 resident queries, language/safety cross-product, KB variants and model lifecycle |
| Chat HTTP contract | 8 test methods pass |
| Development admin security | 6/6 pass, including concurrent-write conflict behavior |
| Freshness / factual safety | 6/6 pass; scans include answer variants |
| Knowledge architecture | 8/8 pass |
| Current-app integration guidance | 15/15 cases pass |
| Existing holdout/regression suite | 201/201 cases pass |
| Existing multilingual suite | 39/39 cases pass |

Additional executed checks:

- Original resident diagnostic improved from 32/72 expected routes to 72/72.
- Expansion diagnostic: 30/30 expected routes and selected languages.
- Real ephemeral loopback HTTP server: `/health` 200; mixed-language emergency
  chat 200; repository KB download 404; invalid-message payload 400; separate
  anonymous password-recovery request 200. The temporary server was stopped.
- Preparation script run twice: identical training/KB hashes before, after first
  run and after second run. It rejects evaluation-query leakage.
- Model fingerprint matches current training bytes; modified Python modules
  parse successfully; `git diff --check` passes.

### What these numbers do not establish

The query sets were inspected during remediation. They are now regression gates,
not a blind test or statistical real-world accuracy estimate. No exact normalized
training duplicates does not rule out paraphrase similarity. The classifier
contains deliberate keyword rules; its confidence is a routing score, not a
calibrated probability. No guarantee is made about all possible wording,
multilingual fluency, safety-topic recall, medical triage, negation handling or
every device/network. Native-speaker review of the drafted Ilocano/Pangasinan
copy remains pending.

## Gate before external integration

See `INTEGRATION_HANDOFF.md`. This work does not authorize deployment.

1. Choose and authorize hosting; protect the Python service through private
   networking or authenticated server-to-server access. There is no service-token
   enforcement or production request-rate limiter in this prototype.
2. Audit/pin a minimal current runtime. The legacy dependency file includes
   unused experimental ML libraries. Do not run Flask's development server as
   a public production server or enable the development admin dashboard.
3. Implement an authenticated/rate-limited Mission17 backend adapter that keeps
   the app's `{message, history}` -> `{reply}` contract. Derive an opaque session
   ID server-side and use a timeout plus safe fallback; the Python service's
   process-local sessions are not identity authorization or shared worker state.
4. If separately approved, connect only the active public Officials roster;
   exclude archived officials and keep the safe fallback when unavailable.
5. Verify the actual deployed endpoint, service failures/timeouts, and the
   existing APK on real phones and weak connections. Those are not yet tested.

## Reproduce

Run from this separate repository, not inside Mission17:

```powershell
python -B -X utf8 prepare_integration_training.py
python -B -X utf8 smart_classifier.py
python -B -X utf8 run_checks.py
```

Review the diff after preparation if staff have edited KB answers since this
checkpoint. The generated model is ignored by Git and is rebuilt from trusted
curated data; never load an untrusted pickle.
