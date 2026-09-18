# BrgyLink AI data review

`knowledge_base.json` is the chatbot source of truth for Barangay Bagong Pag-asa, San Jacinto. Its BrgyLink navigation flows come from Mission17's `USER_MANUAL.md` and backend request models. It remains deliberately cautious: entries with `needs_staff_verification: true` must be reviewed by an authorized barangay representative before release.

## Before publishing

1. Replace the barangay name, address, office hours, contacts, fees, requirements, processing times, and garbage schedules.
2. Have a staff member mark each reviewed service `false` and update `last_updated`.
3. Add approved translations for Pangasinan and Ilocano; do not use unreviewed machine translations for official procedures.
4. Expand each intent in `data/intents_brgylink_curated.json` with real resident phrasing. Aim for at least 30 examples per intent.
5. Keep sensitive reports out of chat. Send blotter complaints to the official report flow instead.

## Training data policy

Only train the barangay FAQ model with `data/intents_brgylink_curated.json`. The general Alpaca language datasets can help a separate experimental LLM, but they are not authoritative barangay policy and must not be mixed into the FAQ classifier.
