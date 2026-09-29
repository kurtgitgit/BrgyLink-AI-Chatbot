# BrgyLink AI data review

`knowledge_base.json` is the chatbot's only runtime answer source for Barangay Bagong Pag-asa, San Jacinto. It remains deliberately cautious: `barangay_fact` entries require review by an authorized barangay representative before they may be presented as confirmed.

## Before publishing

1. Review the barangay name, address, office hours, contacts, fees, requirements, processing times, garbage schedules, and officials roster against an authorized source.
2. For each approved `barangay_fact`, use the admin workflow to record a real source, reviewer identity, verification timestamp, and expiry date before setting `verified` to `true`.
3. Do not mark data verified because it was imported, generated, or tested. Never use placeholder or `.local` sources.
4. Add approved translations for Pangasinan and Ilocano; do not use unreviewed machine translations for barangay procedures.
5. Expand each intent in `data/intents_brgylink_curated.json` with real resident phrasing. Aim for at least 30 examples per intent.
6. Keep sensitive reports out of chat. Send blotter complaints to the proper report flow instead.

## Content types

- `system_copy` is non-factual interface text and does not require fact verification.
- `barangay_fact` is operational information. If it is unverified or expired, the chatbot must avoid exact schedules, fees, requirements, contacts, and eligibility claims, and must direct residents to the barangay office or approved announcements.

## Officials roster

The app's Officials section can be used as a review source, but the chatbot does not yet read it dynamically. Until a shared verified data source is connected, the chatbot must direct users to that section instead of listing names.

## Training data policy

Only train the barangay FAQ model with `data/intents_brgylink_curated.json`. The general Alpaca language datasets can help a separate experimental LLM, but they are not authoritative barangay policy and must not be mixed into the FAQ classifier.
