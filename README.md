# BrgyLink AI Chatbot

An offline barangay-assistant prototype for Barangay Bagong Pag-asa, San Jacinto. It uses a local character n-gram classifier trained from curated intents; it does not call a hosted AI API.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python smart_classifier.py
python test_smart_classifier.py
```

The local tester supports `reset` to clear its language preference and pending document selection. Service details in `knowledge_base.json` remain staff-review drafts.

## Included data

`data/intents_brgylink_curated.json` contains the chatbot's curated training examples. Large general-language datasets, local model adapters, and generated model files are intentionally not committed.
