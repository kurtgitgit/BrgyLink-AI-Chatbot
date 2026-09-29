"""Regression checks for local conversation behavior."""
from smart_classifier import handle_message, new_session, train

model = train()
tests = [
    (["Tagalog ang isagot mo", "paano mag file ng blotter?"], "report_incident", "tagalog"),
    (["paano mag file ng document?", "indigency"], "indigency", "english"),
    (["Status ng clearance ko?"], "document_status", "tagalog"),
    (["marunong ka mag ilocano?"], "language_support", "ilocano"),
    (["Tell me a joke"], "out_of_scope", "english"),
]
passed = 0
for messages, expected_intent, expected_language in tests:
    session = new_session()
    for message in messages:
        result = handle_message(message, session, model)
    ok = result["intent"] == expected_intent and result["language"] == expected_language
    passed += ok
    if not ok:
        print(f"FAIL {messages}: got {result['intent']} / {result['language']}")
print(f"Conversation checks: {passed}/{len(tests)}")
