from smart_classifier import handle_message, load_model, new_session

try:
    model = load_model()
except (FileNotFoundError, ValueError) as error:
    raise SystemExit(f"{error}\nTrain first with: python smart_classifier.py")

session = new_session()
print("BrgyLink offline smart classifier. Type 'reset' to clear preferences or 'quit' to exit.\n")
while True:
    message = input("You: ").strip()
    if message.lower() in {"quit", "exit"}:
        break
    if not message:
        continue
    result = handle_message(message, session, model)
    print(f"Intent: {result['intent']} | Similarity: {result['similarity']:.1%} | Language: {result['language']}")
    print(f"BrgyLink AI: {result['response']}\n")
