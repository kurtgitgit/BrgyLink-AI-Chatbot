"""Evaluate BrgyLink's locally trained classifier with held-out test questions."""

import json
import os

from test_classifier import load_classifier, predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_FILE = os.path.join(BASE_DIR, "data", "evaluation_cases.json")


def main():
    with open(EVALUATION_FILE, encoding="utf-8") as file:
        cases = json.load(file)

    model, words, classes, responses = load_classifier()
    passed = 0
    failures = []

    for case in cases:
        result = predict(case["message"], model, words, classes, responses)
        expected = case["expected_intent"]
        actual = result["intent"]
        if actual == expected:
            passed += 1
        else:
            failures.append((case["message"], expected, actual, result["confidence"]))

    total = len(cases)
    print(f"Passed: {passed}/{total} ({passed / total:.1%})")
    print(f"Failed: {len(failures)}")
    for message, expected, actual, confidence in failures:
        print(f"- {message!r}")
        print(f"  expected={expected}; got={actual} ({confidence:.1%})")


if __name__ == "__main__":
    main()
