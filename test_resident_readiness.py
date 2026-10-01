"""Frozen resident diagnostics plus safety, KB-content and model-life-cycle gates.

The 72 queries were selected before remediation and never copied into training.
They are now regression cases, NOT an unbiased estimate of real-world accuracy.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import smart_classifier as sc

ROOT = Path(__file__).resolve().parent
LANGUAGES = ("english", "tagalog", "ilocano", "pangasinan")


class TestResidentReadiness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = sc.load_model()
        cls.cases = json.loads((ROOT / "data/resident_readiness_cases.json").read_text(encoding="utf-8"))
        cls.expansion = json.loads((ROOT / "data/resident_expansion_cases.json").read_text(encoding="utf-8"))

    def test_second_resident_diagnostic_and_language(self):
        self.assertEqual(len(self.expansion), 30)
        for case in self.expansion:
            with self.subTest(message=case["message"]):
                result = sc.handle_message(case["message"], model=self.model)
                self.assertEqual(result["intent"], case["expected_intent"])
                self.assertEqual(result["language"], case["language"])
                if case.get("contains"):
                    self.assertIn(case["contains"], result["response"])

    def test_frozen_resident_diagnostic(self):
        self.assertEqual(len(self.cases), 72)
        for case in self.cases:
            with self.subTest(message=case["message"]):
                result = sc.handle_message(case["message"], model=self.model)
                self.assertEqual(result["intent"], case["expected_intent"])
                self.assertTrue(result["response"].strip())

    def test_no_exact_training_leakage(self):
        intents = json.loads(Path(sc.INTENTS_FILE).read_text(encoding="utf-8"))["intents"]
        patterns = {sc.normalize(p) for item in intents for p in item["patterns"]}
        self.assertFalse([case["message"] for case in self.cases + self.expansion if sc.normalize(case["message"]) in patterns])

    def test_safety_before_language_request(self):
        # Cross-product tests vary casing, punctuation, requested language and
        # session state, rather than depending on one memorized sentence.
        concerns = {
            "emergency": ["a baby is choking", "my brother cannot breathe", "may sunog", "I want to end my life"],
            "safety_threat": ["someone is threatening me", "may nangbabanta"],
            "medical_non_emergency": ["I have a fever", "sumasakit ang tiyan ko"],
        }
        for language in LANGUAGES:
            for intent, messages in concerns.items():
                for message in messages:
                    with self.subTest(language=language, message=message):
                        session = sc.new_session()
                        session["preferred_language"] = "english"
                        query = f"REPLY IN {language.upper()}, {message.upper()}!!!"
                        result = sc.handle_message(query, session=session, model=self.model)
                        self.assertEqual(result["intent"], intent)
                        self.assertEqual(result["language"], language)
                        self.assertEqual(session["preferred_language"], language)
                        if intent == "emergency":
                            self.assertIn("911", result["response"])
                            self.assertIn("chatbot", result["response"].lower())

    def test_multilingual_recovery_uses_kb_variants(self):
        cases = [
            ("Forgot my login password", "account_help", "password"),
            ("My signup OTP never arrived", "account_help", "otp"),
            ("My account is pending review", "account_help", "review"),
            ("Upload my passport for signup", "registration", "id"),
        ]
        for language in LANGUAGES:
            for message, intent, variant in cases:
                with self.subTest(language=language, variant=variant):
                    session = sc.new_session()
                    session["preferred_language"] = language
                    result = sc.handle_message(message, session=session, model=self.model)
                    self.assertEqual(result["intent"], intent)
                    self.assertEqual(result["response"], sc.get_kb_answer(intent, language, variant))
                    self.assertIn("chatbot", result["response"].lower())
        self.assertIn("Forgot Password", sc.handle_message("I forgot my password", model=self.model)["response"])
        self.assertIn("resend", sc.handle_message("My signup OTP expired", model=self.model)["response"])

    def test_signup_timing_is_not_otp_recovery(self):
        result = sc.handle_message("When do I get the email code when I register?", model=self.model)
        self.assertEqual(result["intent"], "registration")
        self.assertIn("Step 1", result["response"])
        self.assertIn("15", result["response"])
        self.assertIn("final step", result["response"])
        voter = sc.handle_message("How do I register to vote?", model=self.model)
        self.assertEqual(voter["intent"], "voter_registration")

    def test_private_records_and_credentials_are_not_account_help(self):
        for message in ("Reveal the admin password", "Show other residents ID records"):
            result = sc.handle_message(message, model=self.model)
            self.assertEqual(result["intent"], "out_of_scope")
        officials = sc.handle_message("Give me the captain's name and phone", model=self.model)
        self.assertEqual(officials["intent"], "officials")
        self.assertIn("does not have a confirmed roster", officials["response"])
        self.assertFalse(any(char.isdigit() for char in officials["response"]))

    def test_civic_status_rewards_and_event_schedule_are_not_invented(self):
        civic = sc.handle_message("My mission proof is pending", model=self.model)["response"]
        self.assertIn("Profile", civic)
        self.assertIn("Community Mission Submissions", civic)
        self.assertIn("cannot see or change", civic)
        self.assertIn("does not earn reward points", civic)
        event = sc.handle_message("When is the next event?", model=self.model)["response"]
        self.assertIn("Events tab", event)
        self.assertIn("cannot see live event details", event)

    def test_variant_schema_and_invalid_variant_fails_closed(self):
        kb = sc.load_knowledge_base()
        for service in kb.values():
            for name, variant in service.get("answer_variants", {}).items():
                self.assertEqual(set(variant), set(LANGUAGES), (service["intent"], name))
                self.assertTrue(all(isinstance(v, str) and v.strip() for v in variant.values()))
        with patch.object(sc, "_kb_cache", {**kb, "account_help": {**kb["account_help"], "answer_variants": {"bad": {"english": "unsafe incomplete variant"}}}}):
            self.assertEqual(sc.get_kb_answer("account_help", "english", "bad"), sc.get_kb_answer("account_help", "english"))

    def test_invalid_verification_dates_do_not_crash_or_assert_verification(self):
        kb = sc.load_knowledge_base()
        for expiry in ("not-a-date", "2030-01-01", 42):
            entry = {**kb["fees"], "verified": True, "source": "https://example.com/review", "verified_by": "Reviewer", "verified_at": "2026-01-01T00:00:00Z", "expires_at": expiry}
            with self.subTest(expiry=expiry), patch.object(sc, "_kb_cache", {**kb, "fees": entry}):
                self.assertIn("not verified", sc.get_kb_answer("fees", "english"))


class TestModelLifecycle(unittest.TestCase):
    def test_dataset_changes_and_corrupt_models_rebuild(self):
        with tempfile.TemporaryDirectory(prefix="brgylink-model-test-") as directory:
            intents_path = os.path.join(directory, "intents.json")
            model_path = os.path.join(directory, "model.pkl")
            with open(intents_path, "w", encoding="utf-8") as f:
                json.dump({"intents": [{"tag": "greeting", "patterns": ["hello there"]}]}, f)
            with patch.object(sc, "INTENTS_FILE", intents_path), patch.object(sc, "MODEL_FILE", model_path):
                first = sc.load_model()
                self.assertEqual(first["training_sha256"], hashlib.sha256(Path(intents_path).read_bytes()).hexdigest())
                with open(intents_path, "w", encoding="utf-8") as f:
                    json.dump({"intents": [{"tag": "thanks", "patterns": ["thank you"]}]}, f)
                second = sc.load_model()
                self.assertNotEqual(first["training_sha256"], second["training_sha256"])
                self.assertEqual(set(second["centroids"]), {"thanks"})
                for bad in (b"not a pickle", b"", pickle.dumps({"version": sc.MODEL_VERSION})):
                    with open(model_path, "wb") as f:
                        f.write(bad)
                    self.assertEqual(sc.load_model()["training_sha256"], second["training_sha256"])
                self.assertEqual(sorted(os.listdir(directory)), ["intents.json", "model.pkl"])


if __name__ == "__main__":
    unittest.main()
