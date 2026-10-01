"""
Architecture Test Suite for BrgyLink AI.

Enforces:
- Strict v6 schema for every KB service
- KB-only answer resolution
- content_type / requires_verification semantics
- No fabricated verification (system/migration actors, .local sources)
- system_copy entries never receive the factual disclaimer
"""
import unittest
import json
from smart_classifier import get_kb_answer, handle_message, load_model, new_session

FAKE_REVIEWERS = {"Migration Script", "System", "system", "migration script"}
SYSTEM_COPY_INTENTS = {"greeting", "goodbye", "thanks", "language_support",
                       "out_of_scope", "fallback", "about_app", "events"}


class TestArchitecture(unittest.TestCase):
    def setUp(self):
        self.model = load_model()
        self.session = new_session()
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            self.kb = json.load(f)

    # ------------------------------------------------------------------
    # 1. Every KB service resolves from the KB
    # ------------------------------------------------------------------
    def test_kb_only_resolution(self):
        for kb_svc in self.kb["services"]:
            intent = kb_svc.get("intent") or kb_svc.get("id")
            ans = get_kb_answer(intent, "english")
            expected = kb_svc["answer"].get("english", "")
            self.assertIn(expected, ans)

    # ------------------------------------------------------------------
    # 2. Unknown intents fall back to out_of_scope
    # ------------------------------------------------------------------
    def test_unknown_intent_fallback(self):
        ans = get_kb_answer("some_random_nonexistent_intent", "english")
        oos_svc = next(s for s in self.kb["services"]
                       if s.get("intent") == "out_of_scope")
        expected = oos_svc["answer"]["english"]
        self.assertIn(expected, ans)

    # ------------------------------------------------------------------
    # 3. Strict v6 schema for every entry
    # ------------------------------------------------------------------
    def test_v6_schema_strict(self):
        self.assertEqual(
            self.kb.get("metadata", {}).get("status"), "prototype_draft")

        for svc in self.kb["services"]:
            intent = svc.get("intent", "?")

            # No legacy answer_* fields
            for legacy in ["answer_en", "answer_fil", "answer_ilo", "answer_pag"]:
                self.assertNotIn(legacy, svc,
                                 f"Legacy field {legacy} in {intent}")

            # answer must be a 4-language dict of non-empty strings
            ans = svc.get("answer")
            self.assertIsInstance(ans, dict, f"answer not dict in {intent}")
            self.assertEqual(
                set(ans.keys()),
                {"english", "tagalog", "ilocano", "pangasinan"},
                f"Missing lang keys in {intent}")
            for lang, text in ans.items():
                self.assertIsInstance(text, str,
                                     f"{lang} not str in {intent}")
                self.assertTrue(text.strip(),
                                f"{lang} empty in {intent}")

            # content_type and requires_verification must be present
            self.assertIn("content_type", svc,
                          f"Missing content_type in {intent}")
            self.assertIn(svc["content_type"],
                          ("system_copy", "barangay_fact"),
                          f"Invalid content_type in {intent}")
            self.assertIn("requires_verification", svc,
                          f"Missing requires_verification in {intent}")

            # verified entries must have all four metadata fields
            if svc.get("verified") is True:
                for f in ["verified_by", "verified_at", "expires_at", "source"]:
                    self.assertIn(f, svc,
                                  f"Missing {f} in verified {intent}")

    # ------------------------------------------------------------------
    # 4. Core categories present in KB
    # ------------------------------------------------------------------
    def test_core_categories_pull_from_kb(self):
        for cat in ["clearance", "emergency", "greeting", "goodbye"]:
            found = any(s.get("intent") == cat for s in self.kb["services"])
            self.assertTrue(found, f"Core category {cat} missing from KB")

    # ------------------------------------------------------------------
    # 5. Multilingual runtime responses
    # ------------------------------------------------------------------
    def test_multilingual_responses_actual_text(self):
        queries = [
            ("Magkano ang bayad sa clearance?", "tagalog", "fees"),
            ("Ania ti oras ti opisina?", "ilocano", "office_hours"),
            ("Mano so bayad na clearance?", "pangasinan", "fees"),
        ]
        for query, expected_lang, expected_intent in queries:
            result = handle_message(query)
            self.assertEqual(result["language"], expected_lang,
                             f"Language mismatch for: {query}")
            self.assertEqual(result["intent"], expected_intent,
                             f"Intent mismatch for: {query}")
            kb_svc = next(s for s in self.kb["services"]
                          if s.get("intent") == expected_intent)
            expected_answer = kb_svc["answer"][expected_lang]
            self.assertTrue(
                result["response"].startswith(expected_answer),
                f"Response mismatch for {query}")

    # ------------------------------------------------------------------
    # 6. No fabricated verification
    # ------------------------------------------------------------------
    def test_no_fake_verification(self):
        """No barangay_fact may be verified by a migration/system actor
        or with a .local source."""
        for svc in self.kb["services"]:
            intent = svc.get("intent", "?")
            if svc.get("content_type") != "barangay_fact":
                continue
            if not svc.get("verified"):
                continue

            vb = svc.get("verified_by") or ""
            src = svc.get("source") or ""

            self.assertNotIn(
                vb, FAKE_REVIEWERS,
                f"{intent} verified by fake reviewer '{vb}'")
            self.assertFalse(
                src.rstrip("/").endswith(".local")
                or ".local/" in src,
                f"{intent} uses .local source '{src}'")
            # Must have all four verification fields filled
            for f in ["verified_by", "verified_at", "expires_at", "source"]:
                val = svc.get(f)
                self.assertTrue(
                    val is not None and str(val).strip(),
                    f"{intent} verified but {f} is empty/null")

    def test_system_copy_not_verified_by_fake(self):
        """system_copy entries must not carry fake verification either."""
        for svc in self.kb["services"]:
            if svc.get("content_type") != "system_copy":
                continue
            if svc.get("verified"):
                vb = svc.get("verified_by") or ""
                src = svc.get("source") or ""
                self.assertNotIn(vb, FAKE_REVIEWERS,
                                 f"{svc['intent']} verified by fake '{vb}'")
                self.assertFalse(
                    src.rstrip("/").endswith(".local") or ".local/" in src,
                    f"{svc['intent']} uses .local source '{src}'")

    # ------------------------------------------------------------------
    # 7. system_copy never gets the factual disclaimer
    # ------------------------------------------------------------------
    def test_system_copy_no_disclaimer(self):
        """Greeting, goodbye, thanks, etc. must never have the
        '(Note: This information is not verified…)' disclaimer."""
        disclaimer_fragment = "not verified"
        for intent in SYSTEM_COPY_INTENTS:
            for lang in ["english", "tagalog", "ilocano", "pangasinan"]:
                resp = get_kb_answer(intent, lang)
                self.assertNotIn(
                    disclaimer_fragment, resp.lower(),
                    f"system_copy '{intent}' [{lang}] got factual disclaimer")


if __name__ == "__main__":
    unittest.main()
