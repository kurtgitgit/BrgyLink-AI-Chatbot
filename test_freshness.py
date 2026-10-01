"""
Freshness & Factual-Safety Test Suite for BrgyLink AI.

Ensures that:
1. No forbidden marketing phrases appear anywhere in data sources.
2. Unverified KB answers contain NO concrete real-world claims
   (prices, hours, schedules, contacts, eligibility assertions).
3. Runtime queries for unverified intents do not leak factual claims.
"""
import os
import json
import re
import unittest

from smart_classifier import handle_message, load_model, new_session

FORBIDDEN_PHRASES = [
    "official virtual assistant",
    "fluent in",
    "fluent",
    "verified roster",
    "current roster",
    "updated information",
    "specific office hours",
    "specific unverified fees",
]

# Patterns that must NOT appear in any unverified answer.
FACTUAL_CLAIM_PATTERNS = [
    (re.compile(r"₱\d", re.IGNORECASE), "peso amount"),
    (re.compile(r"\bP\d+[\s\-–—]*P?\d*\b"), "peso range"),
    (re.compile(r"\d{1,2}:\d{2}\s*(AM|PM|am|pm|a\.m\.|p\.m\.)", re.IGNORECASE), "specific time"),
    (re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b", re.IGNORECASE), "weekday schedule"),
    (re.compile(r"\b24/7\b|\b24-7\b|\btwenty.?four.?seven\b", re.IGNORECASE), "24/7 availability"),
    (re.compile(r"\bfree of charge\b|\blibre.*\b(konsulta|checkup|bakuna|gamot|vitamins?|medicines?)\b", re.IGNORECASE), "free-service claim"),
    (re.compile(r"\bfree (for|basic|consult|immun|vaccin|maintenance|prenatal)\b", re.IGNORECASE), "free-service claim"),
    (re.compile(r"\bCertificates are free\b", re.IGNORECASE), "free-certificate claim"),
    (re.compile(r"\bIndigency is free\b", re.IGNORECASE), "free-indigency claim"),
    (re.compile(r"\bRA\s*\d{4,}", re.IGNORECASE), "law/RA as factual source"),
    (re.compile(r"\bNo Segregation.*No Collection\b", re.IGNORECASE), "policy claim"),
    (re.compile(r"(?:Tanod|tanod).*\b24\b", re.IGNORECASE), "tanod 24hr claim"),
    (re.compile(r"\bopen\s+Monday\b", re.IGNORECASE), "schedule claim"),
    (re.compile(r"\b\d{3,4}[-.]?\d{3,4}\b", re.IGNORECASE), "phone/contact number"),
    (re.compile(r"\b\d+\s*(working\s*)?days?\b", re.IGNORECASE), "processing time claim"),
    (re.compile(r"\b(2x2|1x1)\s*photo\b", re.IGNORECASE), "specific requirement claim"),
    (re.compile(r"\b(DTI|SEC)\s*(registration|certificate)\b", re.IGNORECASE), "specific requirement claim"),
    (re.compile(r"\bproof of address\b", re.IGNORECASE), "specific requirement claim"),
]


class TestFreshnessClaims(unittest.TestCase):

    def check_text(self, text, source_info):
        if not isinstance(text, str):
            return
        text_lower = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text_lower:
                if phrase == "fluent":
                    if re.search(r"\bfluent\b", text_lower):
                        self.fail(f"Forbidden phrase '{phrase}' found in {source_info}: {text}")
                else:
                    self.fail(f"Forbidden phrase '{phrase}' found in {source_info}: {text}")

    def traverse_dict(self, d, source_info):
        if isinstance(d, dict):
            for k, v in d.items():
                self.traverse_dict(v, f"{source_info} -> {k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                self.traverse_dict(v, f"{source_info}[{i}]")
        elif isinstance(d, str):
            self.check_text(d, source_info)

    # ------------------------------------------------------------------
    # Legacy checks
    # ------------------------------------------------------------------
    def test_curated_data(self):
        path = "data/intents_brgylink_curated.json"
        if not os.path.exists(path):
            self.skipTest("File not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.traverse_dict(data, path)

    def test_runtime_knowledge_base(self):
        path = "knowledge_base.json"
        if not os.path.exists(path):
            self.skipTest("File not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.traverse_dict(data.get("metadata", {}), f"{path} metadata")

        for intent_data in data.get("services", []):
            intent_name = intent_data.get("intent", "unknown")
            for key in ["answer", "answer_variants", "answer_en", "answer_fil", "answer_ilo", "answer_pag"]:
                if key in intent_data:
                    self.traverse_dict(intent_data[key], f"{path} {intent_name} {key}")

    def test_build_full_knowledge_source(self):
        path = "build_full_knowledge.py"
        if not os.path.exists(path):
            self.skipTest("File not found")
        with open(path, "r", encoding="utf-8") as f:
            source_text = f.read()
        self.check_text(source_text, path)

    # ------------------------------------------------------------------
    # NEW: Unverified-answer factual-claim scanner
    # ------------------------------------------------------------------
    def test_unverified_answers_contain_no_factual_claims(self):
        """Every unverified KB answer in every language must be free of
        concrete real-world operational facts."""
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            kb = json.load(f)

        violations = []
        for svc in kb["services"]:
            if svc.get("verified", False):
                continue
            intent = svc.get("intent", "?")
            answer_sets = [("default", svc.get("answer", {}))]
            answer_sets.extend(svc.get("answer_variants", {}).items())
            for variant, answer in answer_sets:
                if not isinstance(answer, dict):
                    continue
                for lang, text in answer.items():
                    if not isinstance(text, str):
                        continue
                    for pattern, label in FACTUAL_CLAIM_PATTERNS:
                        m = pattern.search(text)
                        if m:
                            violations.append(
                                f"  {intent}/{variant} [{lang}]: {label} -> \"{m.group()}\""
                            )

        if violations:
            self.fail(
                f"Factual claims found in {len(violations)} unverified answer(s):\n"
                + "\n".join(violations)
            )

    # ------------------------------------------------------------------
    # NEW: Runtime regression — unverified fee/hour queries
    # ------------------------------------------------------------------
    def test_unverified_fee_query_no_peso(self):
        """An unverified fee query must NOT return a peso amount."""
        model = load_model()
        queries = [
            "Magkano ang bayad sa clearance?",
            "How much is a barangay clearance?",
            "Mano so bayad na clearance?",
            "Mano ti bayad ti clearance?",
        ]
        for q in queries:
            result = handle_message(q, new_session(), model)
            resp = result["response"]
            self.assertIsNone(
                re.search(r"₱\d", resp),
                f"Peso amount leaked in response to '{q}': {resp[:120]}",
            )
            self.assertIsNone(
                re.search(r"\bfree for\b", resp, re.IGNORECASE),
                f"'free for' leaked in response to '{q}': {resp[:120]}",
            )

    def test_unverified_office_hours_query_no_time(self):
        """An unverified office-hours query must NOT return specific hours."""
        model = load_model()
        queries = [
            "Ano ang oras ng opisina?",
            "Ania ti oras ti opisina?",
            "Antoy oras na opisina?",
            "What are the office hours?",
        ]
        for q in queries:
            result = handle_message(q, new_session(), model)
            resp = result["response"]
            self.assertIsNone(
                re.search(r"\d{1,2}:\d{2}\s*(AM|PM)", resp, re.IGNORECASE),
                f"Specific time leaked in response to '{q}': {resp[:120]}",
            )
            self.assertIsNone(
                re.search(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday)\b", resp, re.IGNORECASE),
                f"Weekday schedule leaked in response to '{q}': {resp[:120]}",
            )
            self.assertIsNone(
                re.search(r"\b24/7\b", resp),
                f"24/7 claim leaked in response to '{q}': {resp[:120]}",
            )


if __name__ == "__main__":
    unittest.main()
