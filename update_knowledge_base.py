"""
Upgrade knowledge_base.json to be fully multilingual:
English, Tagalog, Ilocano, and Pangasinan.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_FILE = os.path.join(BASE_DIR, "knowledge_base.json")
INTENTS_FILE = os.path.join(BASE_DIR, "data", "intents_brgylink_curated.json")


def update_kb():
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        intents_data = json.load(f)["intents"]

    intent_map = {i["tag"]: i["multilingual_responses"] for i in intents_data if "multilingual_responses" in i}

    with open(KB_FILE, "r", encoding="utf-8") as f:
        kb = json.load(f)

    kb["metadata"]["languages"] = ["English", "Tagalog", "Ilocano", "Pangasinan"]
    kb["metadata"]["municipality"] = "San Jacinto, Pangasinan"
    kb["metadata"]["status"] = "multilingual_verified_production"

    for service in kb.get("services", []):
        tag = service.get("intent")
        if tag in intent_map:
            resp = intent_map[tag]
            service["answer_en"] = resp["english"]
            service["answer_fil"] = resp["tagalog"]
            service["answer_ilo"] = resp["ilocano"]
            service["answer_pag"] = resp["pangasinan"]

    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print("Updated knowledge_base.json with full 4-language support!")


if __name__ == "__main__":
    update_kb()
