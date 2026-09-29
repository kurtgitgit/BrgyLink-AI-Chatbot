"""
Multilingual Validation Suite for BrgyLink AI Chatbot (Prototype).
Tests Tagalog, Ilocano, and Pangasinan intent classification,
language detection, and localized responses across barangay services.

NOTE: This test suite validates the prototype classifier. It does not
claim 100% accuracy. Results are reported honestly with per-language
breakdowns.
"""

from smart_classifier import handle_message, load_model, new_session

model = load_model()

TEST_CASES = [
    # --- PANGASINAN TESTS ---
    {
        "query": "Maabig ya kabuasan kabaleyan!",
        "expected_intent": "greeting",
        "expected_lang": "pangasinan",
        "check_response": "BrgyLink"
    },
    {
        "query": "Panon so mangala na barangay clearance para ed trabaho?",
        "expected_intent": "clearance",
        "expected_lang": "pangasinan",
        "check_response": "BrgyLink"
    },
    {
        "query": "Antoy kailangan para ed Certificate of Indigency?",
        "expected_intent": "indigency",
        "expected_lang": "pangasinan",
        "check_response": "Certificate of Indigency"
    },
    {
        "query": "Antoy oras na opisina na barangay Bagong Pag-asa?",
        "expected_intent": "office_hours",
        "expected_lang": "pangasinan",
        "check_response": "opisina"
    },
    {
        "query": "Mano so bayad na clearance tan permit?",
        "expected_intent": "fees",
        "expected_lang": "pangasinan",
        "check_response": "bayar"
    },
    {
        "query": "Kapigan so panangala na basura ed purok tayo?",
        "expected_intent": "garbage",
        "expected_lang": "pangasinan",
        "check_response": "basura"
    },
    {
        "query": "Wala so alitan tan kolkolan na sankaabay iner so paka-ayos to?",
        "expected_intent": "lupon",
        "expected_lang": "pangasinan",
        "check_response": "Lupon"
    },
    {
        "query": "Panon so makaawat na ayuda odino AICS tulong-salapi?",
        "expected_intent": "aics_assistance",
        "expected_lang": "pangasinan",
        "check_response": "AICS"
    },
    {
        "query": "Tulongan yo ak apo wala so apoy apoy sunog!",
        "expected_intent": "emergency",
        "expected_lang": "pangasinan",
        "check_response": "911"
    },
    {
        "query": "Antoy kailangan para ed Solo Parent ID?",
        "expected_intent": "solo_parent",
        "expected_lang": "pangasinan",
        "check_response": "Solo Parent"
    },
    {
        "query": "Antoy serbisyo na Barangay Health Center BHC para ed ugugaw?",
        "expected_intent": "health_services",
        "expected_lang": "pangasinan",
        "check_response": "health"
    },
    {
        "query": "Balbaleg ya salamat ed tulong mo kabaleyan!",
        "expected_intent": "thanks",
        "expected_lang": "pangasinan",
        "check_response": "kabaleyan"
    },
    {
        "query": "Makaalis ak la sige paalam!",
        "expected_intent": "goodbye",
        "expected_lang": "pangasinan",
        "check_response": "salamat"
    },

    # --- ILOCANO TESTS ---
    {
        "query": "Naimbag a bigat kadakayo amin!",
        "expected_intent": "greeting",
        "expected_lang": "ilocano",
        "check_response": "BrgyLink"
    },
    {
        "query": "Kasano ti agala ti barangay clearance para iti trabaho?",
        "expected_intent": "clearance",
        "expected_lang": "ilocano",
        "check_response": "BrgyLink"
    },
    {
        "query": "Ania dagiti kasapulan para iti Certificate of Indigency?",
        "expected_intent": "indigency",
        "expected_lang": "ilocano",
        "check_response": "Certificate of Indigency"
    },
    {
        "query": "Ania ti oras ti opisina ti barangay hall?",
        "expected_intent": "office_hours",
        "expected_lang": "ilocano",
        "check_response": "opisina"
    },
    {
        "query": "Mano ti bayad ti sertipiko ken clearance?",
        "expected_intent": "fees",
        "expected_lang": "ilocano",
        "check_response": "bayad"
    },
    {
        "query": "Kaano ti panag-ala ti basura iti purok?",
        "expected_intent": "garbage",
        "expected_lang": "ilocano",
        "check_response": "basura"
    },
    {
        "query": "Adda kolkol ken riri ti kaarruba kasano a maasikaso iti Lupon?",
        "expected_intent": "lupon",
        "expected_lang": "ilocano",
        "check_response": "Lupon"
    },
    {
        "query": "Kasano ti makaawat iti ayuda wenno AICS tulong-pinansyal?",
        "expected_intent": "aics_assistance",
        "expected_lang": "ilocano",
        "check_response": "AICS"
    },
    {
        "query": "Tulongannak adda uram uram uram!",
        "expected_intent": "emergency",
        "expected_lang": "ilocano",
        "check_response": "911"
    },
    {
        "query": "Ania dagiti serbisyo ti Barangay Health Center para kadagiti ubbing?",
        "expected_intent": "health_services",
        "expected_lang": "ilocano",
        "check_response": "health"
    },
    {
        "query": "Agyamanak unay kabsat iti tulongmo!",
        "expected_intent": "thanks",
        "expected_lang": "ilocano",
        "check_response": "kabsat"
    },
    {
        "query": "Agpakada akon kasta pay ken agyamanak!",
        "expected_intent": "goodbye",
        "expected_lang": "ilocano",
        "check_response": "agyamanak"
    },

    # --- TAGALOG TESTS ---
    {
        "query": "Magandang umaga po sa inyong lahat!",
        "expected_intent": "greeting",
        "expected_lang": "tagalog",
        "check_response": "BrgyLink"
    },
    {
        "query": "Paano kumuha ng barangay clearance para sa trabaho?",
        "expected_intent": "clearance",
        "expected_lang": "tagalog",
        "check_response": "BrgyLink"
    },
    {
        "query": "Kailangan ko ng Certificate of Indigency para sa ospital",
        "expected_intent": "indigency",
        "expected_lang": "tagalog",
        "check_response": "Certificate of Indigency"
    },
    {
        "query": "Ano ang oras ng opisina ng barangay hall?",
        "expected_intent": "office_hours",
        "expected_lang": "tagalog",
        "check_response": "opisina"
    },
    {
        "query": "Magkano ang bayad sa barangay clearance?",
        "expected_intent": "fees",
        "expected_lang": "tagalog",
        "check_response": "bayarin"
    },
    {
        "query": "Kailan ang hakot ng basura sa purok namin?",
        "expected_intent": "garbage",
        "expected_lang": "tagalog",
        "check_response": "basura"
    },
    {
        "query": "Saan ako magrereport ng paulit-ulit na ingay ng kapitbahay?",
        "expected_intent": "report_incident",
        "expected_lang": "tagalog",
        "check_response": "Blotter"
    },
    {
        "query": "Paano magpa-lupon ng may utang sa akin?",
        "expected_intent": "lupon",
        "expected_lang": "tagalog",
        "check_response": "Lupon"
    },
    {
        "query": "May libreng bakuna ba sa health center para sa baby?",
        "expected_intent": "health_services",
        "expected_lang": "tagalog",
        "check_response": "health"
    },
    {
        "query": "Maraming salamat po sa malaking tulong!",
        "expected_intent": "thanks",
        "expected_lang": "tagalog",
        "check_response": "anuman"
    },
    {
        "query": "Paalam na po maraming salamat!",
        "expected_intent": "goodbye",
        "expected_lang": "tagalog",
        "check_response": "salamat"
    },

    # --- LANGUAGE SWITCHING & GENERAL CAPABILITIES ---
    {
        "query": "Ilocano ti usarem a sungbat",
        "expected_intent": "language_support",
        "expected_lang": "ilocano",
        "check_response": "Ilocano"
    },
    {
        "query": "Pangasinan so usaren mo ya pansalita",
        "expected_intent": "language_support",
        "expected_lang": "pangasinan",
        "check_response": "Pangasinan"
    },
    {
        "query": "Tagalog ang isagot mo sa akin",
        "expected_intent": "language_support",
        "expected_lang": "tagalog",
        "check_response": "Tagalog"
    }
]


def run_tests():
    passed = 0
    total = len(TEST_CASES)
    by_lang = {"pangasinan": [0, 0], "ilocano": [0, 0], "tagalog": [0, 0], "english": [0, 0]}
    failures = []

    print(f"Running {total} multilingual checks across Pangasinan, Ilocano, and Tagalog...\n")

    for tc in TEST_CASES:
        session = new_session()
        res = handle_message(tc["query"], session, model)
        ok_intent = (res["intent"] == tc["expected_intent"])
        ok_lang = (res["language"] == tc["expected_lang"])
        ok_resp = tc["check_response"].lower() in res["response"].lower()

        lang_key = tc["expected_lang"]
        if lang_key in by_lang:
            by_lang[lang_key][1] += 1

        if ok_intent and ok_lang and ok_resp:
            passed += 1
            if lang_key in by_lang:
                by_lang[lang_key][0] += 1
            print(f"✓ [{res['language'].upper()[:3]}] {tc['query'][:45]:<45} -> {res['intent']} ({res['similarity']:.2f})")
        else:
            failures.append({
                "query": tc["query"],
                "expected": f"intent={tc['expected_intent']}, lang={tc['expected_lang']}",
                "got": f"intent={res['intent']}, lang={res['language']}, resp_snippet={res['response'][:60]}..."
            })
            print(f"✗ FAIL: {tc['query']}")
            print(f"   Expected: intent={tc['expected_intent']}, lang={tc['expected_lang']}")
            print(f"   Got     : intent={res['intent']}, lang={res['language']}")

    print(f"\n==========================================")
    print(f"RESULTS: {passed}/{total} Passed ({passed/total:.1%})")
    print(f"==========================================")

    # Per-language breakdown
    print("\nRegression pass rate by language:")
    for lang, (p, t) in by_lang.items():
        if t > 0:
            print(f"  {lang:12s}: {p}/{t} ({p/t:.1%})")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  Query: {f['query']}")
            print(f"    Expected: {f['expected']}")
            print(f"    Got     : {f['got']}")
        return False
    else:
        print("\nAll multilingual checks passed.")
        return True


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
