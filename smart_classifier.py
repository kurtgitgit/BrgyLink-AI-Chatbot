"""
BrgyLink Smart Classifier v5 — High-Performance Multilingual Offline NLP Engine.
Authoritative Barangay Knowledge Engine for Barangay Bagong Pag-asa, San Jacinto.
Languages: Pangasinan, Ilocano, Tagalog, English.
"""

import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "data", "intents_brgylink_curated.json")
MODEL_FILE = os.path.join(BASE_DIR, "smart_classifier.pkl")
MODEL_VERSION = 5
FALLBACK = "fallback"
CONFIDENCE_THRESHOLD = 0.08

# ---------------------------------------------------------------------------
# Document aliases -> standard display name
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "barangay clearance": "Barangay Clearance",
    "clearance": "Barangay Clearance",
    "indigency": "Certificate of Indigency",
    "certificate of indigency": "Certificate of Indigency",
    "residency": "Certificate of Residency",
    "certificate of residency": "Certificate of Residency",
    "business clearance": "Business Clearance",
    "business permit": "Business Clearance",
    "good moral": "Certificate of Good Moral Character",
    "good moral character": "Certificate of Good Moral Character",
    "barangay id": "Barangay ID",
    "bgy id": "Barangay ID",
    "barangay i.d": "Barangay ID",
    "barangay identification": "Barangay ID",
    "cert": "Barangay Clearance",
}

DOC_TO_INTENT = {
    "Barangay Clearance": "clearance",
    "Certificate of Indigency": "indigency",
    "Certificate of Residency": "residency",
    "Business Clearance": "business_permit",
    "Certificate of Good Moral Character": "good_moral",
    "Barangay ID": "barangay_id",
}

# ---------------------------------------------------------------------------
# Alias normalization
# ---------------------------------------------------------------------------
ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bbgy\b"), "barangay"),
    (re.compile(r"\bbrgy\b"), "barangay"),
    (re.compile(r"\bcert\b"), "certificate"),
    (re.compile(r"\bclrnce\b"), "clearance"),
    (re.compile(r"\bindgncy\b"), "indigency"),
    (re.compile(r"\bindigensy\b"), "indigency"),
    (re.compile(r"\bindigincy\b"), "indigency"),
    (re.compile(r"\bresdncy\b"), "residency"),
    (re.compile(r"\bbus\.?\s*pmt\b"), "business permit"),
    (re.compile(r"\bgood\s*moral\b"), "good moral"),
    (re.compile(r"\bblotter\b"), "blotter report"),
    (re.compile(r"\baics\b"), "aics assistance"),
    (re.compile(r"\bdswd\b"), "dswd assistance"),
    (re.compile(r"\bosca\b"), "osca senior citizen"),
    (re.compile(r"\bbhc\b"), "barangay health center"),
    (re.compile(r"\b4ps\b"), "4ps pantawid pamilya"),
    (re.compile(r"\bsap\b"), "social amelioration program"),
    (re.compile(r"\bkp\b"), "katarungang pambarangay"),
    (re.compile(r"\bcomelec\b"), "comelec voter registration"),
    (re.compile(r"\botp\b"), "one time password otp verification"),
    (re.compile(r"\bpwd\b"), "person with disability pwd"),
]

# ---------------------------------------------------------------------------
# Keyword-boost dictionary {pattern: {intent: boost_value}}
# High-signal terms across Tagalog, Ilocano, Pangasinan, and English
# ---------------------------------------------------------------------------
KEYWORD_BOOSTS: list[tuple[re.Pattern, dict]] = [
    # Goodbye & Farewell (must overcome polite 'salamat')
    (re.compile(r"\b(paalam|makaalis|agpakada|innakon|goodbye|bye\s*bye|quit|exit|signing\s*off)\b"), {"goodbye": 0.50}),
    (re.compile(r"\b(salamat|maraming salamat|agyamanak|balbaleg)\b"), {"thanks": 0.25}),

    # Document intents
    (re.compile(r"\bclearance\b"), {"clearance": 0.35}),
    (re.compile(r"^cert$"), {"clearance": 0.40}),
    (re.compile(r"\bindigenc[yi]\b"), {"indigency": 0.45}),
    (re.compile(r"\bresidency\b"), {"residency": 0.35}),
    (re.compile(r"\bgood moral\b"), {"good_moral": 0.40}),
    (re.compile(r"\bbarangay i\.?d\.?\b|\bbgy id\b"), {"barangay_id": 0.50}),
    (re.compile(r"\bbusiness (clearance|permit)\b"), {"business_permit": 0.40}),
    (re.compile(r"\b(negosyo|sari.?sari|tindahan)\b"), {"business_permit": 0.25}),

    # Emergency
    (re.compile(r"\b(sunog|apoy|uram|fire|911|ambulance|ambulansya)\b"), {"emergency": 0.55}),
    (re.compile(r"\b(saklolo|tulong.*tulong|aksidente|danger|peligro)\b"), {"emergency": 0.40}),

    # Blotter / complaints / disputes
    (re.compile(r"\b(blotter|reklamo|magreklamo|ireport|panagreklamo)\b"), {"report_incident": 0.40}),
    (re.compile(r"\b(ingay|maingay|maingal|makapaingal|riri|away|nakawan|takew)\b"), {"report_incident": 0.35}),

    # Lupon / Katarungang Pambarangay
    (re.compile(r"\b(lupon|tagapamayapa|katarungang|mediation|conciliation|alitan|kolkol|kolkolan|pangkat)\b"), {"lupon": 0.50}),

    # Assistance / Ayuda / AICS
    (re.compile(r"\b(aics|ayuda|tulong.?salapi|tulong.?pinansyal|relief|4ps|pantawid|ponpon|burial)\b"), {"aics_assistance": 0.45}),

    # Senior Citizen & Solo Parent
    (re.compile(r"\b(senior citizen|osca|lolo|lola|matatken|lallakay|babbaket)\b"), {"senior_citizen": 0.50}),
    (re.compile(r"\b(solo parent|single parent|nag.?iisang magulang|agsolsolo a nagannak)\b"), {"solo_parent": 0.50}),

    # Health
    (re.compile(r"\b(health center|bhc|bakuna|vaccin\w*|prenatal|gamot|agas|bitamina|salun.?at)\b"), {"health_services": 0.45}),
    (re.compile(r"\b(ugugaw|ubbing|sanggol|bata)\b"), {"health_services": 0.20}),

    # Garbage
    (re.compile(r"\b(basura|garbage|trash|hakot|panangala na basura|panag.?ala ti basura|kolekta)\b"), {"garbage": 0.50}),

    # Office hours
    (re.compile(r"\b(office hours|oras na opisina|oras ti opisina|tanggapan|bukas.*barangay|schedule)\b"), {"office_hours": 0.45}),

    # Fees
    (re.compile(r"\b(fees?|bayad|bayar|magkano|panpiga|mano ti bayad|mano so bayad|presyo|singil)\b"), {"fees": 0.45}),

    # Voter Registration
    (re.compile(r"\b(voter|botante|comelec|rehistro.*boto|eleksyon)\b"), {"voter_registration": 0.50}),

    # SDG / Community Initiatives
    (re.compile(r"\b(sdg|community initiative|civic task|tree planting|clean.?up)\b"), {"sdg_mission": 0.50}),

    # Officials
    (re.compile(r"\b(kapitan|punong barangay|kagawad|opisyal|officials|tanod|sk chairman)\b"), {"officials": 0.45}),

    # Status / Tracking
    (re.compile(r"\b(status|track|ready for pickup|nasaan na|subaybayan|nabantayan)\b"), {"document_status": 0.35}),

    # About App
    (re.compile(r"\b(what is brgylink|ano ang brgylink|ania ti brgylink|antoy brgylink|features of brgylink)\b"), {"about_app": 0.50}),
]

# ---------------------------------------------------------------------------
# Core UI / Helper responses
# ---------------------------------------------------------------------------
HELPER_TEXTS = {
    "choose_document": {
        "english": "Which document do you need? You can request a Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, or Barangay ID.",
        "tagalog": "Anong dokumento po ang kailangan ninyo? Maaari kayong humiling ng Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, o Barangay ID.",
        "ilocano": "Ania a dokumento ti kasapulam kabsat? Mabalin ti agkiddaw iti Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, wenno Barangay ID.",
        "pangasinan": "Antoy dokumento ya kasapulan mo kabaleyan? Nayari kayon mangala na Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, odino Barangay ID."
    },
    "status": {
        "english": "Open Document Requests in BrgyLink to check the live status of your application (Pending, Processing, Ready for Pickup, Completed, or Rejected).",
        "tagalog": "Buksan ang Document Requests sa BrgyLink upang makita ang kasalukuyang status ng iyong request (Pending, Processing, Ready for Pickup, Completed, o Rejected).",
        "ilocano": "Lukatan ti Document Requests iti BrgyLink tapno makita ti kasasaad ti kineddawmo a dokumento (Pending, Processing, Ready for Pickup, Completed, wenno Rejected).",
        "pangasinan": "Lukatan so Document Requests ed BrgyLink pian nengnengen so kasalukuyan ya status na kineddeng mon dokumento (Pending, Processing, Ready for Pickup, Completed, odino Rejected)."
    }
}


def document_response(document: str, language: str) -> str:
    """Generate a fluent, localized response for a specific document request."""
    if language == "tagalog":
        return (
            f"Para humiling ng {document}, buksan ang Document Requests sa BrgyLink, "
            f"piliin ang {document}, ilagay ang layunin ng pagkuha, mag-attach ng larawan ng balidong ID, "
            f"at isumite ang request. Masusubaybayan mo ang progreso sa app."
        )
    if language == "ilocano":
        return (
            f"Tapno agkiddaw iti {document}, lukatan ti Document Requests iti BrgyLink, "
            f"piliem ti {document}, isurat ti panggep ti panagkiddaw, mangikabil iti ladawan ti valid ID, "
            f"ket isumitem. Mabalinmo a subaybayan ti progreso iti uneg ti app."
        )
    if language == "pangasinan":
        return (
            f"Pian mangikeddeng na {document}, lukatan so Document Requests ed BrgyLink, "
            f"piliyen so {document}, isulat so layunin odino rason, mangikabil na litrato na balidong ID, "
            f"tan isumite. Nayarim ya bantayan so progreso diad uneg na app."
        )
    return (
        f"To request {document}, open Document Requests in BrgyLink, "
        f"select {document}, enter its purpose, attach a photo of a valid ID, "
        f"and submit. Track the request status directly in the app."
    )


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def apply_aliases(text: str) -> str:
    for pattern, replacement in ALIASES:
        text = pattern.sub(replacement, text)
    return text


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def char_ngrams(text: str, ns=(2, 3, 4, 5)) -> Counter:
    padded = f" {text} "
    c = Counter()
    for n in ns:
        for i in range(len(padded) - n + 1):
            c[("c", padded[i:i+n])] += 1
    return c


def word_ngrams(tokens: list[str], ns=(1, 2)) -> Counter:
    c = Counter()
    for n in ns:
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i+n])
            c[("w", gram)] += 1
    return c


def featurize(text: str) -> Counter:
    norm = normalize(apply_aliases(text))
    tokens = norm.split()
    features = char_ngrams(norm)
    features.update(word_ngrams(tokens))
    return features


# ---------------------------------------------------------------------------
# Scored Language Detection (Ilocano, Pangasinan, Tagalog, English)
# ---------------------------------------------------------------------------
_LANG_RULES: dict[str, list[tuple[re.Pattern, float]]] = {
    "ilocano": [
        (re.compile(r"\b(dagiti|kadagiti|wenno|ket|tapno|iti|ken|nga|ti|met|pay|laeng|amin|ditoy|daytoy|kaniak|kenka|kadakayo|datayo|dakami|isuda|kasta)\b"), 1.4),
        (re.compile(r"\b(mabalin|agkiddaw|kasano|wen|saan|apay|sadino|kaano|kaanu|mano ti|anian|ania|anya|masapulko|masapul)\b"), 2.5),
        (re.compile(r"\b(ilocano|ilokano|kabsat|kailian|agas|ubbing|kolkol|riri|uram|panagbiag|malungsot|agyamanak|agpakada|innakon|agsapa)\b"), 3.5),
        (re.compile(r"\b(naimbag|aldaw|bigat|malem|rabii|dios ti agngina|makasungbatak|makaawat)\b"), 3.5),
        (re.compile(r"\b(mangited|mangaramid|pangngeddeng|panangsalaysay|lukatan|piliem|isumitem|panagkiddaw)\b"), 2.0),
    ],
    "pangasinan": [
        (re.compile(r"\b(saray|diad|onla|ed|odino|pian|tan|met|so|ya|kabaleyan|natan|laeng|labat|siak|sika|sikato|sikara|tayo|iner|siopa|akin|ak|la|et|to|mi|yo|da)\b"), 1.5),
        (re.compile(r"\b(mano so|panon so|antoy|anto|panon|kapigan|kasapulan|amtaen|ugugaw|makapaingal|alitan|kolkolan|apoy|ponpon|mabiin|kailangan koy|panpiga|piga)\b"), 3.2),
        (re.compile(r"\b(pangasinan|makatalos|anggad|nayarin|manggawa|man.?ingat|makaalis|balbaleg|maabig|kabuasan|ngarem|agew|labi|masantos)\b"), 3.5),
        (re.compile(r"\b(say|inkuan|nanengneng|ipaliwawa|piliyen|mangikeddeng|nengnengen|silpin)\b"), 2.0),
    ],
    "tagalog": [
        (re.compile(r"\b(ang|ng|mga|sa|ay|ko|mo|po|opo|naman|lang|ba|dito|ito|yan|yun|natin|ninyo|amin|atin|nila|siya|sila)\b"), 0.8),
        (re.compile(r"\b(magandang|kumusta|kamusta|salamat|paalam|sige|walang anuman|pasensya|maraming salamat|ingat po)\b"), 2.0),
        (re.compile(r"\b(paano|anong|ano|saan|kailan|sino|bakit|kailangan|gusto|mayroon|magkano|bayad|libre|puwede|maaari)\b"), 2.0),
        (re.compile(r"\b(tagalog|filipino|pilipino|kumuha|makuha|humiling|ireport|magreklamo|kapitan|kagawad|tanod|nakatira)\b"), 3.0),
    ],
}


def detect_language(message: str) -> str:
    """Accurately detect whether text is Ilocano, Pangasinan, Tagalog, or English."""
    value = normalize(message)
    scores = {lang: 0.0 for lang in _LANG_RULES}
    for lang, rules in _LANG_RULES.items():
        for pattern, weight in rules:
            matches = len(pattern.findall(value))
            scores[lang] += matches * weight
    best_lang = max(scores, key=scores.__getitem__)
    return best_lang if scores[best_lang] > 0.4 else "english"


# ---------------------------------------------------------------------------
# Language switch intent detection
# ---------------------------------------------------------------------------
_LANG_SWITCH = re.compile(
    r"(sagot|sumagot|reply|response|speak|salita|wika|sagutin|usaren|pansalita).{0,30}"
    r"(tagalog|filipino|ilocano|ilokano|pangasinan)"
    r"|(tagalog|filipino|ilocano|ilokano|pangasinan).{0,30}"
    r"(ang|lang|only|please|lamang|na lang|so usaren|ti usarem)",
    re.IGNORECASE,
)


def language_switch(message: str) -> bool:
    return bool(_LANG_SWITCH.search(message))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> dict:
    with open(INTENTS_FILE, encoding="utf-8") as f:
        intents = json.load(f)["intents"]

    docs = [(item["tag"], pattern) for item in intents for pattern in item["patterns"]]
    freq = Counter()
    features_list = []
    for _, text in docs:
        fv = featurize(text)
        features_list.append(fv)
        freq.update(fv.keys())

    N = len(docs)
    idf = {k: math.log((1 + N) / (1 + cnt)) + 1 for k, cnt in freq.items()}

    centroids = defaultdict(Counter)
    counts = Counter()
    for (tag, _), fv in zip(docs, features_list):
        counts[tag] += 1
        for k, v in fv.items():
            centroids[tag][k] += v * idf.get(k, 1)

    for tag in centroids:
        n = counts[tag]
        for k in centroids[tag]:
            centroids[tag][k] /= n

    multilingual_map = {}
    default_responses = {}
    for i in intents:
        tag = i["tag"]
        default_responses[tag] = i["responses"][0]
        if "multilingual_responses" in i:
            multilingual_map[tag] = i["multilingual_responses"]

    model = {
        "version": MODEL_VERSION,
        "idf": idf,
        "centroids": dict(centroids),
        "responses": default_responses,
        "multilingual_responses": multilingual_map,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


def load_model() -> dict:
    if not os.path.exists(MODEL_FILE):
        return train()
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    if model.get("version") != MODEL_VERSION:
        return train()
    return model


def cosine(left: Counter, right: dict) -> float:
    dot = sum(v * right.get(k, 0) for k, v in left.items())
    a = math.sqrt(sum(x * x for x in left.values()))
    b = math.sqrt(sum(x * x for x in right.values()))
    return dot / (a * b) if a and b else 0.0


def compute_boost(value: str) -> dict[str, float]:
    boost = defaultdict(float)
    for pattern, intent_boosts in KEYWORD_BOOSTS:
        if pattern.search(value):
            for intent, amount in intent_boosts.items():
                boost[intent] += amount
    return boost


def document_in(message: str) -> str | None:
    value = normalize(apply_aliases(message))
    return next((doc for key, doc in DOCUMENTS.items() if key in value), None)


def new_session() -> dict:
    return {
        "preferred_language": None,
        "pending": None,
        "selected_document": None,
        "last_intent": None,
        "turn_count": 0,
    }


def classify(message: str, model: dict) -> tuple[str, float]:
    fv = featurize(message)
    vec = {k: v * model["idf"].get(k, 1) for k, v in fv.items()}
    boost = compute_boost(normalize(message))

    ranked = sorted(
        (
            (tag, cosine(vec, centroid) + boost.get(tag, 0))
            for tag, centroid in model["centroids"].items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    intent, score = ranked[0]
    return (FALLBACK if score < CONFIDENCE_THRESHOLD else intent), score


def _build(intent: str, score: float, language: str, response: str, session: dict) -> dict:
    session["last_intent"] = intent
    session["turn_count"] = session.get("turn_count", 0) + 1
    return {
        "intent": intent,
        "similarity": round(score, 4),
        "language": language,
        "response": response,
        "session": session,
    }


def get_multilingual_response(intent: str, language: str, model: dict) -> str:
    """Retrieve verified multilingual answer for any intent and language."""
    multi = model.get("multilingual_responses", {})
    if intent in multi:
        return multi[intent].get(language, multi[intent].get("english", model["responses"].get(intent, "")))
    return model["responses"].get(intent, "")


def handle_message(message: str, session: dict | None = None, model: dict | None = None) -> dict:
    session = session or new_session()
    model = model or load_model()
    value = normalize(message)

    # Reset command
    if value in {"reset", "restart", "clear", "start over"}:
        session.clear()
        session.update(new_session())
        return _build("reset", 1.0, "english", "Conversation reset.", session)

    # Detect language & user switch request
    detected = detect_language(message)
    language = session["preferred_language"] or detected
    if language_switch(message):
        language = detected
        session["preferred_language"] = language

    # Compound query: "fees for X"
    _COMPOUND_DOC_PATTERN = re.compile(
        r"\b(fees?|bayad|bayar|magkano|panpiga|cost|price|singil)\b.{0,30}\b(clearance|indigency|residency|good moral|barangay id|business|permit)\b",
        re.IGNORECASE,
    )
    if _COMPOUND_DOC_PATTERN.search(value):
        doc = document_in(message)
        if doc:
            session["selected_document"] = doc
        fees_resp = get_multilingual_response("fees", language, model)
        return _build("fees", 0.95, language, fees_resp, session)

    # Compound query: "clearance for business"
    if re.search(r"\bclearance\b.{0,20}\bbusiness\b|\bbusiness\b.{0,20}\bclearance\b", value):
        return _build("business_permit", 0.95, language, document_response("Business Clearance", language), session)

    intent, score = classify(message, model)

    # Blotter status check
    if re.search(r"\b(blotter|reklamo|complaint|incident report)\b", value) and \
       re.search(r"\b(status|track|history|nasaan|naitala|in-process|update|naasikaso|pakaamta)\b", value):
        return _build("blotter_status", score, language, get_multilingual_response("blotter_status", language, model), session)

    # Document status check
    _STATUS_RE = re.compile(
        r"\b(status|track|ready for pickup|nasaan na|pending ba|approved na|processing na|rejected|kailan makukuha|kailan maaayos|done na|naaprubaran|subaybayan|nabantayan)\b",
        re.IGNORECASE,
    )
    if _STATUS_RE.search(value) or intent == "document_status":
        return _build("document_status", score, language, HELPER_TEXTS["status"][language], session)

    # Single-word / short document direct route
    document = document_in(message)
    if document and len(value.split()) <= 4 and not session.get("pending"):
        session["selected_document"] = document
        doc_intent = DOC_TO_INTENT.get(document, "document_request")
        return _build(doc_intent, 1.0, language, document_response(document, language), session)

    # Resolve pending document choice
    if session.get("pending") == "document_choice" and document:
        session["pending"] = None
        session["selected_document"] = document
        doc_intent = DOC_TO_INTENT.get(document, "document_request")
        return _build(doc_intent, 1.0, language, document_response(document, language), session)

    # Hard overrides for critical safety / emergency
    if re.search(r"\b(sunog|911|ambulance|fire|apoy|uram|tulong tulong|saklolo|aksidente|inatake)\b", value):
        return _build("emergency", score, language, get_multilingual_response("emergency", language, model), session)

    # Lupon mediation
    if re.search(r"\b(lupon|katarungang pambarangay|mediation|pangkat|alitan|kolkol|kolkolan)\b", value):
        return _build("lupon", score, language, get_multilingual_response("lupon", language, model), session)

    # Generic document request
    generic_doc = bool(re.search(
        r"\b(file|apply|request|submit|humiling|mag.?file|magfile|kailangan|masapul|kasapulan|mangikeddeng)\b.{0,30}\b(document|dokumento|certificate|sertipiko|cert)\b"
        r"|\b(document|dokumento|certificate|sertipiko)\b.{0,30}\b(file|apply|request|submit|kailangan|masapul|mangikeddeng)\b",
        value,
    ))
    if generic_doc and not document:
        session["pending"] = "document_choice"
        return _build("document_request", score, language, HELPER_TEXTS["choose_document"][language], session)

    # Named document mentioned
    if document and intent not in {"fees", FALLBACK}:
        session["selected_document"] = document
        doc_intent = DOC_TO_INTENT.get(document, intent)
        return _build(doc_intent, score, language, document_response(document, language), session)

    # Multilingual response lookup
    response = get_multilingual_response(intent, language, model)
    return _build(intent, score, language, response, session)


if __name__ == "__main__":
    m = train()
    total_patterns = sum(
        len(i["patterns"])
        for i in json.load(open(INTENTS_FILE, encoding="utf-8"))["intents"]
    )
    print(
        f"BrgyLink Smart Classifier v{MODEL_VERSION} trained and saved -> {MODEL_FILE}\n"
        f"  Intents : {len(m['responses'])}\n"
        f"  Patterns: {total_patterns}\n"
        f"  Features: {len(m['idf'])} unique TF-IDF keys\n"
        f"  Languages: Pangasinan, Ilocano, Tagalog, English"
    )
