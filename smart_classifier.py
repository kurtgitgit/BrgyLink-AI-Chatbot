"""
BrgyLink Smart Classifier v7 — Prototype Multilingual Offline NLP Engine.
Barangay assistant prototype for Barangay Bagong Pag-asa, San Jacinto.
Languages supported: Pangasinan, Ilocano, Tagalog, English.

IMPORTANT: This is a prototype. Answers come from knowledge_base.json.
Unverified facts are flagged. The chatbot does not provide legal or medical
advice and cannot dispatch emergency services.
"""

import json
import hashlib
import math
import os
import pickle
import re
import tempfile
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "data", "intents_brgylink_curated.json")
KB_FILE = os.path.join(BASE_DIR, "knowledge_base.json")
MODEL_FILE = os.path.join(BASE_DIR, "smart_classifier.pkl")
MODEL_VERSION = 7
FALLBACK = "out_of_scope"

# --- Tuned thresholds ---
CONFIDENCE_THRESHOLD = 0.18        # minimum score to accept an intent
MARGIN_THRESHOLD = 0.06            # minimum gap between #1 and #2 intents

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
# SAFETY-CRITICAL: Pre-classifier keyword rules
# These ALWAYS override the TF-IDF classifier.
# ---------------------------------------------------------------------------
_EMERGENCY_RE = re.compile(
    r"\b(sunog|apoy|uram|fire|911|ambulance|ambulansya|saklolo|tulong\s+tulong|aksidente|"
    r"inatake|heart\s*attack|aktibong\s*krimen|hostage|crime\s*in\s*progress|"
    r"nahulog|drowning|earthquake|lindol|bagyo|typhoon|flood|baha|"
    r"nawawala\w*\s*(na\s*)?(bata|tao|anak)|kidnap|active\s*shooter|nawawalang|"
    r"weapons?|active\s*assault|imminent\s*killing|severe\s*bleeding|inability\s*to\s*breathe|"
    r"unconsciousness|poisoning|binubugbog|binugbog|sinasaksak|may\s*baril|binabaril|"
    r"assault|sinakal|pinugutan|patay|pinatay|pumatay|self[\s\-]?harm|suicide|"
    r"papatayin|patayin|gustong\s*mamatay|ayaw\s*(ko\s*)?na\s*mabuhay|kill|nananaksak|"
    r"nagdudugo|bleeding|hirap\s*huminga|difficulty\s*breathing|"
    r"(?:cannot|can\s*t|cant|unable\s*to)\s*breathe|"
    r"(?:not|stopped|stops)\s*breathing|chok(?:e|es|ed|ing)|"
    r"(?:cannot|can\s*t|cant)\s*catch\s*(?:my|their|his|her|our)?\s*breath|"
    r"struggling\s*to\s*breathe|gasping\s*(?:for\s*)?air|hurt\s*myself|"
    r"hindi\s*(?:na\s*)?makahinga|di\s*makahinga|"
    r"nauubusan\s*(?:ng\s*)?hangin|saan\s*a\s*makaanges|ag\s*makainawa|"
    r"end\s*my\s*life|wakasan\s*(?:ang\s*)?(?:aking\s*)?buhay)\b",
    re.IGNORECASE,
)

_THREAT_RE = re.compile(
    r"\b(nangbabanta|nagbanta|banta|sinasaktan|inaabuso|"
    r"domestic\s*violence|violence|abuse\w*|sinaktan|pinagsasaktan|"
    r"hinoldap|holdup|rape|pangmomolestiya|molestiya|"
    r"stalker|stalking|nananakit|pinagbabantaan|binabanta|"
    r"threat\w*|harass\w*|haras\w*|pang-?aabuso|"
    r"nag-aaway|nag-aalburoto|nagwawala|naghaharas|"
    r"nagbabanta|nang-?aaway|mangdadakip)\b",
    re.IGNORECASE,
)

_MEDICAL_SYMPTOM_RE = re.compile(
    r"\b(lagnat|fever|ubo|cough|sakit\s*(ng|na)\s*(ulo|tiyan|dibdib|katawan)|"
    r"pain|masakit|sumasakit|sugat|wound|allergy|"
    r"hilo|dizzy|diarrhea|vomit|nausea|nilalagnat|may\s*sakit|"
    r"nahihilo|nasusuka|"
    r"sore|infection|impeksyon|skin\s*rash|singaw|"
    r"buntis|pregnant|prenatal|masakit\s*ang|anak\s*ko\s*may\s*sakit|"
    r"my\s*child\s*(?:has\s*a\s*fever|is\s*sick|needs\s*help))\b",
    re.IGNORECASE,
)

_LEGAL_RE = re.compile(
    r"\b(abogado|lawyer|attorney|legal\s*advice|kaso|court\s*case|"
    r"legal\s*consultation|fiscal|hukom|judge|demanda|magdemanda|"
    r"file\s*(?:a\s*)?case|sue|kasuhan|sampahan)\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_RE = re.compile(
    r"\b(passport|visa|nbi\s*clearance|psa|birth\s*certificate|"
    r"driver.?s?\s*license|sss|pag.?ibig|phil\s*health|"
    r"school\s*enrollment|tuition|"
    r"basketball|nba|movie|joke|sing\s*(me\s*)?a\s*song|poem|"
    r"weather\s*in|president\s*of|capital\s*of|recipe|"
    r"pizza|burger|game|football|calculus|cryptocurrency|bitcoin|stock\s*market|"
    r"renew\s*(my\s*)?passport|how\s*to\s*cook)\b",
    re.IGNORECASE,
)

# App-flow routing is deliberately narrow. These rules only clarify the
# current BrgyLink registration flow; they run after the safety rules above
# and do not disclose account-specific information.
_SIGNUP_OTP_RE = re.compile(
    r"(?=.*\b(?:otp|one\s*time\s*password|verification\s*code|email\s*code)\b)"
    r"(?=.*\b(?:sign\s*up|signup|register|registration|mag-?register|rehistro|parehistro)\b)",
    re.IGNORECASE,
)

_SIGNUP_RE = re.compile(
    r"\b(sign\s*up|signup|register|registration|mag\s*register|"
    r"rehistro|parehistro|agparehistro|magparehistro|panagparehistro)\b"
)
_CREATE_ACCOUNT_RE = re.compile(
    r"\b(create|make|open|new|gumawa|gagawa|agaramid|manggawa)\b.{0,30}\b(?:resident\s*)?account\b|"
    r"\baccount\b.{0,25}\b(create|make|open|new|gumawa)\b"
)
_VOTER_RE = re.compile(r"\b(voters?|vote|voting|botante|comelec|boto|eleksyon)\b")
_CODE_RE = re.compile(r"\b(otp|verification\s*code|email\s*code|code)\b")
_CODE_PROBLEM_RE = re.compile(
    r"\b(expir\w*|incorrect|invalid|wrong|never|missing|resend|"
    r"not\s*(?:receive\w*|arriv\w*)|did\s*not|didn\s*t|"
    r"hindi|di\s*dumating|wala|naawat|awan|anggapo|mali|"
    r"waiting|waited|does\s*not|doesn\s*t)\b"
)
_PASSWORD_RE = re.compile(r"\b(password|pass\s*word|pasword)\b")
_PASSWORD_PROBLEM_RE = re.compile(
    r"\b(forgot\w*|forget\w*|lost|lose|losing|reset|recover\w*|nakalimut\w*|nalipat\w*|"
    r"alingwan\w*|wrong|incorrect|invalid|mali|change|palitan)\b"
)
_ACCOUNT_RE = re.compile(r"\b(account|login|log\s*in|sign\s*in|sumrek|makapasok)\b")
_ACCOUNT_REVIEW_RE = re.compile(r"\b(pending|reject\w*|refus\w*|declin\w*|approv\w*|review|submitted|naisumite)\b")
_OFFICIALS_RE = re.compile(
    r"\b(kapitan|captain|punong\s*barangay|kagawad|opisyal|officials?|"
    r"barangay\s*(?:chairman|chairperson|council|leaders?)|councillors?|councilors?|tanod|sk\s*chairman)\b"
)
_CIVIC_RE = re.compile(r"\b(sdg|civic\s*(?:tasks?|participation)|missions?|community\s*initiative)\b")
_EVENT_RE = re.compile(r"\b(events?|community\s*activities)\b")
_ID_RE = re.compile(r"\b(valid\s*id|government\s*(?:issued\s*)?id|passport|driver\w*\s*license)\b")
_ID_PHOTO_RE = re.compile(
    r"(?=.*\b(?:id|identification|passport|licen[cs]e)\b)"
    r"(?=.*\b(?:photos?|pictures?|upload\w*|attach\w*|image|readable|blurry|photograph\w*)\b)"
)
# Never confuse a request for private credentials or somebody else's records
# with help recovering the resident's own password. No records are accessed.
_PRIVATE_DATA_RE = re.compile(
    r"\b(?:admin|administrator)\b.{0,30}\b(?:password|credentials|token|secret)\b|"
    r"\b(?:password|credentials|token|secret)\b.{0,30}\b(?:admin|administrator)\b|"
    r"\b(?:other|another|all)\s+residents?\b.{0,40}\b(?:ids?|records?|details|data)\b|"
    r"\b(?:neighbou?r\w*|someone|somebody|another|other)\b.{0,40}\b(?:password|credentials|private\s*records)\b"
)

# ---------------------------------------------------------------------------
# Keyword-boost dictionary {pattern: {intent: boost_value}}
# ---------------------------------------------------------------------------
KEYWORD_BOOSTS: list[tuple[re.Pattern, dict]] = [
    # Greetings
    (re.compile(r"\b(hello|hi|hey|good\s*(morning|afternoon|evening)|kumusta|kamusta|naimbag|maabig|magandang)\b"), {"greeting": 0.50}),

    # Goodbye & Farewell
    (re.compile(r"\b(paalam|makaalis|agpakada|innakon|goodbye|bye\s*bye|quit|exit|signing\s*off)\b"), {"goodbye": 0.50}),
    (re.compile(r"\b(salamat|maraming salamat|agyamanak|balbaleg)\b"), {"thanks": 0.25}),

    # Document intents
    (re.compile(r"\bclearance\b"), {"clearance": 0.30}),
    (re.compile(r"\bindigenc[yi]\b"), {"indigency": 0.40}),
    (re.compile(r"\bresidency\b"), {"residency": 0.35}),
    (re.compile(r"\bkatibayan\s+na\s+nakatira\b"), {"residency": 0.60}),
    (re.compile(r"\bproof\s+of\s+residen\w*\b"), {"residency": 0.55}),
    (re.compile(r"\bgood moral\b(?!\s*morning)"), {"good_moral": 0.40}),
    (re.compile(r"\bbarangay i\.?d\.?\b|\bbgy id\b"), {"barangay_id": 0.50}),
    (re.compile(r"\bbusiness (clearance|permit)\b"), {"business_permit": 0.40}),
    (re.compile(r"\b(negosyo|sari.?sari|tindahan)\b"), {"business_permit": 0.25}),

    # Emergency (supplement safety pre-classifier)
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
    (re.compile(r"\b(senior citizen|osca|matatken|lallakay|babbaket)\b"), {"senior_citizen": 0.50}),
    (re.compile(r"\b(solo parent|single parent|nag.?iisang magulang|agsolsolo a nagannak)\b"), {"solo_parent": 0.50}),

    # Health (but not medical symptoms — those go to safety pre-classifier)
    (re.compile(r"\b(health center|bhc|bakuna|vaccin\w*|prenatal|gamot|agas|bitamina|salun.?at)\b"), {"health_services": 0.45}),

    # Garbage
    (re.compile(r"\b(basura|garbage|trash|hakot|panangala na basura|panag.?ala ti basura|kolekta)\b"), {"garbage": 0.50}),

    # Office hours / Location
    (re.compile(r"\b(office hours|opening times|oras na opisina|oras ti opisina|tanggapan|schedule\s*(ng|na|ti)\s*opisina)\b|\b(?:bukas|open)\b.{0,25}\b(?:opisina|office|barangay hall)\b"), {"office_hours": 0.45}),
    (re.compile(r"\b(where\s*(?:is|can\s*i\s*find)\s*(?:the\s*)?(?:office|barangay\s*hall)|nasaan\s*(ang\s*)?opisina|saan\s*(so|ti)\s*opisina|address|location\s*(of|ng|na)?\s*(the\s*)?(office|barangay)?)\b"), {"office_hours": 0.45}),

    # Fees — must beat clearance when fee-related words present
    (re.compile(r"\b(fees?|bayad|bayar|magkano|panpiga|mano ti bayad|mano so bayad|presyo|singil|cost|price|how\s*much)\b"), {"fees": 0.50}),

    # Voter Registration
    (re.compile(r"\b(voter|botante|comelec|rehistro.*boto|eleksyon)\b"), {"voter_registration": 0.50}),

    # SDG / Community Initiatives
    (re.compile(r"\b(sdg|community initiative|civic task|tree planting|clean.?up)\b"), {"sdg_mission": 0.50}),

    # Officials
    (re.compile(r"\b(kapitan|punong barangay|kagawad|opisyal|officials|tanod|sk chairman)\b"), {"officials": 0.45}),

    # Status / Tracking
    (re.compile(r"\b(status|track|ready for pickup|nasaan na|subaybayan|nabantayan)\b"), {"document_status": 0.30}),

    # About App
    (re.compile(r"\b(what is brgylink|ano ang brgylink|ania ti brgylink|antoy brgylink|features of brgylink)\b"), {"about_app": 0.50}),
]

# ---------------------------------------------------------------------------
# Helper responses (non-factual UI prompts)
# ---------------------------------------------------------------------------
HELPER_TEXTS = {
    "choose_document": {
        "english": "Which document do you need? You can request a Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, or Barangay ID.",
        "tagalog": "Anong dokumento po ang kailangan ninyo? Maaari kayong humiling ng Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, o Barangay ID.",
        "ilocano": "Ania a dokumento ti kasapulam kabsat? Mabalin ti agkiddaw iti Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, wenno Barangay ID.",
        "pangasinan": "Antoy dokumento ya kasapulan mo kabaleyan? Nayari kayon mangala na Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, odino Barangay ID."
    },
    "status": {
        "english": "Open Document Requests in BrgyLink to check the status of your application (Pending, Processing, Ready for Pickup, Completed, or Rejected). The chatbot cannot access personal request details.",
        "tagalog": "Buksan ang Document Requests sa BrgyLink upang makita ang status ng iyong request (Pending, Processing, Ready for Pickup, Completed, o Rejected). Hindi maa-access ng chatbot ang personal na request details.",
        "ilocano": "Lukatan ti Document Requests iti BrgyLink tapno makita ti kasasaad ti kineddawmo a dokumento (Pending, Processing, Ready for Pickup, Completed, wenno Rejected). Saan a maakses ti chatbot ti personal a request details.",
        "pangasinan": "Lukatan so Document Requests ed BrgyLink pian nengnengen so status na kineddeng mon dokumento (Pending, Processing, Ready for Pickup, Completed, odino Rejected). Ag nayarin aksesan na chatbot so personal ya request details."
    },
    "legal_disclaimer": {
        "english": "The chatbot cannot provide legal advice. For legal concerns, you may visit the barangay office to inquire about Lupon Tagapamayapa (mediation) if the dispute is within barangay jurisdiction, or consult a lawyer for matters beyond barangay scope.",
        "tagalog": "Hindi makapagbigay ng legal advice ang chatbot. Para sa legal concerns, bumisita sa opisina ng barangay para sa Lupon Tagapamayapa (mediation) kung sakop ng barangay ang usapin, o kumonsulta sa abogado para sa mga bagay na lampas sa sakop ng barangay.",
        "ilocano": "Saan a makaited ti chatbot iti legal advice. Para kadagiti legal concerns, bisitaem ti opisina ti barangay tapno umammo maipapan iti Lupon Tagapamayapa (mediation) no sakop ti barangay, wenno konsultaem ti abogado para kadagiti saan a sakop.",
        "pangasinan": "Ag makaiter na legal advice so chatbot. Para ed saray legal concerns, bisitaen so opisina na barangay para ed Lupon Tagapamayapa (mediation) no sakop na barangay, odino konsultaen so abogado para ed saray agtaay ed sakop."
    },
    "unverified_disclaimer": {
        "english": "\n\n(Note: This information is not verified. Please confirm with the barangay office.)",
        "tagalog": "\n\n(Paalala: Ang impormasyong ito ay hindi pa verified. Pakikumpirma sa opisina ng barangay.)",
        "ilocano": "\n\n(Pammalagip: Daytoy nga impormasion ket saan pay a verified. Pangngaasi a kumpirmaen iti opisina ti barangay.)",
        "pangasinan": "\n\n(Paimano: Saya ya impormasyon et agni verified. Kumpirmaen ed opisina na barangay.)"
    }
}


# ---------------------------------------------------------------------------
# Knowledge Base Loader
# ---------------------------------------------------------------------------
_kb_cache: dict | None = None

def load_knowledge_base() -> dict:
    """Load knowledge_base.json and index by intent."""
    global _kb_cache
    if _kb_cache is not None:
        return _kb_cache
    with open(KB_FILE, encoding="utf-8") as f:
        kb = json.load(f)
    index = {}
    for svc in kb.get("services", []):
        index[svc["intent"]] = svc
    _kb_cache = index
    return index


def get_kb_answer(intent: str, language: str, variant: str | None = None) -> str:
    """Get the answer for an intent from the knowledge base.

    Appends an unverified-data disclaimer only when content_type is
    'barangay_fact' and the entry is not validly verified.
    System copy (greetings, fallback, etc.) never gets the disclaimer.
    """
    kb = load_knowledge_base()
    entry = kb.get(intent)

    def is_valid_answer(ans):
        if not isinstance(ans, dict):
            return False
        for lang in ["english", "tagalog", "ilocano", "pangasinan"]:
            if lang not in ans or not isinstance(ans[lang], str) or not ans[lang].strip():
                return False
        return True

    if not entry or not is_valid_answer(entry.get("answer")):
        # Fall through to out_of_scope
        entry = kb.get("out_of_scope", {})

    answer_obj = entry.get("answer", {})
    # Variants are still curated KB content, never generated factual answers.
    candidate = entry.get("answer_variants", {}).get(variant) if variant else None
    if is_valid_answer(candidate):
        answer_obj = candidate
    text = answer_obj.get(language, answer_obj.get("english", ""))

    # Only barangay_fact entries can carry the unverified disclaimer.
    content_type = entry.get("content_type", "barangay_fact")
    if content_type != "barangay_fact":
        return text

    import datetime

    is_verified = entry.get("verified", False)

    # Check verification requirements
    if is_verified:
        has_reqs = all(entry.get(f) for f in ["source", "verified_by", "verified_at", "expires_at"])
        if not has_reqs:
            is_verified = False
        else:
            try:
                expiry_date = datetime.datetime.fromisoformat(entry["expires_at"].replace("Z", "+00:00"))
                if datetime.datetime.now(datetime.timezone.utc) > expiry_date:
                    is_verified = False
            except (ValueError, TypeError, AttributeError):
                is_verified = False

    if not is_verified:
        disclaimer = HELPER_TEXTS["unverified_disclaimer"].get(language, HELPER_TEXTS["unverified_disclaimer"]["english"])
        if disclaimer not in text:
            text += disclaimer

    return text


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
    norm = normalize(apply_aliases(normalize(text)))
    tokens = norm.split()
    features = char_ngrams(norm)
    features.update(word_ngrams(tokens))
    return features


# ---------------------------------------------------------------------------
# Scored Language Detection (Ilocano, Pangasinan, Tagalog, English)
# ---------------------------------------------------------------------------
# Short common tokens that overlap between Pangasinan and English.
# These get reduced weight to prevent English from being classified as Pangasinan.
_SHORT_PAN_TOKENS = {"ed", "so", "ya", "et", "to", "mi", "yo", "da", "ak", "la"}

_LANG_RULES: dict[str, list[tuple[re.Pattern, float]]] = {
    "ilocano": [
        (re.compile(r"\b(dagiti|kadagiti|wenno|ket|tapno|ken|nga|met|pay|laeng|amin|ditoy|daytoy|kaniak|kenka|kadakayo|datayo|dakami|isuda|kasta)\b"), 1.4),
        (re.compile(r"\b(mabalin|agkiddaw|kasano|wen|saan|apay|sadino|kaano|kaanu|mano ti|anian|ania|anya|masapulko|masapul)\b"), 2.5),
        (re.compile(r"\b(ilocano|ilokano|siasino|nalipatak|makaanges|ubing|kabsat|kailian|agas|ubbing|kolkol|riri|uram|panagbiag|malungsot|agyamanak|agpakada|innakon|agsapa)\b"), 3.5),
        (re.compile(r"\b(naimbag|aldaw|bigat|malem|rabii|dios ti agngina|makasungbatak|makaawat)\b"), 3.5),
        (re.compile(r"\b(mangited|mangaramid|pangngeddeng|panangsalaysay|lukatan|piliem|isumitem|panagkiddaw)\b"), 2.0),
        # iti/ti get reduced weight to avoid false positives on short text
        (re.compile(r"\biti\b"), 0.6),
        (re.compile(r"\bti\b"), 0.4),
    ],
    "pangasinan": [
        # High-confidence Pangasinan markers
        (re.compile(r"\b(saray|diad|onla|odino|pian|kabaleyan|natan)\b"), 2.5),
        (re.compile(r"\b(mano so|panon so|antoy|anto|panon|kapigan|kasapulan|amtaen|ugugaw|makapaingal|alitan|kolkolan|apoy|ponpon|mabiin|kailangan koy|panpiga|piga)\b"), 3.2),
        (re.compile(r"\b(pangasinan|anggapo|agko|siopa|ugaw|makainawa|makatalos|anggad|nayarin|manggawa|man.?ingat|makaalis|balbaleg|maabig|kabuasan|ngarem|agew|labi|masantos)\b"), 3.5),
        (re.compile(r"\b(say|inkuan|nanengneng|ipaliwawa|piliyen|mangikeddeng|nengnengen|silpin)\b"), 2.0),
        # Pangasinan pronoun 'ak' (first-person) — medium weight
        (re.compile(r"\bak\b"), 1.5),
        # Short tokens that also appear in English — low weight
        (re.compile(r"\b(ed|so|ya|et|tan|met|labat)\b"), 0.3),
    ],
    "tagalog": [
        (re.compile(r"\b(ang|ng|mga|sa|ay|ko|mo|po|opo|naman|lang|ba|dito|ito|yan|yun|natin|ninyo|amin|atin|nila|siya|sila)\b"), 0.8),
        (re.compile(r"\b(magandang|kumusta|kamusta|salamat|paalam|sige|walang anuman|pasensya|maraming salamat|ingat po)\b"), 2.0),
        (re.compile(r"\b(paano|anong|ano|saan|kailan|sino|bakit|hindi|yung|maling|kailangan|gusto|mayroon|magkano|bayad|libre|puwede|maaari)\b"), 2.0),
        (re.compile(r"\b(tagalog|filipino|pilipino|kumuha|makuha|humiling|ireport|magreklamo|kapitan|kagawad|tanod|nakatira)\b"), 3.0),
    ],
}


def detect_language(message: str) -> str:
    """Detect whether text is Ilocano, Pangasinan, Tagalog, or English.

    Uses a scoring system with safety checks against false positives
    from short overlapping tokens (e.g. 'to', 'so', 'ed' which appear
    in both English and Pangasinan).
    """
    value = normalize(message)
    tokens = set(value.split())

    scores = {lang: 0.0 for lang in _LANG_RULES}
    match_counts = {lang: 0 for lang in _LANG_RULES}

    for lang, rules in _LANG_RULES.items():
        for pattern, weight in rules:
            matches = len(pattern.findall(value))
            if matches > 0:
                scores[lang] += matches * weight
                match_counts[lang] += matches

    best_lang = max(scores, key=scores.__getitem__)
    best_score = scores[best_lang]

    # Require a minimum threshold
    if best_score <= 0.5:
        return "english"

    # Safety: If Pangasinan won but ALL matching tokens are short/ambiguous,
    # and there are no high-confidence Pangasinan markers, default to English.
    if best_lang == "pangasinan":
        pan_only_short = tokens.issubset(
            _SHORT_PAN_TOKENS | {t for t in tokens if len(t) <= 3}
            | set(value.split())
        )
        # Check if any high-confidence Pangasinan word is present
        high_conf_pan = re.search(
            r"\b(saray|diad|onla|odino|pian|kabaleyan|pangasinan|makatalos|"
            r"maabig|kabuasan|balbaleg|antoy|panon|kapigan|nayarin|mangikeddeng|"
            r"nengnengen|say|anggad|ugugaw|kolkolan|panpiga|mabiin)\b",
            value,
        )
        if not high_conf_pan and best_score < 2.0:
            return "english"

    return best_lang


# ---------------------------------------------------------------------------
# Language switch intent detection
# ---------------------------------------------------------------------------
_LANG_SWITCH = re.compile(
    r"(sagot|sumagot|reply|response|respond|speak|salita|wika|sagutin|usaren|pansalita|answer|isagot).{0,30}"
    r"(tagalog|filipino|ilocano|ilokano|pangasinan|english|ingles)"
    r"|(tagalog|filipino|ilocano|ilokano|pangasinan|english|ingles).{0,30}"
    r"(ang|lang|only|please|lamang|na lang|so usaren|ti usarem)",
    re.IGNORECASE,
)

_LANG_SWITCH_ENGLISH = re.compile(
    r"(answer|reply|speak|respond|talk).{0,20}(in\s+)?(english|ingles)"
    r"|(english|ingles).{0,20}(please|lang|only|po)",
    re.IGNORECASE,
)

def _extract_switch_language(message: str) -> str | None:
    """Extract the target language from a language-switch request."""
    value = message.lower()
    if _LANG_SWITCH_ENGLISH.search(value):
        return "english"
    m = _LANG_SWITCH.search(value)
    if not m:
        return None
    full = m.group(0).lower()
    if "ilocano" in full or "ilokano" in full:
        return "ilocano"
    if "pangasinan" in full:
        return "pangasinan"
    if "tagalog" in full or "filipino" in full:
        return "tagalog"
    if "english" in full or "ingles" in full:
        return "english"
    return None


def language_switch(message: str) -> bool:
    return _extract_switch_language(message) is not None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> dict:
    with open(INTENTS_FILE, "rb") as f:
        training_bytes = f.read()
    intents = json.loads(training_bytes)["intents"]

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

    model = {
        "version": MODEL_VERSION,
        "training_sha256": hashlib.sha256(training_bytes).hexdigest(),
        "idf": idf,
        "centroids": dict(centroids),
    }
    # An interrupted training run must not replace a working model with a
    # partially written pickle. Save alongside the target and atomically swap.
    fd, temporary_path = tempfile.mkstemp(prefix=".classifier-", dir=os.path.dirname(MODEL_FILE))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(model, f)
        os.replace(temporary_path, MODEL_FILE)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return model


def load_model() -> dict:
    if not os.path.exists(MODEL_FILE):
        return train()
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ValueError, AttributeError):
        return train()
    with open(INTENTS_FILE, "rb") as f:
        fingerprint = hashlib.sha256(f.read()).hexdigest()
    if (not isinstance(model, dict) or model.get("version") != MODEL_VERSION
            or model.get("training_sha256") != fingerprint
            or not isinstance(model.get("idf"), dict)
            or not isinstance(model.get("centroids"), dict) or not model["centroids"]):
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
    value = normalize(apply_aliases(normalize(message)))
    # Don't match "cert" alone — too ambiguous
    return next((doc for key, doc in DOCUMENTS.items() if key != "cert" and key in value), None)


def new_session() -> dict:
    return {
        "preferred_language": None,
        "pending": None,
        "selected_document": None,
        "last_intent": None,
        "turn_count": 0,
    }


def classify(message: str, model: dict) -> tuple[str, float, float]:
    """Classify a message. Returns (intent, score, margin).

    margin is the gap between the top intent and the runner-up.
    """
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
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = score - runner_up_score

    # Strict threshold + margin requirement
    if score < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
        return FALLBACK, score, margin

    return intent, score, margin


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


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------
def handle_message(message: str, session: dict | None = None, model: dict | None = None) -> dict:
    session = session or new_session()
    model = model or load_model()
    value = normalize(message)

    # Reset command
    if value in {"reset", "restart", "clear", "start over"}:
        session.clear()
        session.update(new_session())
        return _build("reset", 1.0, "english", "Conversation reset.", session)

    # --- Language detection ---
    detected = detect_language(message)
    language = session.get("preferred_language") or detected

    # Language switch request
    switch_lang = _extract_switch_language(message)
    if switch_lang:
        language = switch_lang
        session["preferred_language"] = language

    # ===================================================================
    # SAFETY PRE-CLASSIFIER — always runs first, overrides everything
    # ===================================================================

    # 1. Emergency (life-threatening)
    if _EMERGENCY_RE.search(value):
        resp = get_kb_answer("emergency", language)
        return _build("emergency", 1.0, language, resp, session)

    # 2. Threats, violence, abuse, self-harm
    if _THREAT_RE.search(value):
        resp = get_kb_answer("safety_threat", language)
        return _build("safety_threat", 1.0, language, resp, session)

    # 3. Medical symptoms (non-emergency) — route to health guidance
    if _MEDICAL_SYMPTOM_RE.search(value):
        # Check if also emergency-level
        resp = get_kb_answer("medical_non_emergency", language)
        return _build("medical_non_emergency", 1.0, language, resp, session)

    # 4. Legal questions
    if _LEGAL_RE.search(value):
        resp = HELPER_TEXTS["legal_disclaimer"][language]
        return _build("legal_disclaimer", 1.0, language, resp, session)

    # 5. Obvious out-of-scope topics
    # Passport/licence *upload* guidance for signup is in scope, not advice
    # about obtaining or renewing those national documents.
    signup_id = bool((_SIGNUP_RE.search(value) or _CREATE_ACCOUNT_RE.search(value)) and _ID_RE.search(value))
    national_document_request = re.search(
        r"\b(renew|renewal|apply\s*for|obtain|get\s*(?:a|my))\b.{0,25}\b(passport|license)\b|"
        r"\b(passport|license)\b.{0,25}\b(renew|renewal)\b", value
    )
    scope_value = _ID_RE.sub("", value) if signup_id and not national_document_request else value
    if _PRIVATE_DATA_RE.search(value) or _OUT_OF_SCOPE_RE.search(scope_value):
        resp = get_kb_answer("out_of_scope", language)
        return _build("out_of_scope", 1.0, language, resp, session)

    # Explicit app concepts resolve common short wording without lowering the
    # classifier's uncertainty threshold. Recovery wins over signup timing.
    if _CODE_RE.search(value) and _CODE_PROBLEM_RE.search(value):
        return _build("account_help", 0.98, language, get_kb_answer("account_help", language, "otp"), session)
    if _PASSWORD_RE.search(value) and (_PASSWORD_PROBLEM_RE.search(value) or re.search(r"\bagko\s*amta\b", value)):
        return _build("account_help", 0.98, language, get_kb_answer("account_help", language, "password"), session)
    signup_question = bool(_SIGNUP_RE.search(value) or _CREATE_ACCOUNT_RE.search(value))
    if (_ACCOUNT_RE.search(value) or signup_question) and _ACCOUNT_REVIEW_RE.search(value):
        return _build("account_help", 0.98, language, get_kb_answer("account_help", language, "review"), session)
    if _ACCOUNT_RE.search(value) and not signup_question:
        variant = "review" if _ACCOUNT_REVIEW_RE.search(value) else None
        return _build("account_help", 0.98, language, get_kb_answer("account_help", language, variant), session)
    if _EVENT_RE.search(value):
        return _build("events", 0.98, language, get_kb_answer("events", language), session)
    if _CIVIC_RE.search(value):
        return _build("sdg_mission", 0.98, language, get_kb_answer("sdg_mission", language), session)
    if _OFFICIALS_RE.search(value):
        return _build("officials", 0.98, language, get_kb_answer("officials", language), session)
    if _SIGNUP_RE.search(value) and _VOTER_RE.search(value):
        return _build("voter_registration", 0.98, language, get_kb_answer("voter_registration", language), session)
    if _SIGNUP_OTP_RE.search(value) and not _VOTER_RE.search(value):
        resp = get_kb_answer("registration", language)
        return _build("registration", 0.98, language, resp, session)
    if signup_question and not _VOTER_RE.search(value):
        variant = "id" if _ID_RE.search(value) else None
        return _build("registration", 0.98, language, get_kb_answer("registration", language, variant), session)
    if _ID_RE.search(value) and re.search(r"\b(?:\d{1,2}|minor|age)\b", value):
        return _build("registration", 0.98, language, get_kb_answer("registration", language, "id"), session)
    if _ID_PHOTO_RE.search(value):
        return _build("registration", 0.98, language, get_kb_answer("registration", language, "id"), session)

    # ===================================================================
    # COMPOUND QUERY DETECTION
    # ===================================================================

    # "How much for clearance?" / "bayad sa clearance" → fees, not clearance
    _COMPOUND_FEE_RE = re.compile(
        r"\b(fees?|bayad|bayar|magkano|panpiga|cost|price|singil|how\s*much|mano)\b"
        r".{0,30}\b(clearance|indigency|residency|good moral|barangay id|business|permit|certificate|sertipiko|dokumento)\b",
        re.IGNORECASE,
    )
    _COMPOUND_FEE_RE2 = re.compile(
        r"\b(clearance|indigency|residency|good moral|barangay id|business|permit|certificate|sertipiko|dokumento)\b"
        r".{0,30}\b(fees?|bayad|bayar|magkano|panpiga|cost|price|singil|how\s*much|mano)\b",
        re.IGNORECASE,
    )
    if _COMPOUND_FEE_RE.search(value) or _COMPOUND_FEE_RE2.search(value):
        doc = document_in(message)
        if doc:
            session["selected_document"] = doc
        resp = get_kb_answer("fees", language)
        return _build("fees", 0.95, language, resp, session)

    # "clearance for business" → business_permit
    if re.search(r"\bclearance\b.{0,20}\bbusiness\b|\bbusiness\b.{0,20}\bclearance\b", value):
        resp = get_kb_answer("business_permit", language)
        return _build("business_permit", 0.95, language, resp, session)

    # ===================================================================
    # TF-IDF CLASSIFIER
    # ===================================================================
    intent, score, margin = classify(message, model)

    # Blotter status check
    if re.search(r"\b(blotter|reklamo|complaint|incident report)\b", value) and \
       re.search(r"\b(status|track|history|nasaan|naitala|in-process|update|naasikaso|pakaamta)\b", value):
        resp = get_kb_answer("blotter_status", language)
        return _build("blotter_status", score, language, resp, session)

    # Document status check — but only if the message is clearly about status
    _STATUS_RE = re.compile(
        r"\b(status|track|ready for pickup|nasaan na|pending ba|approved na|processing na|rejected|"
        r"kailan makukuha|kailan maaayos|done na|naaprubaran|subaybayan|nabantayan)\b",
        re.IGNORECASE,
    )
    _DOC_OR_NAME_RE = re.compile(
        r"\b(document|dokumento|request|hiniling|kineddaw|inkerew|"
        r"clearance|indigency|residency|business|good\s*moral|barangay\s*id|sertipiko|certificate)\b",
        re.IGNORECASE,
    )
    if _STATUS_RE.search(value) and (intent == "document_status" or _DOC_OR_NAME_RE.search(value)):
        resp = get_kb_answer("document_status", language)
        return _build("document_status", score, language, resp, session)

    # Single-word / short document direct route
    document = document_in(message)
    if document and len(value.split()) <= 4 and not session.get("pending"):
        session["selected_document"] = document
        doc_intent = DOC_TO_INTENT.get(document, "document_request")
        resp = get_kb_answer(doc_intent, language)
        return _build(doc_intent, 1.0, language, resp, session)

    # Resolve pending document choice
    if session.get("pending") == "document_choice" and document:
        session["pending"] = None
        session["selected_document"] = document
        doc_intent = DOC_TO_INTENT.get(document, "document_request")
        resp = get_kb_answer(doc_intent, language)
        return _build(doc_intent, 1.0, language, resp, session)

    # Lupon mediation
    if re.search(r"\b(lupon|katarungang pambarangay|mediation|pangkat|alitan|kolkol|kolkolan)\b", value):
        resp = get_kb_answer("lupon", language)
        return _build("lupon", score, language, resp, session)

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
        resp = get_kb_answer(doc_intent, language)
        return _build(doc_intent, score, language, resp, session)

    # ===================================================================
    # ANSWER FROM KNOWLEDGE BASE
    # ===================================================================
    if switch_lang and intent in {FALLBACK, "fallback", "language_support"}:
        intent = "language_support"
    elif intent == "fallback":
        intent = FALLBACK
    resp = get_kb_answer(intent, language)
    return _build(intent, score, language, resp, session)


if __name__ == "__main__":
    m = train()
    total_patterns = sum(
        len(i["patterns"])
        for i in json.load(open(INTENTS_FILE, encoding="utf-8"))["intents"]
    )
    print(
        f"BrgyLink Smart Classifier v{MODEL_VERSION} (prototype) trained -> {MODEL_FILE}\n"
        f"  Intents : {len(m['centroids'])}\n"
        f"  Patterns: {total_patterns}\n"
        f"  Features: {len(m['idf'])} unique TF-IDF keys\n"
        f"  Languages: Pangasinan, Ilocano, Tagalog, English\n"
        f"  Note: This is a prototype. Answers sourced from knowledge_base.json."
    )
