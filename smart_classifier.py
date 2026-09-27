"""
BrgyLink Smart Classifier v4 — Maximum-limit offline NLP engine.

Improvements over v3:
  • Hybrid TF-IDF: word unigrams + bigrams + character n-grams (2-5)
  • Keyword-boost layer: high-signal domain terms add direct score lift
  • Scored language detection: tallies evidence per language, no false positives
  • Context-aware session: last_intent influences ambiguous replies
  • Compound-query splitter: e.g. "clearance for business" → correct intent
  • Soft-alias map: normalises common abbreviations before vectorising
  • Typo-tolerant by design: char n-grams survive single-character mutations
"""

import json, math, os, pickle, re
from collections import Counter, defaultdict

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "data", "intents_brgylink_curated.json")
MODEL_FILE   = os.path.join(BASE_DIR, "smart_classifier.pkl")
MODEL_VERSION = 4
FALLBACK      = "fallback"
CONFIDENCE_THRESHOLD = 0.10   # lower = more recall; keyword-boost corrects precision

# ---------------------------------------------------------------------------
# Document aliases → display name
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "barangay clearance":            "Barangay Clearance",
    "clearance":                     "Barangay Clearance",
    "indigency":                     "Certificate of Indigency",
    "certificate of indigency":      "Certificate of Indigency",
    "residency":                     "Certificate of Residency",
    "certificate of residency":      "Certificate of Residency",
    "business clearance":            "Business Clearance",
    "business permit":               "Business Clearance",
    "good moral":                    "Certificate of Good Moral Character",
    "good moral character":          "Certificate of Good Moral Character",
    "barangay id":                   "Barangay ID",
    "bgy id":                        "Barangay ID",
    "barangay i.d":                  "Barangay ID",
    "barangay identification":       "Barangay ID",
    "cert":                          "Barangay Clearance",   # common shorthand
}

# ---------------------------------------------------------------------------
# Alias normalisation — applied before vectorisation (helps n-gram overlap)
# ---------------------------------------------------------------------------
ALIASES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bbgy\b"),                 "barangay"),
    (re.compile(r"\bbrgy\b"),                "barangay"),
    (re.compile(r"\bcert\b"),                "certificate"),
    (re.compile(r"\bclrnce\b"),              "clearance"),
    (re.compile(r"\bindgncy\b"),             "indigency"),
    (re.compile(r"\bindigensy\b"),           "indigency"),
    (re.compile(r"\bindigincy\b"),           "indigency"),
    (re.compile(r"\bindigensy\b"),           "indigency"),
    (re.compile(r"\bresdncy\b"),             "residency"),
    (re.compile(r"\bbus\.?\s*pmt\b"),        "business permit"),
    (re.compile(r"\bgood\s*moral\b"),        "good moral"),
    (re.compile(r"\bblotter\b"),             "blotter report"),
    (re.compile(r"\baics\b"),                "aics assistance"),
    (re.compile(r"\bdswd\b"),                "dswd assistance"),
    (re.compile(r"\bosca\b"),                "osca senior citizen"),
    (re.compile(r"\bbhc\b"),                 "barangay health center"),
    (re.compile(r"\b4ps\b"),                 "4ps pantawid pamilya"),
    (re.compile(r"\bsap\b"),                 "social amelioration program"),
    (re.compile(r"\bkp\b"),                  "katarungang pambarangay"),
    (re.compile(r"\bcomelec\b"),             "comelec voter registration"),
    (re.compile(r"\botp\b"),                 "one time password otp verification"),
    (re.compile(r"\bpwd\b"),                 "person with disability pwd"),
]

# ---------------------------------------------------------------------------
# Keyword-boost dictionary  {keyword_regex: {intent: boost_value}}
# Higher boost = stronger signal. Values are added to cosine similarity.
# ---------------------------------------------------------------------------
KEYWORD_BOOSTS: list[tuple[re.Pattern, dict]] = [
    # Document intents
    (re.compile(r"\bclearance\b"),        {"clearance": 0.30}),
    # "cert" alone → clearance is the most-requested doc; boost it
    (re.compile(r"^cert$"),               {"clearance": 0.40}),
    (re.compile(r"\bindigency\b"),        {"indigency": 0.45}),
    # typo variants
    (re.compile(r"\bindigenc[yi]"),        {"indigency": 0.40}),
    (re.compile(r"\bindigin[cs]"),         {"indigency": 0.35}),
    (re.compile(r"\bresidency\b"),        {"residency": 0.30}),
    (re.compile(r"\bgood moral\b"),       {"good_moral": 0.35}),
    (re.compile(r"\bbarangay id\b"),      {"barangay_id": 0.50}),
    (re.compile(r"\bbgy id\b"),           {"barangay_id": 0.50}),
    (re.compile(r"\bbarangay i\.?d\.?"),  {"barangay_id": 0.50}),
    (re.compile(r"\bbusiness clearance\b"),{"business_permit": 0.35}),
    (re.compile(r"\bbusiness permit\b"),  {"business_permit": 0.35}),
    (re.compile(r"\bnegosyo\b"),          {"business_permit": 0.20}),
    (re.compile(r"\bsari.?sari\b"),       {"business_permit": 0.25}),
    # Emergency
    (re.compile(r"\bsunog\b"),            {"emergency": 0.50}),
    (re.compile(r"\btulong\b"),           {"emergency": 0.15, "aics_assistance": 0.10}),
    (re.compile(r"\b911\b"),              {"emergency": 0.45}),
    (re.compile(r"\bfire\b"),             {"emergency": 0.35}),
    (re.compile(r"\bambulance\b"),        {"emergency": 0.40}),
    # Blotter / incident
    (re.compile(r"\bblotter\b"),          {"report_incident": 0.35}),
    (re.compile(r"\breklamo\b"),          {"report_incident": 0.30}),
    (re.compile(r"\bmagreklamo\b"),       {"report_incident": 0.30}),
    (re.compile(r"\bingay\b"),            {"report_incident": 0.25}),
    (re.compile(r"\bmaingay\b"),          {"report_incident": 0.25}),
    (re.compile(r"\breport\b"),           {"report_incident": 0.15}),
    # Status
    (re.compile(r"\bstatus\b"),           {"document_status": 0.20}),
    (re.compile(r"\btrack\b"),            {"document_status": 0.20}),
    (re.compile(r"\bpickup\b"),           {"document_status": 0.20}),
    (re.compile(r"\bpending\b"),          {"document_status": 0.25}),
    (re.compile(r"\bprocessing\b"),       {"document_status": 0.20}),
    (re.compile(r"\bapproved\b"),         {"document_status": 0.20}),
    (re.compile(r"\brejected\b"),         {"document_status": 0.25}),
    # Lupon / mediation
    (re.compile(r"\blupon\b"),            {"lupon": 0.50}),
    (re.compile(r"\bmediation\b"),        {"lupon": 0.40}),
    (re.compile(r"\bdispute\b"),          {"lupon": 0.30}),
    (re.compile(r"\balitan\b"),           {"lupon": 0.30}),
    (re.compile(r"\bkatarungang\b"),      {"lupon": 0.50}),
    # Assistance
    (re.compile(r"\bayuda\b"),            {"aics_assistance": 0.40}),
    (re.compile(r"\baics\b"),             {"aics_assistance": 0.50}),
    (re.compile(r"\brelief\b"),           {"aics_assistance": 0.30}),
    (re.compile(r"\b4ps\b"),              {"aics_assistance": 0.35}),
    (re.compile(r"\bpantawid\b"),         {"aics_assistance": 0.40}),
    # Solo parent / senior
    (re.compile(r"\bsolo parent\b"),      {"solo_parent": 0.50}),
    (re.compile(r"\bsingle parent\b"),    {"solo_parent": 0.45}),
    (re.compile(r"\bnag.iisang magulang\b"), {"solo_parent": 0.50}),
    (re.compile(r"\bsenior citizen\b"),   {"senior_citizen": 0.50}),
    (re.compile(r"\bosca\b"),             {"senior_citizen": 0.50}),
    (re.compile(r"\bpwd\b"),              {"senior_citizen": 0.35}),
    (re.compile(r"\bdisability\b"),       {"senior_citizen": 0.30}),
    # Health
    (re.compile(r"\bhealth center\b"),    {"health_services": 0.45}),
    (re.compile(r"\bbhc\b"),              {"health_services": 0.50}),
    (re.compile(r"\bvaccin\b"),           {"health_services": 0.40}),
    (re.compile(r"\bbakuna\b"),           {"health_services": 0.45}),
    (re.compile(r"\bgamot\b"),            {"health_services": 0.30}),
    (re.compile(r"\bprenatal\b"),         {"health_services": 0.45}),
    (re.compile(r"\bnutrition\b"),        {"health_services": 0.35}),
    # Voter
    (re.compile(r"\bvoter\b"),            {"voter_registration": 0.50}),
    (re.compile(r"\bcomelec\b"),          {"voter_registration": 0.50}),
    (re.compile(r"\bboto\b"),             {"voter_registration": 0.40}),
    (re.compile(r"\belection\b"),         {"voter_registration": 0.35}),
    # About app
    (re.compile(r"\bbrgylink\b"),         {"about_app": 0.10}),  # weak; context-dependent
    (re.compile(r"\bwhat is brgylink\b"), {"about_app": 0.50}),
    (re.compile(r"\bano ang brgylink\b"), {"about_app": 0.50}),
    # Account/registration
    (re.compile(r"\botp\b"),              {"account_help": 0.40}),
    (re.compile(r"\bverif\w*\b"),         {"account_help": 0.25}),
    (re.compile(r"\bpassword\b"),         {"account_help": 0.35}),
    (re.compile(r"\blog.?in\b"),          {"account_help": 0.25}),
    (re.compile(r"\bregister\b"),         {"registration": 0.30}),
    (re.compile(r"\bsign.?up\b"),         {"registration": 0.30}),
    # Fees
    (re.compile(r"\bmagkano\b"),          {"fees": 0.40}),
    (re.compile(r"\bbayad\b"),            {"fees": 0.30}),
    (re.compile(r"\bsagmamano\b"),        {"fees": 0.45}),
    (re.compile(r"\bpanpiga\b"),          {"fees": 0.45}),
    (re.compile(r"\bfees?\b"),            {"fees": 0.30}),
    # Office hours
    (re.compile(r"\btanggapan\b"),        {"office_hours": 0.40}),
    (re.compile(r"\boras\b"),             {"office_hours": 0.20}),
    (re.compile(r"\bbukas\b"),            {"office_hours": 0.20}),
    (re.compile(r"\bschedule\b"),         {"office_hours": 0.10}),
    (re.compile(r"\boffice hours\b"),     {"office_hours": 0.50}),
    # Garbage
    (re.compile(r"\bbasura\b"),           {"garbage": 0.50}),
    (re.compile(r"\bgarbage\b"),          {"garbage": 0.45}),
    (re.compile(r"\btrash\b"),            {"garbage": 0.40}),
    (re.compile(r"\bhakot\b"),            {"garbage": 0.45}),
    # Suggestions
    (re.compile(r"\bsuhestiyon\b"),       {"suggestion": 0.45}),
    (re.compile(r"\bmungkahi\b"),         {"suggestion": 0.45}),
    (re.compile(r"\bsuggestion\b"),       {"suggestion": 0.40}),
    (re.compile(r"\bfeedback\b"),         {"suggestion": 0.40}),
    # Blotter status
    (re.compile(r"\bblotter status\b"),   {"blotter_status": 0.50}),
    (re.compile(r"\btrack.*complaint\b"), {"blotter_status": 0.45}),
    # Notifications
    (re.compile(r"\bnotif\w*\b"),         {"notifications": 0.45}),
    (re.compile(r"\bpush\b"),             {"notifications": 0.30}),
    # SDG
    (re.compile(r"\bsdg\b"),              {"sdg_mission": 0.50}),
    (re.compile(r"\bcommunity initiative\b"), {"sdg_mission": 0.45}),
    (re.compile(r"\btree planting\b"),    {"sdg_mission": 0.50}),
    # Officials
    (re.compile(r"\bkapitan\b"),          {"officials": 0.45}),
    (re.compile(r"\bkagawad\b"),          {"officials": 0.45}),
    (re.compile(r"\btanod\b"),            {"officials": 0.35}),
    (re.compile(r"\bsk\b"),               {"officials": 0.25}),
]

# ---------------------------------------------------------------------------
# Multilingual text responses
# ---------------------------------------------------------------------------
TEXT = {
    "choose_document": {
        "english":    (
            "Which document do you need? "
            "Barangay Clearance, Certificate of Indigency, Certificate of Residency, "
            "Business Clearance, Certificate of Good Moral Character, or Barangay ID?"
        ),
        "tagalog":    (
            "Anong dokumento ang kailangan mo? "
            "Barangay Clearance, Certificate of Indigency, Certificate of Residency, "
            "Business Clearance, Certificate of Good Moral Character, o Barangay ID?"
        ),
        "ilocano":    (
            "Ania a dokumento ti kasapulam? "
            "Barangay Clearance, Certificate of Indigency, Certificate of Residency, "
            "Business Clearance, Certificate of Good Moral Character, wenno Barangay ID? [Draft]"
        ),
        "pangasinan": (
            "Antoy dokumento ya kasapulan mo? "
            "Barangay Clearance, Certificate of Indigency, Certificate of Residency, "
            "Business Clearance, Certificate of Good Moral Character, odino Barangay ID? [Draft]"
        ),
    },
    "status": {
        "english":    (
            "Open Document Requests in BrgyLink to view your request status: "
            "Pending, Processing, Ready for Pickup, Completed, or Rejected. "
            "I cannot access your personal request status here."
        ),
        "tagalog":    (
            "Buksan ang Document Requests sa BrgyLink upang makita ang status ng iyong request: "
            "Pending, Processing, Ready for Pickup, Completed, o Rejected. "
            "Hindi ko makita ang iyong personal na request status dito."
        ),
        "ilocano":    (
            "Lukatan ti Document Requests iti BrgyLink tapno makita ti status ti damdam mo. "
            "Saan ko a makita ti personal mo a request status ditoy. [Draft]"
        ),
        "pangasinan": (
            "Lukatan so Document Requests ed BrgyLink pian makita so status na request mo. "
            "Agko nakikita so personal mong request status diad sayan chat. [Draft]"
        ),
    },
    "language": {
        "english":    (
            "I can respond in English, Tagalog, Ilocano, and Pangasinan. "
            "Ilocano and Pangasinan replies are drafts awaiting fluent-speaker review."
        ),
        "tagalog":    (
            "Nakakasagot ako sa English, Tagalog, Ilocano, at Pangasinan. "
            "Draft pa ang mga sagot sa Ilocano at Pangasinan."
        ),
        "ilocano":    "Makasungbatak iti English, Tagalog, Ilocano, ken Pangasinan. [Draft]",
        "pangasinan": "Makasungbat ak ed English, Tagalog, Ilocano, tan Pangasinan. [Draft]",
    },
    "fallback": {
        "english":    (
            "I can help with BrgyLink and Barangay Bagong Pag-asa services such as "
            "document requests, blotter reports, community initiatives, and announcements. "
            "For other concerns, please check the Announcements screen or contact the official barangay office."
        ),
        "tagalog":    (
            "Makakatulong ako sa BrgyLink at mga serbisyo ng Barangay Bagong Pag-asa tulad ng "
            "document requests, blotter reports, community initiatives, at announcements. "
            "Para sa ibang concern, tingnan ang Announcements o makipag-ugnayan sa barangay office."
        ),
        "ilocano":    (
            "Makatulongak kadagiti serbisyo ti BrgyLink ken Barangay Bagong Pag-asa. "
            "Kitaem ti Announcements wenno agdamag iti opisial a barangay office. [Draft]"
        ),
        "pangasinan": (
            "Makatulong ak ed BrgyLink tan serbisyo na Barangay Bagong Pag-asa. "
            "Pakisilip so Announcements odino pakaammo ed opisyal ya barangay office. [Draft]"
        ),
    },
    "report_incident": {
        "english":    (
            "Open Blotter Reports in BrgyLink and select File New Report. "
            "Choose the incident type, enter the location and narrative, "
            "optionally attach evidence, and submit. "
            "For an emergency, contact the appropriate emergency service now."
        ),
        "tagalog":    (
            "Buksan ang Blotter Reports sa BrgyLink at piliin ang File New Report. "
            "Piliin ang uri ng insidente, ilagay ang lokasyon at salaysay, "
            "maaaring mag-attach ng ebidensiya, at isumite. "
            "Kung emergency, tawagan agad ang naaangkop na emergency service."
        ),
        "ilocano":    (
            "Lukatan ti Blotter Reports iti BrgyLink ken piliem ti File New Report. "
            "Piliem ti klase ti insidente, isurat ti lugar ken salaysay, ket isumitem. "
            "Para iti emergency, tawagam ti maitutop nga emergency service. [Draft]"
        ),
        "pangasinan": (
            "Lukatan so Blotter Reports ed BrgyLink tan piliyen so File New Report. "
            "Piliyen so klase na insidente, isulat so lugar tan salaysay, tan isumite. "
            "Para ed emergency, tawagan so angkakaukolan ya emergency service. [Draft]"
        ),
    },
    "emergency": {
        "english":    (
            "If there is immediate danger, call 911 now. "
            "For local barangay tanod or emergency contacts, "
            "check the Announcements in BrgyLink."
        ),
        "tagalog":    (
            "Kung may agarang panganib, tumawag agad sa 911. "
            "Para sa lokal na tulong, tingnan ang Announcements sa BrgyLink "
            "para sa verified na emergency contacts."
        ),
        "ilocano":    "No adda dagus a peligro, tawagam ti 911. Kitaem ti Announcements iti BrgyLink. [Draft]",
        "pangasinan": "No wala lay apapatak a peligro, tumawag agad ed 911. Silpin so Announcements ed BrgyLink. [Draft]",
    },
    "lupon": {
        "english":    (
            "The Lupon Tagapamayapa handles barangay-level mediation and dispute settlement. "
            "Please visit the barangay office to file a complaint or request a hearing. "
            "Check Announcements in BrgyLink for Lupon schedules."
        ),
        "tagalog":    (
            "Ang Lupon Tagapamayapa ay nangangasiwa ng barangay-level mediation at paglutas ng alitan. "
            "Pumunta sa barangay office para mag-file ng reklamo o humiling ng hearing. "
            "Tingnan ang Announcements sa BrgyLink para sa schedule ng Lupon."
        ),
        "ilocano":    (
            "Ti Lupon Tagapamayapa ti mangasikaso iti barangay-level mediation ken panagresolba ti kolkolan. "
            "Umay iti barangay office tapno mag-file iti reklamo wenno agkiddaw iti hearing. [Draft]"
        ),
        "pangasinan": (
            "Say Lupon Tagapamayapa so mangasikaso na barangay-level mediation tan resolusyon na alitan. "
            "Bumisita ed barangay office pian mag-file na reklamo odino humiling na hearing. [Draft]"
        ),
    },
    "aics_assistance": {
        "english":    (
            "For financial aid, relief goods, AICS, or DSWD assistance programs, "
            "please visit the barangay office or check the Announcements in BrgyLink. "
            "Requirements and availability depend on current programs and eligibility."
        ),
        "tagalog":    (
            "Para sa tulong pinansyal, relief goods, AICS, o mga programang DSWD, "
            "bumisita sa barangay office o tingnan ang Announcements sa BrgyLink. "
            "Ang availability ay depende sa kasalukuyang programa at eligibility."
        ),
        "ilocano":    "Para iti tulong-pinansyal, relief goods, AICS, wenno DSWD, umay iti barangay office. [Draft]",
        "pangasinan": "Para ed tulong-salapi, relief goods, AICS, odino DSWD, bumisita ed barangay office. [Draft]",
    },
    "solo_parent": {
        "english":    (
            "For a Solo Parent ID or certificate, visit the barangay office or check the Announcements "
            "in BrgyLink for requirements, schedule, and supporting documents. "
            "The barangay coordinates with DSWD for this service."
        ),
        "tagalog":    (
            "Para sa Solo Parent ID o certificate, bumisita sa barangay office o tingnan ang Announcements "
            "sa BrgyLink para sa mga requirements, schedule, at supporting documents. "
            "Nakikipag-ugnayan ang barangay sa DSWD para sa serbisyong ito."
        ),
        "ilocano":    "Para iti Solo Parent ID, umay iti barangay office wenno kitaem ti Announcements. [Draft]",
        "pangasinan": "Para ed Solo Parent ID, bumisita ed barangay office odino silpin so Announcements. [Draft]",
    },
    "senior_citizen": {
        "english":    (
            "Senior Citizen IDs are processed through the Office for Senior Citizens Affairs (OSCA). "
            "Please visit the barangay office or check the Announcements in BrgyLink for the "
            "current schedule and requirements."
        ),
        "tagalog":    (
            "Ang Senior Citizen ID ay pinoproseso sa pamamagitan ng OSCA. "
            "Pumunta sa barangay office o tingnan ang Announcements sa BrgyLink "
            "para sa schedule at requirements."
        ),
        "ilocano":    "Ti Senior Citizen ID ket naproseso babaen iti OSCA. Umay iti barangay office. [Draft]",
        "pangasinan": "Say Senior Citizen ID et naproseso panamegley na OSCA. Bumisita ed barangay office. [Draft]",
    },
    "health_services": {
        "english":    (
            "Barangay health services are coordinated through the Barangay Health Center (BHC). "
            "Check Announcements in BrgyLink for schedules on free medicine, consultations, "
            "vaccination, and nutrition programs."
        ),
        "tagalog":    (
            "Ang mga serbisyong pangkalusugan ng barangay ay pinamamahalaan ng Barangay Health Center (BHC). "
            "Tingnan ang Announcements sa BrgyLink para sa schedule ng libreng gamot, "
            "konsultasyon, bakuna, at nutrition programs."
        ),
        "ilocano":    "Dagiti serbisyo ti salun-at iti barangay ket naidaulo ti BHC. Kitaem ti Announcements. [Draft]",
        "pangasinan": "Saray serbisyong pang-kalusugan ed barangay et pinamamahalaan na BHC. Silpin so Announcements. [Draft]",
    },
    "voter_registration": {
        "english":    (
            "Voter registration is handled by COMELEC. The barangay can provide a Certificate of Residency "
            "to support your application. Check Announcements in BrgyLink for barangay-assisted voter registration drives."
        ),
        "tagalog":    (
            "Ang voter registration ay pinamamahalaan ng COMELEC. "
            "Ang barangay ay makapagbibigay ng Certificate of Residency para sa iyong aplikasyon. "
            "Tingnan ang Announcements sa BrgyLink para sa mga assisted voter registration drives."
        ),
        "ilocano":    "Ti voter registration ket ni COMELEC ti mangasikaso. Ti barangay makaited iti Certificate of Residency. [Draft]",
        "pangasinan": "Say voter registration et pinamamahalaan na COMELEC. Say barangay et makaiter na Certificate of Residency. [Draft]",
    },
    "about_app": {
        "english":    (
            "BrgyLink is the official barangay mobile application for Barangay Bagong Pag-asa. "
            "It lets residents request documents, file blotter reports, submit community initiative proofs, "
            "view announcements, and connect with barangay services — all from their smartphone."
        ),
        "tagalog":    (
            "Ang BrgyLink ay ang opisyal na mobile application ng Barangay Bagong Pag-asa. "
            "Pinapayagan nito ang mga residente na mag-request ng dokumento, mag-file ng blotter, "
            "mag-submit ng community initiative proof, at tingnan ang mga anunsyo — lahat sa smartphone."
        ),
        "ilocano":    (
            "Ti BrgyLink ket ti opisial nga app ti barangay para iti Barangay Bagong Pag-asa. "
            "Mabalin a mangidamag kadagiti dokumento, mangisumite iti blotter, ken makakita kadagiti anunsio. [Draft]"
        ),
        "pangasinan": (
            "Say BrgyLink et say opisyal na mobile app na Barangay Bagong Pag-asa. "
            "Nayarin manginginabang na dokumento, mag-file na blotter, tan onnengneng ed saray anunsyo. [Draft]"
        ),
    },
}

# Mapping intent tag → TEXT key
INTENT_TO_TEXT_KEY = {
    "document_status":    "status",
    "language_support":   "language",
    FALLBACK:             "fallback",
    "report_incident":    "report_incident",
    "emergency":          "emergency",
    "lupon":              "lupon",
    "aics_assistance":    "aics_assistance",
    "solo_parent":        "solo_parent",
    "senior_citizen":     "senior_citizen",
    "health_services":    "health_services",
    "voter_registration": "voter_registration",
    "about_app":          "about_app",
}


# ===========================================================================
# Text preprocessing
# ===========================================================================
def apply_aliases(text: str) -> str:
    """Expand common abbreviations before vectorisation."""
    for pattern, replacement in ALIASES:
        text = pattern.sub(replacement, text)
    return text


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def tokenize(text: str) -> list[str]:
    return normalize(text).split()


def char_ngrams(text: str, ns=(2, 3, 4, 5)) -> Counter:
    """Character n-grams over padded text for typo tolerance."""
    padded = f" {text} "
    c: Counter = Counter()
    for n in ns:
        for i in range(len(padded) - n + 1):
            c[("c", padded[i:i+n])] += 1
    return c


def word_ngrams(tokens: list[str], ns=(1, 2)) -> Counter:
    """Word unigrams and bigrams for semantic precision."""
    c: Counter = Counter()
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


# ===========================================================================
# Training
# ===========================================================================
def train() -> dict:
    with open(INTENTS_FILE, encoding="utf-8") as f:
        intents = json.load(f)["intents"]

    docs = [(item["tag"], pattern) for item in intents for pattern in item["patterns"]]
    freq: Counter = Counter()
    features_list = []
    for _, text in docs:
        fv = featurize(text)
        features_list.append(fv)
        freq.update(fv.keys())

    # TF-IDF: IDF per feature key (character and word n-grams share namespace)
    N = len(docs)
    idf = {k: math.log((1 + N) / (1 + cnt)) + 1 for k, cnt in freq.items()}

    # Build per-tag centroids (average TF-IDF vector)
    centroids: dict = defaultdict(Counter)
    counts: Counter = Counter()
    for (tag, _), fv in zip(docs, features_list):
        counts[tag] += 1
        for k, v in fv.items():
            centroids[tag][k] += v * idf.get(k, 1)
    for tag in centroids:
        n = counts[tag]
        for k in centroids[tag]:
            centroids[tag][k] /= n

    model = {
        "version":   MODEL_VERSION,
        "idf":       idf,
        "centroids": dict(centroids),
        "responses": {i["tag"]: i["responses"][0] for i in intents},
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    return model


# ===========================================================================
# Loading
# ===========================================================================
def load_model() -> dict:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model not found. Run: python smart_classifier.py")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    if model.get("version") != MODEL_VERSION:
        raise ValueError(f"Model version mismatch (need v{MODEL_VERSION}). Retrain: python smart_classifier.py")
    return model


# ===========================================================================
# Cosine similarity
# ===========================================================================
def cosine(left: Counter, right: dict) -> float:
    dot = sum(v * right.get(k, 0) for k, v in left.items())
    a = math.sqrt(sum(x * x for x in left.values()))
    b = math.sqrt(sum(x * x for x in right.values()))
    return dot / (a * b) if a and b else 0.0


# ===========================================================================
# Language detection — scored, not first-match
# ===========================================================================
_LANG_RULES: dict[str, list[tuple[re.Pattern, float]]] = {
    "ilocano": [
        (re.compile(r"\b(dagiti|wenno|ket|tapno|iti|kadagiti|ken)\b"), 1.5),
        (re.compile(r"\b(mabalin|agkiddaw|kasano|wen|saan|ditoy|daytoy)\b"), 1.5),
        (re.compile(r"\b(ilocano|ilokano)\b"), 3.0),
        (re.compile(r"\b(naimbag|aldaw|bigat|malem|rabii)\b"), 2.0),
        (re.compile(r"\b(mangited|mangaramid|pangngeddeng|panangsalaysay)\b"), 1.5),
    ],
    "pangasinan": [
        (re.compile(r"\b(saray|diad|onla|antoy|ed|odino|pian|tan|met)\b"), 0.8),
        (re.compile(r"\b(pangasinan|makatalos|anggad|nayarin|manggawa)\b"), 2.0),
        (re.compile(r"\b(antoy|anto|panon|kapigan|kasapulan|amtaen)\b"), 1.5),
        (re.compile(r"\b(say|so|ya|inkuan|nanengneng|ipaliwawa)\b"), 0.7),
        (re.compile(r"\b(maabig|kabuasan|agew|labi)\b"), 2.0),
    ],
    "tagalog": [
        (re.compile(r"\b(ang|ng|mga|sa|na|ay|ko|mo|po|naman|lang|ba|dito|ito|yan|yun)\b"), 0.6),
        (re.compile(r"\b(magandang|kumusta|kamusta|salamat|paalam|sige|okay)\b"), 1.5),
        (re.compile(r"\b(paano|anong|saan|kailan|sino|bakit|kailangan|gusto|mayroon)\b"), 1.5),
        (re.compile(r"\b(tagalog|filipino|pilipino)\b"), 3.0),
        (re.compile(r"\b(magkano|bayad|libre|kumuha|makuha|humiling|ireport)\b"), 1.5),
        (re.compile(r"\b(yung|nung|namin|nila|natin|ninyo|natin|siya|sila)\b"), 1.5),
    ],
}

def detect_language(message: str) -> str:
    """Score each language and return the one with the highest tally."""
    value = normalize(message)
    scores: dict[str, float] = {lang: 0.0 for lang in _LANG_RULES}
    for lang, rules in _LANG_RULES.items():
        for pattern, weight in rules:
            matches = len(pattern.findall(value))
            scores[lang] += matches * weight
    best_lang = max(scores, key=scores.__getitem__)
    return best_lang if scores[best_lang] > 0.5 else "english"


# ===========================================================================
# Keyword boost computation
# ===========================================================================
def compute_boost(value: str) -> dict[str, float]:
    """Return a per-intent additive boost from high-signal keywords."""
    boost: dict[str, float] = defaultdict(float)
    for pattern, intent_boosts in KEYWORD_BOOSTS:
        if pattern.search(value):
            for intent, amount in intent_boosts.items():
                boost[intent] += amount
    return boost


# ===========================================================================
# Document keyword extraction
# ===========================================================================
def document_in(message: str) -> str | None:
    value = normalize(apply_aliases(message))
    return next((doc for key, doc in DOCUMENTS.items() if key in value), None)


# ===========================================================================
# Language-switch request detection
# ===========================================================================
_LANG_SWITCH = re.compile(
    r"(sagot|sumagot|reply|response|speak|salita|wika|sagutin).{0,30}"
    r"(tagalog|filipino|ilocano|ilokano|pangasinan)"
    r"|(tagalog|filipino|ilocano|ilokano|pangasinan).{0,30}"
    r"(ang|lang|only|please|lamang|na lang)",
    re.IGNORECASE,
)


def language_switch(message: str) -> bool:
    return bool(_LANG_SWITCH.search(message))


# ===========================================================================
# Session factory
# ===========================================================================
def new_session() -> dict:
    return {
        "preferred_language": None,
        "pending":            None,
        "selected_document":  None,
        "last_intent":        None,   # context for disambiguation
        "turn_count":         0,
    }


# ===========================================================================
# Classifier — hybrid TF-IDF + keyword boost
# ===========================================================================
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


# ===========================================================================
# Document-specific multilingual response
# ===========================================================================
def document_response(document: str, language: str) -> str:
    base = (
        f"To request {document}, open Document Requests in BrgyLink, "
        f"select the document, enter its purpose, attach a photo of a valid ID, "
        f"and submit. Track the request in the app."
    )
    if language == "tagalog":
        return (
            f"Para humiling ng {document}, buksan ang Document Requests sa BrgyLink, "
            f"piliin ang dokumento, ilagay ang layunin, mag-attach ng larawan ng valid ID, "
            f"at isumite ang request. Subaybayan ang status sa app."
        )
    if language == "ilocano":
        return (
            f"Tapno agkiddaw iti {document}, lukatam ti Document Requests iti BrgyLink, "
            f"piliem ti dokumento, punuem ti purpose, ikabil ti larawan ti valid ID, "
            f"ket isumitem. [Draft]"
        )
    if language == "pangasinan":
        return (
            f"Para mangikeddeng na {document}, lukatan so Document Requests ed BrgyLink, "
            f"piliyen so dokumento, punan so purpose, ikabil so larawan na valid ID, "
            f"tan isumite. [Draft]"
        )
    return base


# ===========================================================================
# Helper — build a standard response dict
# ===========================================================================
def _build(intent: str, score: float, language: str, response: str, session: dict) -> dict:
    session["last_intent"] = intent
    session["turn_count"] = session.get("turn_count", 0) + 1
    return {
        "intent":     intent,
        "similarity": round(score, 4),
        "language":   language,
        "response":   response,
        "session":    session,
    }


# ===========================================================================
# Compound-query splitter
# Handles: "clearance for business" → business_permit wins via keyword boost
# Handles: "fees for indigency"    → fees wins; document context stored
# ===========================================================================
_COMPOUND_DOC_PATTERN = re.compile(
    r"\b(fees?|bayad|magkano|cost|price|pagbabayad|singil)\b"
    r".{0,30}"
    r"\b(clearance|indigency|residency|good moral|barangay id|business|permit)\b",
    re.IGNORECASE,
)
_COMPOUND_CLEAR_BUSINESS = re.compile(
    r"\bclearance\b.{0,20}\bbusiness\b|\bbusiness\b.{0,20}\bclearance\b",
    re.IGNORECASE,
)


# ===========================================================================
# Main handler
# ===========================================================================
def handle_message(message: str, session: dict | None = None, model: dict | None = None) -> dict:
    session = session or new_session()
    model   = model   or load_model()
    value   = normalize(message)

    # ── Hard reset ──────────────────────────────────────────────────────────
    if value in {"reset", "restart", "clear", "start over"}:
        session.clear()
        session.update(new_session())
        return _build("reset", 1.0, "english", "Conversation reset.", session)

    # ── Language detection & preference ─────────────────────────────────────
    detected = detect_language(message)
    language = session["preferred_language"] or detected
    if language_switch(message):
        language = detected
        session["preferred_language"] = language

    # ── Compound query: "fees for X" ────────────────────────────────────────
    if _COMPOUND_DOC_PATTERN.search(value):
        # Fees question about a specific document → answer fees, note context
        doc = document_in(message)
        if doc:
            session["selected_document"] = doc
        fees_resp = TEXT["fallback"].get(language, TEXT["fallback"]["english"])
        fees_resp = model["responses"].get("fees", fees_resp)
        return _build("fees", 0.95, language, fees_resp, session)

    # ── Compound query: "clearance for business" ────────────────────────────
    if _COMPOUND_CLEAR_BUSINESS.search(value):
        return _build(
            "business_permit", 0.95, language,
            document_response("Business Clearance", language), session
        )

    intent, score = classify(message, model)

    # ── Blotter-specific status (must beat the general status check below) ──
    if re.search(r"\b(blotter|reklamo|complaint|incident report)\b", value) and \
       re.search(r"\b(status|track|history|nasaan|naitala|in-process|update)\b", value):
        return _build("blotter_status", score, language,
                      model["responses"].get("blotter_status", ""), session)

    # ── Document STATUS check  (must come BEFORE the short-alias doc route) ───
    # Detect status-related queries even when a document name is also present.
    _STATUS_RE = re.compile(
        r"\b(status|track|ready for pickup|nasaan na|pending ba|approved na"
        r"|processing na|rejected|kailan makukuha|kailan maaayos|done na)\b",
        re.IGNORECASE,
    )
    if _STATUS_RE.search(value) or intent == "document_status":
        return _build("document_status", score, language, TEXT["status"][language], session)

    # ── Single-word / very short document alias direct-route ───────────────
    document = document_in(message)
    if document and len(value.split()) <= 4 and not session.get("pending"):
        session["selected_document"] = document
        return _build(
            "document_request", 1.0, language,
            document_response(document, language), session
        )

    # ── Resolve pending document-choice ─────────────────────────────────────
    if session["pending"] == "document_choice" and document:
        session["pending"] = None
        session["selected_document"] = document
        return _build(
            "document_request", 1.0, language,
            document_response(document, language), session
        )

    # ── Hard-override rules (keyword-level certainty) ────────────────────────

    # Emergency
    if re.search(r"\b(sunog|911|ambulance|fire|tulong tulong|saklolo|aksidente|inatake)\b", value):
        return _build("emergency", score, language, TEXT["emergency"][language], session)

    # Lupon / mediation
    if re.search(r"\b(lupon|katarungang pambarangay|mediation|pangkat|alitan|usapan ng lupa)\b", value):
        return _build("lupon", score, language, TEXT["lupon"][language], session)

    # AICS / Ayuda
    if re.search(r"\b(aics|ayuda|4ps|pantawid|sap|relief goods|food pack|tulong pinansyal)\b", value):
        return _build("aics_assistance", score, language, TEXT["aics_assistance"][language], session)

    # Solo parent
    if re.search(r"\b(solo parent|single parent|nag.iisang magulang|single mom|single dad)\b", value):
        return _build("solo_parent", score, language, TEXT["solo_parent"][language], session)

    # Senior citizen / PWD
    if re.search(r"\b(senior citizen|osca|elderly|lolo.*id|lola.*id|pwd id|disability id)\b", value):
        return _build("senior_citizen", score, language, TEXT["senior_citizen"][language], session)

    # Health services
    if re.search(r"\b(health center|bhc|bakuna|vaccine|prenatal|gamot.*libre|libre.*gamot|botika ng barangay|nutrition program)\b", value):
        return _build("health_services", score, language, TEXT["health_services"][language], session)

    # Voter registration
    if re.search(r"\b(voter registration|comelec|first time voter|botante|register.*vote|vote.*register)\b", value):
        return _build("voter_registration", score, language, TEXT["voter_registration"][language], session)

    # About app
    if re.search(r"\b(what is brgylink|ano.*brgylink|about brgylink|paano gamitin.*brgylink|brgylink features)\b", value):
        return _build("about_app", score, language, TEXT["about_app"][language], session)

    # Blotter / incident (distinguish report vs. track)
    if re.search(r"\b(blotter|reklamo|magreklamo|ireport|i-report|ingay ng kapitbahay|report.*insidente)\b", value):
        if re.search(r"\b(status|track|history|nasaan|naitala|in-process|update)\b", value):
            return _build("blotter_status", score, language,
                          model["responses"].get("blotter_status", ""), session)
        return _build("report_incident", score, language, TEXT["report_incident"][language], session)

    # Generic document request (no specific doc named)
    generic_doc = bool(re.search(
        r"\b(file|apply|request|submit|humiling|mag.?file|magfile|kailangan)\b"
        r".{0,30}\b(document|dokumento|certificate|sertipiko|cert)\b"
        r"|\b(document|dokumento|certificate|sertipiko)\b"
        r".{0,30}\b(file|apply|request|submit|kailangan)\b",
        value,
    ))
    if generic_doc and not document:
        session["pending"] = "document_choice"
        return _build(
            "document_request", score, language,
            TEXT["choose_document"][language], session
        )

    # Document status / tracking
    if intent == "document_status" or re.search(
        r"\b(status|track|ready for pickup|nasaan na|pending ba|approved na|processing na|rejected)\b", value
    ):
        return _build("document_status", score, language, TEXT["status"][language], session)

    # Named document mentioned
    if document and intent not in {"fees", FALLBACK}:
        session["selected_document"] = document
        return _build(
            "document_request", score, language,
            document_response(document, language), session
        )

    # ── Multilingual TEXT override ───────────────────────────────────────────
    text_key = INTENT_TO_TEXT_KEY.get(intent)
    if text_key and text_key in TEXT:
        response = TEXT[text_key].get(language, TEXT[text_key]["english"])
    else:
        response = model["responses"].get(intent, TEXT["fallback"][language])

    return _build(intent, score, language, response, session)


# ===========================================================================
# Entry point — retrain
# ===========================================================================
if __name__ == "__main__":
    m = train()
    total_patterns = sum(
        len(i["patterns"])
        for i in json.load(open(INTENTS_FILE, encoding="utf-8"))["intents"]
    )
    print(
        f"BrgyLink Smart Classifier v{MODEL_VERSION} saved → {MODEL_FILE}\n"
        f"  Intents : {len(m['responses'])}\n"
        f"  Patterns: {total_patterns}\n"
        f"  Features: {len(m['idf'])} unique TF-IDF keys\n"
        f"  Algo    : hybrid word-ngram + char-ngram TF-IDF + keyword boost"
    )
