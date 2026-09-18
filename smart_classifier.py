"""Offline BrgyLink classifier with session-aware responses; no hosted AI API."""
import json, math, os, pickle, re
from collections import Counter, defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_FILE = os.path.join(BASE_DIR, "data", "intents_brgylink_curated.json")
MODEL_FILE = os.path.join(BASE_DIR, "smart_classifier.pkl")
FALLBACK = "fallback"
DOCUMENTS = {"barangay clearance":"Barangay Clearance", "clearance":"Barangay Clearance", "indigency":"Certificate of Indigency", "residency":"Certificate of Residency", "business clearance":"Business Clearance", "business permit":"Business Clearance", "good moral":"Certificate of Good Moral Character", "barangay id":"Barangay ID"}

TEXT = {
 "choose_document": {"english":"Which document do you need: Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, or Barangay ID?", "tagalog":"Anong dokumento ang kailangan mo: Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, o Barangay ID?", "ilocano":"Ania a dokumento ti kasapulam: Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, wenno Barangay ID? [Draft Ilocano]", "pangasinan":"Antoy dokumento ya kasapulan mo: Barangay Clearance, Certificate of Indigency, Certificate of Residency, Business Clearance, Certificate of Good Moral Character, odino Barangay ID? [Draft Pangasinan]"},
 "status": {"english":"Open Document Requests in BrgyLink to view its status: Pending, Processing, Ready for Pickup, Completed, or Rejected. I cannot view your personal request status here.", "tagalog":"Buksan ang Document Requests sa BrgyLink upang makita ang status: Pending, Processing, Ready for Pickup, Completed, o Rejected. Hindi ko nakikita ang personal mong request status dito.", "ilocano":"Lukatan ti Document Requests iti BrgyLink tapno makita ti status. Saan ko a makita ti personal mo a request status. [Draft Ilocano]", "pangasinan":"Lukatan so Document Requests ed BrgyLink pian makita so status. Agko nakikita so personal mong request status. [Draft Pangasinan]"},
 "language": {"english":"I can respond in English, Tagalog, Ilocano, and Pangasinan. Ilocano and Pangasinan replies are drafts awaiting fluent-speaker review.", "tagalog":"Nakakasagot ako sa English, Tagalog, Ilocano, at Pangasinan. Draft pa ang mga sagot sa Ilocano at Pangasinan habang hinihintay ang review ng fluent speaker.", "ilocano":"Makasungbatak iti English, Tagalog, Ilocano, ken Pangasinan. Draft pay dagiti sungbat iti Ilocano ken Pangasinan. [Draft Ilocano]", "pangasinan":"Makasungbat ak ed English, Tagalog, Ilocano, tan Pangasinan. Draft ni ingen so saray ebat ed Ilocano tan Pangasinan. [Draft Pangasinan]"},
 "fallback": {"english":"I can help with BrgyLink and Barangay Bagong Pag-asa services. Please check Announcements or contact the official barangay office for other concerns.", "tagalog":"Makakatulong ako sa BrgyLink at mga serbisyo ng Barangay Bagong Pag-asa. Para sa ibang concern, tingnan ang Announcements o makipag-ugnayan sa opisyal na barangay office.", "ilocano":"Makatulongak kadagiti serbisyo ti BrgyLink ken Barangay Bagong Pag-asa. Kitaem ti Announcements wenno agdamag iti opisial a barangay office. [Draft Ilocano]", "pangasinan":"Makatulong ak ed BrgyLink tan serbisyo na Barangay Bagong Pag-asa. Pakisilip so Announcements odino pakaammo ed opisyal ya barangay office. [Draft Pangasinan]"},
 "report_incident": {"english":"Open Blotter Reports in BrgyLink and select File New Report. Choose the incident type, enter the location and narrative, optionally attach evidence, and submit. For an emergency, contact the appropriate emergency service now.", "tagalog":"Buksan ang Blotter Reports sa BrgyLink at piliin ang File New Report. Piliin ang uri ng insidente, ilagay ang lokasyon at salaysay, maaaring mag-attach ng ebidensiya, at isumite. Kung emergency, tawagan agad ang naaangkop na emergency service.", "ilocano":"Lukatan ti Blotter Reports iti BrgyLink ken piliem ti File New Report. Piliem ti klase ti insidente, isurat ti lugar ken salaysay, ket isumitem. Para iti emergency, tawagam ti maitutop nga emergency service. [Draft Ilocano]", "pangasinan":"Lukatan so Blotter Reports ed BrgyLink tan piliyen so File New Report. Piliyen so klase na insidente, isulat so lugar tan salaysay, tan isumite. Para ed emergency, tawagan so angkakaukolan ya emergency service. [Draft Pangasinan]"},
}

def normalize(text): return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
def grams(text):
    text = f" {normalize(text)} "
    return Counter(text[i:i+n] for n in (3,4,5) for i in range(len(text)-n+1))

def train():
    with open(INTENTS_FILE, encoding="utf-8") as f: intents = json.load(f)["intents"]
    docs = [(item["tag"], pattern) for item in intents for pattern in item["patterns"]]
    freq, features = Counter(), []
    for _, text in docs:
        current = grams(text); features.append(current); freq.update(current.keys())
    idf = {key: math.log((1+len(docs))/(1+count))+1 for key,count in freq.items()}
    centroids, counts = defaultdict(Counter), Counter()
    for (tag,_), current in zip(docs,features):
        counts[tag] += 1
        for key,value in current.items(): centroids[tag][key] += value*idf[key]
    for tag in centroids:
        for key in centroids[tag]: centroids[tag][key] /= counts[tag]
    model = {"version":2,"idf":idf,"centroids":dict(centroids),"responses":{i["tag"]:i["responses"][0] for i in intents}}
    with open(MODEL_FILE,"wb") as f: pickle.dump(model,f)
    return model

def load_model():
    if not os.path.exists(MODEL_FILE): raise FileNotFoundError("No smart model found. Run: python smart_classifier.py")
    with open(MODEL_FILE,"rb") as f: model=pickle.load(f)
    if model.get("version") != 2: raise ValueError("Smart model is outdated. Run: python smart_classifier.py")
    return model

def cosine(left,right):
    dot=sum(value*right.get(key,0) for key,value in left.items())
    a=math.sqrt(sum(x*x for x in left.values())); b=math.sqrt(sum(x*x for x in right.values()))
    return dot/(a*b) if a and b else 0

def detect_language(message):
    value=normalize(message)
    if re.search(r"\b(ilocano|ilokano|mabalin|agkiddaw|kasano|dagiti|wen)\b",value): return "ilocano"
    if re.search(r"\b(pangasinan|antoy|saray|diad|onla|makatalos)\b",value): return "pangasinan"
    if re.search(r"\b(tagalog|filipino|paano|hindi|ako|ang|mga|salamat|status ng|status ko)\b",value): return "tagalog"
    return "english"

def document_in(message):
    value=normalize(message)
    return next((doc for key,doc in DOCUMENTS.items() if key in value),None)
def language_switch(message):
    value=normalize(message)
    return bool(re.search(r"(sagot|sumagot|reply|response|speak|salita|wika).{0,25}(tagalog|filipino|ilocano|ilokano|pangasinan)|(tagalog|filipino|ilocano|ilokano|pangasinan).{0,25}(ang|lang|only|please)",value))
def new_session(): return {"preferred_language":None,"pending":None,"selected_document":None}

def classify(message,model):
    raw=grams(message); vector={key:value*model["idf"][key] for key,value in raw.items() if key in model["idf"]}
    ranked=sorted(((tag,cosine(vector,centroid)) for tag,centroid in model["centroids"].items()),key=lambda item:item[1],reverse=True)
    intent,score=ranked[0]
    return (FALLBACK if intent==FALLBACK or score<.14 else intent),score

def document_response(document, language):
    if language=="tagalog": return f"Para humiling ng {document}, buksan ang Document Requests sa BrgyLink, piliin ang dokumento, ilagay ang layunin, mag-attach ng larawan ng valid ID, at isumite ang request. Subaybayan ang status sa app."
    if language=="ilocano": return f"Tapno agkiddaw iti {document}, lukatam ti Document Requests iti BrgyLink, piliem ti dokumento, punuem ti purpose, ikabil ti valid ID, ket isumitem. [Draft Ilocano]"
    if language=="pangasinan": return f"Para mangikeddeng na {document}, lukatan so Document Requests ed BrgyLink, piliyen so dokumento, punan so purpose, ikabil so valid ID, tan isumite. [Draft Pangasinan]"
    return f"To request {document}, open Document Requests in BrgyLink, select the document, enter its purpose, attach a photo of a valid ID, and submit. Track the request in the app."

def handle_message(message, session=None, model=None):
    session=session or new_session(); model=model or load_model(); value=normalize(message)
    if value in {"reset","restart","clear"}:
        session.clear(); session.update(new_session()); return {"intent":"reset","similarity":1,"language":"english","response":"Conversation reset.","session":session}
    detected=detect_language(message); language=session["preferred_language"] or detected
    if language_switch(message): language=detected; session["preferred_language"]=language
    document=document_in(message)
    if session["pending"]=="document_choice" and document:
        session["pending"]=None; session["selected_document"]=document
        return {"intent":"document_request","similarity":1,"language":language,"response":document_response(document,language),"session":session}
    intent,score=classify(message,model)
    generic_document=bool(re.search(r"\b(file|apply|request|submit|humiling|mag file|magfile)\b.*\b(document|dokumento|certificate)\b|\b(document|dokumento)\b.*\b(file|apply|request|submit)\b",value))
    if generic_document and not document:
        session["pending"]="document_choice"; return {"intent":"document_request","similarity":score,"language":language,"response":TEXT["choose_document"][language],"session":session}
    if intent == "document_status" or re.search(r"\b(status|track|ready for pickup|nasaan na)\b", value):
        return {"intent":"document_status","similarity":score,"language":language,"response":TEXT["status"][language],"session":session}
    if document and intent not in {"fees",FALLBACK}:
        session["selected_document"]=document; return {"intent":"document_request","similarity":score,"language":language,"response":document_response(document,language),"session":session}
    key={"document_status":"status","language_support":"language",FALLBACK:"fallback"}.get(intent)
    response=TEXT[key][language] if key else TEXT.get(intent, {}).get(language, model["responses"][intent])
    return {"intent":intent,"similarity":score,"language":language,"response":response,"session":session}

if __name__=="__main__":
    model=train(); print(f"Saved offline smart classifier to {MODEL_FILE} with {len(model['responses'])} intents.")
