"""Reproducibly align BrgyLink AI's local FAQ training with the current app UI.

This script intentionally changes only app-navigation guidance and classifier
examples. It does not add names, schedules, prices, contact details, or any
other unverified barangay fact.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INTENTS_PATH = ROOT / "data" / "intents_brgylink_curated.json"
KB_PATH = ROOT / "knowledge_base.json"


PATTERNS = {
    "account_help": [
        "My email verification code did not arrive",
        "Where can I resend my BrgyLink verification code?",
        "I verified my email but cannot sign in yet",
        "My account is pending approval and I cannot log in",
        "I typed the wrong email during registration",
        "The signup verification code expired",
        "Hindi dumating ang code sa email ko kahit chineck ko ang spam",
        "Paano mag-resend ng verification code sa BrgyLink?",
        "Na-verify ko na ang email ko pero hindi pa ako makapag-login",
        "Pending pa ang account ko at hindi ako makapasok",
        "Mali ang email na nailagay ko sa registration",
        "Expired na ang verification code ko",
        "Awan ti immay a verification code iti email ko",
        "Panon ti panag-resend ti verification code?",
        "Anggapo so verification code ed email ko",
        "Panon so panag-resend na verification code?",
    ],
    "registration": [
        "Can a 15 year old register for BrgyLink?",
        "When is the email code sent during sign up?",
        "When is the OTP sent during sign up?",
        "Where do I get the OTP during registration?",
        "Do I need to sign in before receiving the registration OTP?",
        "How do I verify email before creating my password?",
        "When do I upload my government ID during registration?",
        "What photo should I submit for my ID?",
        "Pwede ba mag-register ang 15 years old?",
        "Kailan ipinapadala ang email code sa sign up?",
        "Kailan ipinapadala ang OTP sa sign up?",
        "Saan ko makukuha ang OTP sa registration?",
        "Kailangan ba mag-login bago makatanggap ng registration OTP?",
        "Paano i-verify ang email bago gumawa ng password?",
        "Kailan ako mag-a-upload ng government ID?",
        "Anong klaseng larawan ng ID ang dapat i-submit?",
        "Mabalin kadi nga agparehistro ti 15 anyos?",
        "Kaano a maipatulod ti email code iti sign up?",
        "Mabalin kasi mag-register so 15 years old?",
        "Kailan ya maipatulod so email code ed sign up?",
    ],
    "sdg_mission": [
        "How do I submit a civic task proof?",
        "Submit proof for a community task",
        "My civic task photo was rejected",
        "Can I retake my mission photo?",
        "How do I join a task at home?",
        "Paano mag-submit ng proof sa civic task?",
        "Na-reject ang larawan ko sa community task",
        "Pwede ba akong mag-retake ng mission photo?",
        "Paano sumali sa task na puwedeng gawin sa bahay?",
        "Kasano ti panag-submit ti proof iti civic task?",
        "Mabalin kadi a mangretake ti mission photo?",
        "Panon so panag-submit na proof ed civic task?",
        "Nayari kasi mag-retake na mission photo?",
    ],
    "officials": [
        "Where can I see the current officials list in the app?",
        "Show the current barangay leaders",
        "Saan makikita ang listahan ng kasalukuyang opisyal sa app?",
        "Ipakita ang mga kasalukuyang lider ng barangay",
        "Sadino ti pakakitaak iti listaan dagiti agdama nga opisyal?",
        "Iner so listaan na saray kasalukuyan ya opisyal?",
    ],
}

# Development examples are deliberately separate from the frozen diagnostic
# in data/resident_readiness_cases.json. Never copy evaluation queries here.
EXTRA_PATTERNS = {
    "account_help": [
        "I cannot remember the password I chose", "Where is the password reset button?",
        "My one time email password has timed out", "The mail code says invalid",
        "The office declined my resident profile", "My resident profile is still awaiting review",
        "Hindi ko maalala ang password na ginawa ko", "Paano ibalik ang access sa account?",
        "Ayaw tanggapin ang verification code", "Wala pa rin ang email pagkatapos maghintay",
        "Nalipatak ti password ti account ko", "Saan a husto ti verification code",
        "Awan ti email a verification code kalpasan ti panaguray",
        "Alingwan ko so password ko", "Agditer so email na code",
        "Agduga so code ya insulat ko", "Antoy gawaen no rejected so account?",
    ],
    "registration": [
        "Does this app accept residents aged seventeen?", "Is fourteen below the signup minimum?",
        "Which identity photos are required when signing up?", "Is a passport information page enough for signup?",
        "Sa huling step ba gumagawa ng password?", "Kailangan ko bang malinaw ang ID sa pagrehistro?",
        "Kasano ti panagparehistro ditoy app?", "Ania ti minimum nga edad iti panagparehistro?",
        "Panon ak makagawa na BrgyLink account?", "Antoy litrato na ID ya ikarga ko?",
    ],
    "events": [
        "Show community activities I can join", "Where is the event participation submission?",
        "I want to check the details of a scheduled community event",
        "Paano makita ang pakikilahok ko sa event?", "Aling event ang puwedeng salihan?",
        "Kasano ti mangisumite ti event participation proof?", "Ania ti lugar ti event iti app?",
        "Panon so panang-submit na litrato ed event?", "Iner so event schedule ed app?",
    ],
    "officials": [
        "Who holds the Punong Barangay position?", "Where do I find the council roster?",
        "Is the barangay chairman listed here?", "Sino ang namumuno sa barangay natin?",
        "Siasino dagiti opisyal iti barangay?", "Siopa saray opisyal na barangay?",
    ],
    "sdg_mission": [
        "Where can I follow my civic proof review?", "Does the app give points for participation?",
        "Where do rejected civic submissions appear?", "Paano makita ang status ng task submission?",
        "Kasano ti mangkita iti civic task proof?", "Panon so status na civic proof ko?",
    ],
    "office_hours": [
        "When does the barangay office open?", "How can I confirm the office opening hours?",
        "Bukas na ba ang barangay hall?", "Anong oras ang pagbubukas ng opisina?",
    ],
}
for tag, examples in EXTRA_PATTERNS.items():
    PATTERNS.setdefault(tag, []).extend(examples)


EVENT_INTENT = {
    "tag": "events",
    "patterns": [
        "What barangay events are available?",
        "Where can I find community events?",
        "How do I join an event in BrgyLink?",
        "Submit participation proof for an event",
        "Where do I see the event date and location?",
        "I need to retake my event participation photo",
        "May upcoming event ba sa barangay?",
        "Saan makikita ang community events sa BrgyLink?",
        "Paano sumali sa event sa app?",
        "Paano mag-submit ng participation proof sa event?",
        "Saan makikita ang date at location ng event?",
        "Pwede bang mag-retake ng event photo?",
        "Ania dagiti event iti barangay?",
        "Sadino ti pakakitaak iti community events iti BrgyLink?",
        "Kasano ti makipasetak iti event?",
        "Antoy saray event ed barangay?",
        "Iner so community events ed BrgyLink?",
        "Panon so maki-iba ed event?",
    ],
    "responses": [
        "Open the Events tab in BrgyLink, choose an active event, read its date, time, location, and instructions, then submit a clear participation photo when the event asks for proof."
    ],
    "multilingual_responses": {
        "english": "Open the Events tab in BrgyLink, choose an active event, read its date, time, location, and instructions, then submit a clear participation photo when the event asks for proof.",
        "tagalog": "Buksan ang Events tab sa BrgyLink, pumili ng aktibong event, basahin ang petsa, oras, lugar, at mga tagubilin, at mag-submit ng malinaw na larawan ng pakikilahok kung hinihingi ang proof.",
        "ilocano": "Lukatan ti Events tab iti BrgyLink, piliem ti aktibo nga event, basaem ti petsa, oras, lugar, ken tagubilin, ket mangisumite iti nalawag a ladawan ti panagpaset no kasapulan ti proof.",
        "pangasinan": "Lukatan so Events tab ed BrgyLink, piliyen so aktibon event, basaen so petsa, oras, lugar, tan tagubilin, tan mangisumite na malinew ya litrato na pakikiba no kasapulan so proof."
    },
}


KB_UPDATES = {
    "registration": {
        "english": "To register, begin with your email in Step 1. Enter the six-digit code sent to that email before continuing. Then complete your resident details, upload clear photos of the required government-issued ID, create your password in the final step, accept the Privacy Notice and Terms of Use, and submit the registration for barangay review. Residents must be at least 15 years old. Never share your verification code.",
        "tagalog": "Para mag-register, magsimula sa email sa Step 1. Ilagay ang anim na digit na code na ipinadala sa email bago magpatuloy. Kumpletuhin ang resident details, mag-upload ng malinaw na larawan ng hinihinging government-issued ID, gumawa ng password sa huling step, tanggapin ang Privacy Notice at Terms of Use, at isumite ang registration para sa barangay review. Kailangan ay hindi bababa sa 15 taong gulang. Huwag ibahagi ang verification code.",
        "ilocano": "Tapno agparehistro, mangrugi iti email iti Step 1 ken ikabil ti innem a digit a code a naipatulod iti email sakbay a mangtuloy. Kumpletoem dagiti detalye, mangikabil iti nalawag a ladawan ti government-issued ID, ken isumitem para iti barangay review. Saan nga ibagbaga ti verification code.",
        "pangasinan": "Pian mag-register, manggapo ed email ed Step 1 tan isulat so anem ya digit a code ya naipatulod ed email antes ontuloy. Kumpletoen so detalye, mangikabil na malinew ya litrato na government-issued ID, tan isumite para ed barangay review. Agmon iter so verification code.",
    },
    "account_help": {
        "english": "For a registration code, check Inbox, Spam, and Junk and make sure the email was entered correctly. Wait for the resend timer in the app before requesting another code; an expired code needs a new request. Do not share the code. If your registration was submitted, wait for barangay review before signing in. For a forgotten password, use Forgot Password on the sign-in screen.",
        "tagalog": "Para sa registration code, tingnan ang Inbox, Spam, at Junk at tiyaking tama ang email na inilagay. Hintayin ang resend timer sa app bago humiling ulit ng code; kung expired na ang code, humiling ng bago. Huwag ibahagi ang code. Kung naisumite na ang registration, hintayin muna ang barangay review bago mag-sign in. Para sa nakalimutang password, gamitin ang Forgot Password sa sign-in screen.",
        "ilocano": "No awan ti registration code, kitaem ti Inbox, Spam, ken Junk ken siguraduem a husto ti email. Urayem ti resend timer iti app sakbay a mangkiddaw manen iti code. Saan nga ibagbaga ti code. No naipasa ti registration, urayem ti barangay review sakbay a sumrek.",
        "pangasinan": "No anggapo so registration code, nengnengen so Inbox, Spam, tan Junk tan seguroen ya duga so email. Antayan so resend timer ed app antes mangikeddeng lamet na code. Agmon iter so code. No naisumite lay registration, antayan so barangay review antes sumrek.",
    },
    "sdg_mission": {
        "english": "Open Civic Tasks & Community Events, choose an active civic task, and use Capture Live Photo to take a clear photo that shows your participation. Retake it if it is unclear, then tap Submit Task Proof. Your submission is recorded for verification; follow the task instructions and avoid unrelated or reused photos.",
        "tagalog": "Buksan ang Civic Tasks & Community Events, pumili ng aktibong civic task, at gamitin ang Capture Live Photo para kumuha ng malinaw na larawan na nagpapakita ng pakikilahok. Mag-retake kung hindi malinaw, saka i-tap ang Submit Task Proof. Naitatala ang submission para sa verification; sundin ang task instructions at huwag gumamit ng hindi kaugnay o lumang larawan.",
        "ilocano": "Lukatan ti Civic Tasks & Community Events, piliem ti aktibo a civic task, ken usarem ti Capture Live Photo para iti nalawag a ladawan ti panagpasetmo. I-retake no saan a nalawag ken i-submit ti task proof. Surotem dagiti tagubilin ti task.",
        "pangasinan": "Lukatan so Civic Tasks & Community Events, piliyen so aktibon civic task, tan usaren so Capture Live Photo para ed malinew ya litrato na pakikiba mo. I-retake no agmalinew tan isumite so task proof. Suroten so tagubilin na task.",
    },
    "office_hours": {
        "english": "I cannot confirm a physical address or live office schedule. For the location of Barangay Bagong Pag-asa, please check official BrgyLink Announcements or inquire directly with the barangay office. Use the same channels to confirm office hours before visiting.",
        "tagalog": "Hindi ko makukumpirma ang pisikal na address o kasalukuyang office schedule. Para sa lokasyon ng Barangay Bagong Pag-asa, tingnan ang official Announcements sa BrgyLink o magtanong mismo sa opisina ng barangay. Doon din kumpirmahin ang oras ng opisina bago bumisita.",
        "ilocano": "Saan a makumpirma ti chatbot ti pisikal a address wenno agdama nga iskediul ti opisina. Para iti lokasion ti Barangay Bagong Pag-asa, kitaem ti opisial nga Announcements iti BrgyLink wenno saludsodem mismo iti opisina ti barangay. Ikumpirmam met ti oras ti opisina sakbay a bumisita.",
        "pangasinan": "Ag na-confirm na chatbot so pisikal ya address odino kasalukuyan ya schedule na opisina. Para ed lokasyon na Barangay Bagong Pag-asa, nengnengen so opisyal ya Announcements ed BrgyLink odino mantepet mismo ed opisina na barangay. I-confirm met so oras na opisina antes bumisita.",
    },
    "about_app": {
        "english": "BrgyLink AI can guide you through BrgyLink topics: registration and email verification, password recovery, document requests, blotter reports, feedback, announcements, civic tasks, events, and where to check officials information. You may ask in English, Tagalog, Ilocano, or Pangasinan. I cannot access accounts, passwords, OTPs, IDs, personal submissions, live schedules, or emergency services.",
        "tagalog": "Makakatulong ang BrgyLink AI sa mga paksa sa BrgyLink: registration at email verification, password recovery, document requests, blotter reports, feedback, announcements, civic tasks, events, at kung saan titingnan ang impormasyon tungkol sa officials. Maaari kang magtanong sa English, Tagalog, Ilocano, o Pangasinan. Hindi ko naa-access ang accounts, passwords, OTPs, IDs, personal submissions, live schedules, o emergency services.",
        "ilocano": "Makatulong ti BrgyLink AI kadagiti paksa iti BrgyLink: registration ken email verification, password recovery, document requests, blotter reports, feedback, announcements, civic tasks, events, ken sadino a kitaen ti impormasyon maipapan kadagiti officials. Mabalin a mangsaludsod iti English, Tagalog, Ilocano, wenno Pangasinan. Saan a ma-access ti chatbot dagiti account, password, OTP, ID, personal submission, live schedule, wenno emergency services.",
        "pangasinan": "Nayarin makatulong so BrgyLink AI ed saray paksa ed BrgyLink: registration tan email verification, password recovery, document requests, blotter reports, feedback, announcements, civic tasks, events, tan iner ya nengnengen so impormasyon nipaakar ed officials. Nayarin magtanong ed English, Tagalog, Ilocano, odino Pangasinan. Ag na-access na chatbot so accounts, passwords, OTPs, IDs, personal submissions, live schedules, odino emergency services.",
    },
}


EVENT_KB = {
    "intent": "events",
    "title": "Community Events",
    "needs_staff_verification": False,
    "requirements": [],
    "fee_php": None,
    "processing_time": None,
    "answer": EVENT_INTENT["multilingual_responses"],
    "verified": False,
    "verified_by": None,
    "verified_at": None,
    "expires_at": None,
    "source": None,
    "last_reviewed_at": None,
    "review_notes": "App navigation copy only; event details remain in the live Events screen.",
    "content_type": "system_copy",
    "requires_verification": False,
}

KB_VARIANTS = {
    "account_help": {
        "password": {
            "english": "On the sign-in screen, tap Forgot Password, enter the email for your account, and follow the reset instructions sent to it. Check Inbox, Spam, and Junk. Use a newly requested reset link if the old one is invalid or expired. Never send your password or reset link to the chatbot. The chatbot cannot reset your password for you.",
            "tagalog": "Sa sign-in screen, i-tap ang Forgot Password, ilagay ang email ng account, at sundin ang reset instructions na ipinadala rito. Tingnan ang Inbox, Spam, at Junk. Humiling ng bagong reset link kung invalid o expired na ang luma. Huwag ipadala sa chatbot ang password o reset link. Hindi kayang i-reset ng chatbot ang password para sa iyo.",
            "ilocano": "Iti sign-in screen, i-tap ti Forgot Password, ikabil ti email ti account, ken surotem ti reset instructions iti email. Kitaem ti Inbox, Spam, ken Junk. Mangikiddaw iti baro a reset link no invalid wenno expired ti daan. Saan nga ipatulod iti chatbot ti password wenno reset link. Saan a ma-reset ti chatbot ti password mo.",
            "pangasinan": "Ed sign-in screen, i-tap so Forgot Password, isulat so email na account, tan suroten so reset instructions ed email. Nengnengen so Inbox, Spam, tan Junk. Mangikeddeng na balon reset link no invalid odino expired lay daan. Agmon ipatulod ed chatbot so password odino reset link. Ag na-reset na chatbot so password mo.",
        },
        "otp": {
            "english": "Check that the signup email is correct and look in Inbox, Spam, and Junk. If the code has not arrived, wait for the app's resend timer, then request a new code. Enter the latest six-digit code; request another if it is expired or invalid. You can edit an incorrect email on the verification screen. Never share your OTP. The chatbot cannot see your code or confirm delivery.",
            "tagalog": "Tiyaking tama ang signup email at tingnan ang Inbox, Spam, at Junk. Kung wala pa ang code, hintayin ang resend timer sa app at humiling ng bagong code. Ilagay ang pinakabagong anim na digit na code; humiling ulit kung expired o invalid. Maaari mong itama ang email sa verification screen. Huwag ibahagi ang OTP. Hindi nakikita ng chatbot ang code at hindi nito makukumpirma ang delivery.",
            "ilocano": "Siguraduem a husto ti signup email ken kitaem ti Inbox, Spam, ken Junk. No awan pay ti code, urayem ti resend timer iti app ken mangkiddaw iti baro a code. Ikabil ti kabaruan nga innem a digit a code; mangkiddaw manen no expired wenno invalid. Mabalin a baliwan ti email iti verification screen. Saan nga ibagbaga ti OTP. Saan a makita ti chatbot ti code wenno makumpirma ti delivery.",
            "pangasinan": "Seguroen ya duga so signup email tan nengnengen so Inbox, Spam, tan Junk. No anggapo ni so code, antayan so resend timer ed app tan mangikeddeng na balon code. Isulat so pinakabalo ya anem ya digit a code; mangikeddeng lamet no expired odino invalid. Nayari ya ontan so email ed verification screen. Agmon iter so OTP. Ag nanengneng na chatbot so code odino nakumpirma so delivery.",
        },
        "review": {
            "english": "Email verification does not mean the resident account has been approved. Submitted registrations still need barangay review. If the account is rejected, read any rejection reason shown and contact the barangay office about corrections. If it is approved but you still cannot sign in, check the email and password and report the exact error without sharing credentials. The chatbot cannot see or change your approval status.",
            "tagalog": "Ang email verification ay hindi pa approval ng resident account. Kailangan pa rin ng barangay review ang naisumiteng registration. Kung rejected, basahin ang ipinakitang rejection reason at magtanong sa opisina ng barangay tungkol sa pagwawasto. Kung approved pero hindi makapasok, tiyaking tama ang email at password at i-report ang eksaktong error nang hindi ibinabahagi ang credentials. Hindi nakikita o nababago ng chatbot ang approval status.",
            "ilocano": "Ti email verification ket saan pay nga approval ti resident account. Kasapulan pay ti barangay review ti naipasa a registration. No rejected, basaem ti rejection reason a naipakita ken makiuman iti opisina ti barangay maipapan iti panagkorehir. No approved ngem saan a makasrek, kitaem ti email ken password ken i-report ti error nga awan ibagbaga a credentials. Saan a makita wenno baliwan ti chatbot ti approval status.",
            "pangasinan": "Say email verification et agni approval na resident account. Kasapulan ni so barangay review na naisumiten registration. No rejected, basaen so rejection reason ya nanengneng tan mantepet ed opisina na barangay nipaakar ed panangkorehir. No approved balet agmakasumrek, nengnengen so email tan password tan i-report so error ya agiter na credentials. Ag nanengneng odino nabago na chatbot so approval status.",
        },
    },
    "registration": {
        "id": {
            "english": "Residents must be at least 15 to register. In the final signup step, select your ID type and take a clear photo: all details must be readable, with no glare, blur, or cut-off edges. Upload the front and back when requested; Passport uses the information page. If you do not have an accepted ID, ask the barangay office what to do. The chatbot cannot approve an ID or bypass the requirement.",
            "tagalog": "Kailangan ay hindi bababa sa 15 taong gulang para mag-register. Sa huling signup step, piliin ang ID type at kumuha ng malinaw na larawan: nababasa ang detalye, walang glare o blur, at hindi putol ang gilid. I-upload ang front at back kung hinihingi; para sa Passport, information page ang kailangan. Kung walang tinatanggap na ID, magtanong sa opisina ng barangay. Hindi kayang mag-approve ng ID o lumampas sa requirement ang chatbot.",
            "ilocano": "Masapul nga agtawen iti 15 wenno nangatngato tapno agparehistro. Iti maudi a signup step, piliem ti ID type ken alaem ti nalawag a ladawan: mabasa amin a detalye, awan glare wenno blur, ken saan a naputed ti igid. I-upload ti front ken back no kasapulan; ti Passport ket information page. No awan ti maawat nga ID, agsaludsod iti opisina ti barangay. Saan a maka-approve ti chatbot iti ID wenno makalabsing iti requirement.",
            "pangasinan": "Kasapulan ya 15 taon odino mas matanda pian mag-register. Ed unor ya signup step, piliyen so ID type tan alaen so malinew ya litrato: nabasa so amin ya detalye, anggapo so glare odino blur, tan agputol so gilid. I-upload so front tan back no kasapulan; say Passport et information page. No anggapoy tinatanggap ya ID, mantepet ed opisina na barangay. Ag maka-approve na ID so chatbot odino makalampas ed requirement.",
        },
    },
}

# These are navigation/safety templates, not live account or review decisions.
KB_UPDATES["registration"].update({
    "ilocano": "Tapno agparehistro, mangrugi iti email iti Step 1 ken ikabil ti innem a digit a code sakbay a mangtuloy. Kumpletoem dagiti resident details ken mang-upload iti nalawag a ladawan ti government-issued ID. Aramidem ti password iti maudi a step, awatem ti Privacy Notice ken Terms of Use, ken isumitem para iti barangay review. Masapul nga agtawen iti 15 wenno nangatngato. Saan nga ibagbaga ti verification code.",
    "pangasinan": "Pian mag-register, manggapo ed email ed Step 1 tan isulat so anem ya digit a code antes ontuloy. Kumpletoen so resident details tan mang-upload na malinew ya litrato na government-issued ID. Gawaen so password ed unor ya step, awaten so Privacy Notice tan Terms of Use, tan isumite para ed barangay review. Kasapulan ya 15 taon odino mas matanda. Agmon iter so verification code.",
})
KB_UPDATES["sdg_mission"] = {
    "english": "Open Civic Tasks & Community Events, choose an active civic task, and use Capture Live Photo to show your participation clearly. Retake unclear photos, then tap Submit Task Proof. Open Profile and Community Mission Submissions to see your own submission status and any rejection reason. Pending is not approval; the chatbot cannot see or change your result. Civic participation does not earn reward points in the current app. Follow the task instructions and do not use unrelated or reused photos.",
    "tagalog": "Buksan ang Civic Tasks & Community Events, pumili ng aktibong civic task, at gamitin ang Capture Live Photo para malinaw na ipakita ang pakikilahok. Mag-retake kung malabo at i-tap ang Submit Task Proof. Sa Profile, tingnan ang Community Mission Submissions para sa sariling submission status at rejection reason kung mayroon. Hindi pa approval ang Pending; hindi nakikita o nababago ng chatbot ang resulta. Walang reward points ang civic participation sa kasalukuyang app. Sundin ang task instructions at huwag gumamit ng hindi kaugnay o lumang larawan.",
    "ilocano": "Lukatan ti Civic Tasks & Community Events, piliem ti aktibo a civic task, ken usarem ti Capture Live Photo tapno nalawag ti panagpasetmo. I-retake no saan a nalawag ken i-tap ti Submit Task Proof. Iti Profile, kitaem ti Community Mission Submissions para iti bukodmo a submission status ken rejection reason no adda. Saan pay nga approval ti Pending; saan a makita wenno baliwan ti chatbot ti resulta. Awan ti reward points iti agdama nga app. Surotem dagiti task instructions ken saan nga usaren ti saan a kaugnay wenno daan a ladawan.",
    "pangasinan": "Lukatan so Civic Tasks & Community Events, piliyen so aktibon civic task, tan usaren so Capture Live Photo pian malinew so pakikiba mo. I-retake no agmalinew tan i-tap so Submit Task Proof. Ed Profile, nengnengen so Community Mission Submissions para ed bukod mon submission status tan rejection reason no wala. Agni approval so Pending; ag nanengneng odino nabago na chatbot so resulta. Anggapoy reward points ed kasalukuyan ya app. Suroten so task instructions tan ag-usar na agkaugnay odino daan ya litrato.",
}
KB_UPDATES["emergency"] = {
    "english": "If there is immediate danger to life or property, call 911 immediately. Do not wait for a chatbot reply. The chatbot cannot dispatch help or contact emergency services for you.",
    "tagalog": "Kung may agarang panganib sa buhay o ari-arian, tumawag agad sa 911. Huwag maghintay ng sagot ng chatbot. Hindi kayang magpadala ng tulong o tumawag sa emergency services para sa iyo ang chatbot.",
    "ilocano": "No adda dagus a peggad iti biag wenno sanikua, umawag a dagus iti 911. Saan nga urayen ti sungbat ti chatbot. Saan a makapatulod ti chatbot iti tulong wenno umawag kadagiti emergency services para kenka.",
    "pangasinan": "No walay tampol ya peligro ed bilay odino kayarian, tawagan ya tampol so 911. Ag-antayan so ebat na chatbot. Ag makapatulod na tulong so chatbot odino ontawag ed emergency services para ed sika.",
}
# General event guidance cannot claim to know a live schedule or personal result.
EVENT_KB["answer"] = {
    "english": "Open the Events tab under Civic Tasks & Community Events, choose an active event, and read its date, time, location, and instructions. If proof is requested, take a clear participation photo and tap Submit Participation Proof. Submitted is not the same as Approved. Check the app's displayed status or ask the barangay office if no review status is shown. The chatbot cannot see live event details or your personal approval result.",
    "tagalog": "Buksan ang Events tab sa Civic Tasks & Community Events, pumili ng aktibong event, at basahin ang petsa, oras, lugar, at tagubilin. Kung kailangan ng proof, kumuha ng malinaw na larawan ng pakikilahok at i-tap ang Submit Participation Proof. Hindi pareho ang Submitted at Approved. Tingnan ang status na ipinapakita ng app o magtanong sa opisina ng barangay kung walang review status. Hindi nakikita ng chatbot ang live event details o personal na approval result.",
    "ilocano": "Lukatan ti Events tab iti Civic Tasks & Community Events, piliem ti aktibo nga event, ken basaem ti petsa, oras, lugar, ken tagubilin. No kasapulan ti proof, alaem ti nalawag a ladawan ti panagpaset ken i-tap ti Submit Participation Proof. Saan nga agpada ti Submitted ken Approved. Kitaem ti status iti app wenno agsaludsod iti opisina ti barangay no awan ti review status. Saan a makita ti chatbot ti live event details wenno personal a approval result.",
    "pangasinan": "Lukatan so Events tab ed Civic Tasks & Community Events, piliyen so aktibon event, tan basaen so petsa, oras, lugar, tan tagubilin. No kasapulan so proof, alaen so malinew ya litrato na pakikiba tan i-tap so Submit Participation Proof. Agpareho so Submitted tan Approved. Nengnengen so status ed app odino mantepet ed opisina na barangay no anggapo so review status. Ag nanengneng na chatbot so live event details odino personal ya approval result.",
}
EVENT_INTENT["responses"] = [EVENT_KB["answer"]["english"]]
EVENT_INTENT["multilingual_responses"] = EVENT_KB["answer"].copy()


def append_unique(values: list[str], additions: list[str]) -> list[str]:
    seen = {value.casefold() for value in values}
    for value in additions:
        if value.casefold() not in seen:
            values.append(value)
            seen.add(value.casefold())
    return values


def main() -> None:
    intents_document = json.loads(INTENTS_PATH.read_text(encoding="utf-8"))
    intents = intents_document["intents"]
    by_tag = {item["tag"]: item for item in intents}

    for tag, additions in PATTERNS.items():
        if tag != "events":
            append_unique(by_tag[tag]["patterns"], additions)

    if "events" not in by_tag:
        fallback_index = next(index for index, item in enumerate(intents) if item["tag"] == "fallback")
        intents.insert(fallback_index, EVENT_INTENT)
    else:
        existing = by_tag["events"]
        append_unique(existing["patterns"], EVENT_INTENT["patterns"])
        existing["responses"] = EVENT_INTENT["responses"]
        existing["multilingual_responses"] = EVENT_INTENT["multilingual_responses"]
    event = next(item for item in intents if item["tag"] == "events")
    append_unique(event["patterns"], PATTERNS.get("events", []))

    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    services = kb["services"]
    service_by_intent = {item["intent"]: item for item in services}
    for intent, answer in KB_UPDATES.items():
        service_by_intent[intent]["answer"] = answer
    for intent, variants in KB_VARIANTS.items():
        service_by_intent[intent]["answer_variants"] = variants

    if "events" not in service_by_intent:
        fallback_index = next(index for index, item in enumerate(services) if item["intent"] == "fallback")
        services.insert(fallback_index, EVENT_KB)
    else:
        service_by_intent["events"].update(EVENT_KB)

    # Retired experimental trainers must not resurrect stale reward or signup
    # copy from training-response fields. Runtime still reads only the KB.
    for item in intents:
        if item["tag"] in service_by_intent:
            answer = service_by_intent[item["tag"]]["answer"]
            item["responses"] = [answer["english"]]
            item["multilingual_responses"] = answer.copy()

    from smart_classifier import normalize
    evaluation = json.loads((ROOT / "data" / "resident_readiness_cases.json").read_text(encoding="utf-8"))
    evaluation += json.loads((ROOT / "data" / "resident_expansion_cases.json").read_text(encoding="utf-8"))
    held_out = {normalize(case["message"]) for case in evaluation}
    overlap = [p for item in intents for p in item["patterns"] if normalize(p) in held_out]
    if overlap:
        raise ValueError(f"Evaluation questions leaked into training: {overlap}")

    INTENTS_PATH.write_text(json.dumps(intents_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    KB_PATH.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Prepared current-app training examples and safe knowledge-base guidance.")


if __name__ == "__main__":
    main()
