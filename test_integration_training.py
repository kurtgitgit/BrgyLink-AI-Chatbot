"""Regression checks for the current BrgyLink app flows.

These checks deliberately exercise real resident phrasing not included in the
original holdout set. They are local-only and do not require Flask or network
access.
"""

from smart_classifier import handle_message, load_model


def expect(message: str, intent: str, contains: str | None = None) -> None:
    result = handle_message(message, model=MODEL)
    assert result["intent"] == intent, (message, result["intent"], result["response"])
    if contains:
        assert contains.casefold() in result["response"].casefold(), (message, result["response"])


MODEL = load_model()


def main() -> None:
    # Current signup flow: email verification is Step 1, before final password.
    expect("kailan ipinapadala otp sa sign up", "registration", "Step 1")
    expect("Saan ko makukuha ang OTP sa registration?", "registration", "15")
    expect("Pwede ba mag-register ang 15 years old?", "registration", "15")
    expect("hindi dumating otp sa email kahit chineck ko spam", "account_help", "Spam")
    expect("expired na ang verification code ko", "account_help", "expired")
    expect("na-verify ko na email pero pending account ko", "account_help", "barangay review")

    # Current civic-task and event navigation.
    expect("paano submit civic task proof", "sdg_mission", "Capture Live Photo")
    expect("pwede ba mag-retake ng mission photo", "sdg_mission", "Submit Task Proof")
    expect("saan makita mga events", "events", "Events tab")
    expect("paano mag submit ng event proof", "events", "pakikilahok")

    # Roster remains safely non-dynamic until the approved integration is built.
    officials = handle_message("sino ang barangay captain ngayon", model=MODEL)
    assert officials["intent"] == "officials", officials
    assert "roster" in officials["response"].casefold(), officials["response"]

    # Safety routing must win over otherwise related barangay questions.
    expect("may nangbabanta sa akin", "safety_threat", "911")
    expect("may sunog sa bahay ng kapitbahay", "emergency", "911")
    expect("sumasakit ang tiyan ko", "medical_non_emergency", "health center")

    # Event navigation is system copy, so it must not imply an unverified live schedule.
    event = handle_message("Where are the barangay events?", model=MODEL)
    assert event["intent"] == "events", event
    assert "not verified" not in event["response"].casefold(), event["response"]

    print("Integration-training checks: 15/15 passed")


if __name__ == "__main__":
    main()
