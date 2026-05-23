"""Generate per-sign hint JSON from vocabulary.csv.

Wave-1 trained signs get hand-written entries below; everything else falls back
to a generic category template. Re-run this whenever the trained-sign roster
changes or you tighten the wording.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VOCAB = ROOT / "content" / "vocabulary.csv"
HINTS_DIR = ROOT / "content" / "hints"

# Generic per-category fallbacks for untrained signs.
TEMPLATES = {
    "greetings": {
        "handshape": "Use a natural relaxed handshape appropriate for this greeting sign.",
        "movement": "Make one clear movement; avoid repeating nervously.",
        "location": "Sign in front of your chest or neutral space.",
        "framing": "Keep both hands inside the camera guide box.",
    },
    "numbers": {
        "handshape": "Form the correct number handshape with clear finger separation.",
        "movement": "Hold the handshape steady without wiggling fingers.",
        "location": "Sign at chest height in the center of the frame.",
        "framing": "Face the palm toward the camera when required for the number.",
    },
    "colors": {
        "handshape": "Use the specific handshape for this color (often finger variation).",
        "movement": "Shake or brush movement should match the color sign pattern.",
        "location": "Sign near the chin or neutral space as taught for colors.",
        "framing": "Do not block your face with your hands.",
    },
    "default": {
        "handshape": "Check the handshape for this concept; fingers extended or bent as required.",
        "movement": "Use the correct repeated or single movement path.",
        "location": "Sign in the correct spatial zone (face, chest, or neutral).",
        "framing": "Hold the sign still for one second inside the guide box.",
    },
}

# Detailed per-sign hints for the Wave-1 trained roster (matches content/wave1_signs.csv).
# Each entry must define handshape, movement, location, orientation, framing, common_confusions.
SPECIFIC = {
    "hello": {
        "handshape": "Open B (flat hand, fingers together, thumb tucked alongside).",
        "movement": "Touch the thumb side of the hand to your forehead, then move outward and slightly down (like a casual salute).",
        "location": "Starts at the temple/forehead, ends in front of the cheek.",
        "orientation": "Palm faces left (for a right-handed signer); the back of the hand faces the camera at the start.",
        "framing": "Keep your face fully visible; the hand should be next to the head, not blocking it.",
        "common_confusions": "Don't snap the wrist like a military salute. Don't tap repeatedly.",
    },
    "goodbye": {
        "handshape": "Open hand (5), fingers slightly relaxed.",
        "movement": "Fingers flex down to the palm twice (like a small wave/curl), keeping the wrist still.",
        "location": "Hand held up at shoulder height, palm facing forward.",
        "orientation": "Palm faces the camera throughout.",
        "framing": "Keep the hand inside the box; don't swing it across your body.",
        "common_confusions": "This is finger-flex, not a side-to-side wave. The full-arm wave is a different gesture.",
    },
    "please": {
        "handshape": "Open hand (flat B / 5), palm flat against the chest.",
        "movement": "Circular motion on the chest, clockwise from your point of view, two to three small circles.",
        "location": "Center of the chest, over the sternum.",
        "orientation": "Palm flat against the body the entire time.",
        "framing": "Hand should stay on the chest, not float in front of it.",
        "common_confusions": "Don't confuse with SORRY (same circle but with a closed fist). PLEASE uses an open flat hand.",
    },
    "thank_you": {
        "handshape": "Open hand (flat B), fingers together.",
        "movement": "Fingertips touch the chin, then move forward and down toward the person you're thanking, ending palm-up.",
        "location": "Starts at the chin, ends in neutral space in front of you.",
        "orientation": "Palm faces in (toward you) at the chin, then rotates to palm-up as it moves forward.",
        "framing": "Make sure the chin start position is visible above the guide box.",
        "common_confusions": "Don't blow a kiss; the hand should be flat, not pursed. Movement is forward-and-down, not just outward.",
    },
    "sorry": {
        "handshape": "Closed fist (S handshape), thumb wrapped over fingers.",
        "movement": "Circular motion on the chest, two to three small clockwise circles (from your point of view).",
        "location": "Center of the chest, over the sternum.",
        "orientation": "Knuckles face away from the body; the back of the fist circles against the chest.",
        "framing": "Hand stays in contact with the chest; don't drift off the body.",
        "common_confusions": "PLEASE uses the same circle but with an OPEN flat hand. Keep the fist tight to distinguish.",
    },
    "yes": {
        "handshape": "Closed fist (S handshape), thumb on the outside.",
        "movement": "Bob the fist up and down at the wrist, like a head nodding 'yes'. Two clear nods.",
        "location": "Neutral space in front of the chest.",
        "orientation": "Knuckles face up; the fist 'looks' forward like a head.",
        "framing": "Single hand, held at chest height inside the box.",
        "common_confusions": "Don't move the whole arm. The motion is at the wrist only.",
    },
    "no": {
        "handshape": "Index finger and middle finger extended, thumb extended (like a 3 with the thumb out).",
        "movement": "Snap the index and middle fingers down to meet the thumb in one quick motion, like a beak closing.",
        "location": "Neutral space at chest height.",
        "orientation": "Palm faces down/forward; the closing motion is visible to the camera.",
        "framing": "Hand should be close enough to the camera that the finger snap is clearly visible.",
        "common_confusions": "Don't shake your head or wag a single finger. It's a one-shot finger snap.",
    },
    "help": {
        "handshape": "Dominant hand: closed A handshape (fist with thumb up). Non-dominant hand: open flat (B).",
        "movement": "Place the A-hand (thumb up) on top of the flat open palm; lift both hands together as one unit.",
        "location": "Starts in front of the chest; lift to about shoulder height.",
        "orientation": "Flat palm faces up; the A-hand sits on it like an object being lifted.",
        "framing": "Both hands must be visible; don't let the lower hand drift off-screen.",
        "common_confusions": "The two hands move together. Don't lift just the top hand off the palm.",
    },
    "name": {
        "handshape": "H handshape on both hands (index and middle fingers extended together, others closed).",
        "movement": "Tap the fingers of the dominant H-hand on top of the fingers of the non-dominant H-hand twice.",
        "location": "Neutral space at chest height.",
        "orientation": "The hands form a cross/X shape, dominant on top of non-dominant.",
        "framing": "Both hands in the box; the tap should be clearly visible.",
        "common_confusions": "Don't slide the hands — it's two clear taps, not a brush.",
    },
    "nice": {
        "handshape": "Open flat hands (B handshape) on both hands.",
        "movement": "Slide the dominant flat hand across the flat palm of the non-dominant hand, from heel to fingertip.",
        "location": "Neutral space at chest height.",
        "orientation": "Non-dominant palm faces up; dominant palm faces down and slides across.",
        "framing": "Both hands visible; the sliding motion happens horizontally.",
        "common_confusions": "Don't confuse with CLEAN (similar motion). NICE has one clear slide; some teach a single light brush.",
    },
    "meet": {
        "handshape": "Both hands in 1 handshape (index finger up, others closed).",
        "movement": "Start with the two index fingers apart, pointing up. Bring them together so the knuckles meet at the center.",
        "location": "Neutral space at chest height, hands meet at the midline.",
        "orientation": "Index fingers point up; palms face each other.",
        "framing": "Start position: hands shoulder-width apart inside the box.",
        "common_confusions": "Index fingers stay up; don't fold them down. Hands meet, they don't cross.",
    },
    "what": {
        "handshape": "Open 5-hand (all fingers spread), or single open hand.",
        "movement": "Palm up, shake the hand(s) side-to-side slightly while raising eyebrows (facial grammar).",
        "location": "Neutral space at chest-to-waist height.",
        "orientation": "Palm faces up the entire time.",
        "framing": "One hand variant is fine; keep it centered in the box.",
        "common_confusions": "Don't confuse with WHERE (index finger shake, not open hand). WHAT uses the open palm.",
    },
    "how": {
        "handshape": "Both hands curved (bent-B / curved 5), knuckles touching.",
        "movement": "Start with the backs of the curved hands together, knuckles touching. Rotate both hands forward and up so the palms end facing up.",
        "location": "Neutral space at chest height.",
        "orientation": "Starts knuckle-to-knuckle, ends palms-up.",
        "framing": "Both hands together at the midline; rotation happens in front of the chest.",
        "common_confusions": "Single quick rotation, not a continuous shake.",
    },
    "where": {
        "handshape": "1 handshape (index finger up, others closed).",
        "movement": "Shake the index finger side-to-side with small wrist movement.",
        "location": "Neutral space at chest height, slightly to the dominant side.",
        "orientation": "Index points up; palm faces forward (toward the camera).",
        "framing": "Single hand; keep the shake small and contained.",
        "common_confusions": "Don't confuse with WHAT (open palm shake). WHERE keeps the finger pointed up.",
    },
    "who": {
        "handshape": "1 handshape (index finger), or L handshape (thumb + index).",
        "movement": "Index finger makes a small circle in front of the chin, OR the thumb-index L-hand taps the chin.",
        "location": "Near the chin.",
        "orientation": "Index points up; the circle is small and close to the chin.",
        "framing": "Face must be visible; the finger should be close to but not touching the chin.",
        "common_confusions": "Don't confuse with WHY (Y handshape near the forehead).",
    },
    "one": {
        "handshape": "Index finger up (1 handshape), other fingers closed against the palm.",
        "movement": "Static. Hold the handshape clearly for the duration.",
        "location": "Chest height, slightly to the dominant side.",
        "orientation": "Palm faces TOWARD the signer (not the camera); the back of the hand faces the camera.",
        "framing": "One hand, clearly visible, finger pointed straight up.",
        "common_confusions": "Palm faces the SIGNER, not outward. (Palm-out 1 is the letter D or pointing.)",
    },
    "two": {
        "handshape": "Index + middle fingers extended (V shape), thumb closed.",
        "movement": "Static. Hold the handshape steady.",
        "location": "Chest height.",
        "orientation": "Palm faces the signer; back of hand toward the camera.",
        "framing": "Fingers spread clearly in a V; ring and pinky tucked.",
        "common_confusions": "Don't confuse with the letter V (palm out). TWO has palm facing the signer.",
    },
    "three": {
        "handshape": "Thumb, index, and middle fingers extended; ring and pinky closed.",
        "movement": "Static. Hold steady.",
        "location": "Chest height.",
        "orientation": "Palm faces the signer.",
        "framing": "Three fingers clearly spread (thumb + index + middle).",
        "common_confusions": "NOT index + middle + ring (that's a different convention). ASL three is thumb + index + middle.",
    },
    "four": {
        "handshape": "Four fingers extended and slightly spread; thumb tucked across the palm.",
        "movement": "Static. Hold steady.",
        "location": "Chest height.",
        "orientation": "Palm faces the signer.",
        "framing": "Thumb clearly tucked, four fingers up.",
        "common_confusions": "If the thumb sticks out, it reads as FIVE. Tuck it firmly.",
    },
    "five": {
        "handshape": "All five fingers extended and spread (open 5).",
        "movement": "Static. Hold steady.",
        "location": "Chest height.",
        "orientation": "Palm faces the signer.",
        "framing": "Fingers spread, none touching.",
        "common_confusions": "If the thumb is tucked, it reads as FOUR. Spread the thumb wide.",
    },
    "eat": {
        "handshape": "Flat-O (fingers and thumb pinched together as if holding small food).",
        "movement": "Tap the fingertips to the lips twice.",
        "location": "At the mouth.",
        "orientation": "Fingertips toward the mouth; palm faces the signer.",
        "framing": "Face must be visible; hand approaches from below.",
        "common_confusions": "Don't confuse with DRINK (C handshape tilted to mouth). EAT uses pinched fingertips.",
    },
    "sleep": {
        "handshape": "Starts as open 5, ends as flat-O or A.",
        "movement": "Open hand starts at the face palm-in; draws down over the face while closing to a flat-O.",
        "location": "Face — starts at forehead/eyes, ends at the chin.",
        "orientation": "Palm faces the signer the entire time.",
        "framing": "Full face must be visible at the start. Head can tilt slightly.",
        "common_confusions": "One smooth motion top-to-bottom. Don't repeat or open the hand back up.",
    },
    "water": {
        "handshape": "W handshape (index, middle, and ring fingers up; thumb and pinky touching).",
        "movement": "Tap the index finger to the chin twice.",
        "location": "At the chin.",
        "orientation": "Palm faces left (for right-handed signer); the three fingers visible to the camera.",
        "framing": "Chin must be visible; hand approaches from below.",
        "common_confusions": "Initialized sign (the W is for water). Three fingers up, not two.",
    },
    "friend": {
        "handshape": "Both hands in X handshape (index curled into a hook, others closed).",
        "movement": "Hook the dominant index over the non-dominant index. Then unhook, flip both hands, and hook again with the other on top.",
        "location": "Neutral space at chest height.",
        "orientation": "Indexes curl into each other; alternating hand on top.",
        "framing": "Both hands at the midline; the two-hook sequence is one continuous motion.",
        "common_confusions": "Two clear hooks (top-then-flip). Not a single hook.",
    },
    "deaf": {
        "handshape": "1 handshape (index finger extended).",
        "movement": "Touch the index finger near the ear, then near the corner of the mouth (or reverse — both directions are accepted).",
        "location": "Ear and mouth.",
        "orientation": "Index points to the touch points; palm orientation follows naturally.",
        "framing": "Full face/ear in view; don't lean away from the camera.",
        "common_confusions": "Two clear touches. Don't drag the finger continuously across the cheek.",
    },
}


def _maybe_specific(sign_id: str, gloss: str, category: str) -> dict:
    if sign_id in SPECIFIC:
        s = SPECIFIC[sign_id]
        return {
            "sign_id": sign_id,
            "gloss": gloss,
            "handshape": s["handshape"],
            "movement": s["movement"],
            "location": s["location"],
            "orientation": s["orientation"],
            "framing": s["framing"],
            "common_confusions": s["common_confusions"],
        }
    tmpl = TEMPLATES.get(category, TEMPLATES["default"])
    return {
        "sign_id": sign_id,
        "gloss": gloss,
        "handshape": tmpl["handshape"],
        "movement": tmpl["movement"],
        "location": tmpl["location"],
        "orientation": "Palm orientation should match the reference for this sign.",
        "framing": tmpl["framing"],
        "common_confusions": f"If confused with a similar {category} sign, slow down and emphasize handshape.",
    }


def main():
    HINTS_DIR.mkdir(parents=True, exist_ok=True)
    index = {}
    specific_count = 0
    with open(VOCAB, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sign_id = row["sign_id"]
            cat = row.get("category", "default")
            hint = _maybe_specific(sign_id, row["gloss"], cat)
            (HINTS_DIR / f"{sign_id}.json").write_text(
                json.dumps(hint, indent=2), encoding="utf-8"
            )
            index[sign_id] = hint
            if sign_id in SPECIFIC:
                specific_count += 1
    (HINTS_DIR / "_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(
        f"Generated {len(index)} hint files in {HINTS_DIR} "
        f"({specific_count} hand-written, {len(index) - specific_count} templated)"
    )


if __name__ == "__main__":
    main()
