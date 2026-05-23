# Controlled Pilot Conditions

Pilot recognition quality is valid only under the conditions below. Capture sessions and learner sessions that deviate should be flagged in the validation report.

## Browser and device

- **Browsers:** Chrome 120+ or Edge 120+ (Chromium). MediaRecorder webm capture is verified on these.
- **Device:** Laptop or desktop with built-in or external webcam.
- **Camera:** Capture requested at 640×480 (the browser may negotiate higher). Training input is resized to 160×160 center crop.

## Environment

- **Lighting:** Indoor, face and hands evenly lit. Avoid:
  - Strong backlight (bright window behind signer)
  - Single overhead light only (creates harsh shadows under the chin and hands)
  - Mixed fluorescent + window (color casts that vary across the clip)
- **Background:** Plain matte wall preferred. Avoid:
  - Other people walking through the frame
  - Animated screens behind the signer
  - High-contrast patterns (bookshelves, tiles)
- **Distance:** Signer 2–3 feet from the camera. Test framing before recording: head to mid-chest should fill roughly the upper half of the camera view.
- **Framing:** Both hands must stay inside the on-screen guide box during the full 2-second recording. The signer's face should remain fully visible (the model uses face/body context).
- **Camera angle:** Camera should sit at eye-to-shoulder height. Avoid extreme low angles (laptop-on-lap).

## Capture protocol (per clip)

- **Clip duration:** 2 seconds (fixed by the recorder).
- **Performance:** Begin in a neutral resting position. The 3-second countdown gives you time to bring your hands up. Perform the sign at natural pace once during the 2-second window. End in neutral position.
- **Hands:** Hold steady for ~0.25s at the start and end (this prevents motion blur at the edges and gives the model a stable reference).
- **No talking or mouthing English:** Keep face neutral or use ASL-appropriate facial grammar (e.g., raised eyebrows for "what?").
- **One sign per clip:** Do not chain signs or self-correct mid-clip. If you flub, use **Undo last** and re-record.

## Signer diversity (training data)

- **Minimum:** 2 distinct signers (`signer_a`, `signer_b`).
- **Recommended:** 3 signers if available, with one entire signer held out for the test split (signer-disjoint evaluation in `ml/scripts/build_manifest.py --signer-disjoint`).
- **Per signer:** 30 clips per sign minimum. Vary slightly between takes (small differences in hand position) so the model doesn't memorize exact pixel patterns.
- **Per session:** Try to vary lighting/background slightly across sessions (different time of day, different room corner) to broaden the data distribution.

## Practice protocol (per attempt)

- Same browser, distance, framing, and lighting requirements as capture.
- The practice page records on the same 2-second window so practice conditions match training conditions by construction.
- Confidence thresholds (see [VALIDATION_REPORT.md](VALIDATION_REPORT.md)):
  - **Pass:** correct top-1 class with confidence ≥ 0.90
  - **Retry (uncertain):** confidence between 0.70 and 0.90 — model declines to commit
  - **Fail:** wrong class or confidence < 0.70

## Model scope

- **Trained vocabulary:** 25 signs (see [content/wave1_signs.csv](../content/wave1_signs.csv)).
- **Reference-only vocabulary:** the remaining 78 signs in [content/vocabulary.csv](../content/vocabulary.csv) are shown for learner study but are NOT evaluated by the model. They render as `reference` in the UI and skip the recognition step.
- **Not supported:** fingerspelling sequences, role shift, non-manual grammar beyond simple wh-question raises, conversational ASL, multi-sign phrases, BSL or other signed languages.

## Known limitations

- Self-collected data with two signers will not generalize to all skin tones, hand sizes, or signing styles.
- Visually-similar signs in the trained set may confuse the model:
  - **please / sorry** (both circle on chest; different handshape)
  - **what / where** (both shake; different handshape)
  - **four / five** (only the thumb position differs)
  - **eat / drink** (drink is not in the trained set, but learners may attempt it)
- Performance degrades with: backlight, low light, hands cropped by the guide box, very fast or partial sign execution.
- No pretrained models are used (Requirement 7); accuracy on signer-disjoint data is bounded by self-collected dataset size.
