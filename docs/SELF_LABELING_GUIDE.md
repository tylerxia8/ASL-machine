# Self-Labeling Guide (No Instructor)

The capture page shows handshape/movement/location references for each sign. This doc is for the recording mindset and quality control around those references.

## Before recording

1. Open the capture page; navigate to the target sign. The **Reference** panel renders the canonical form (handshape / movement / location / orientation / framing).
2. Cross-check the reference against a trusted external source (a deaf-instructor video on the web, a curriculum video, or a published ASL textbook). If they disagree, **the external source wins** — update the hint JSON in [`content/hints/`](../content/hints/) before recording dozens of clips with a wrong reference.
3. Practice the sign 3–5 times until it feels natural inside the 2-second window. Many ASL 1 signs are 1–1.5s; you don't need to drag them out.
4. Set up lighting and framing per [`CONTROLLED_CONDITIONS.md`](CONTROLLED_CONDITIONS.md).

## Per clip

- Both hands and face in the guide box throughout.
- Start from a relaxed neutral position, perform the sign **once** during the 2s window, return to neutral.
- The countdown gives you 3 seconds to get into starting position — use it.
- Discard takes where you self-corrected mid-clip, blinked at the wrong moment, or had your hands cut off.

## Volume targets

| Phase | Trained signs | Clips/sign/signer | Signers | Total |
|-------|---------------|-------------------|---------|-------|
| Shakedown (validates pipeline) | 5 | 5 | 1 | 25 |
| Minimum retrain | 25 | 10 | 2 | 500 |
| Wave 1 pilot target | 25 | 30 | 2 | 1,500 |
| Stretch (signer-disjoint test) | 25 | 30 | 3 | 2,250 |

## Signer metadata

The capture page's **Signer ID** dropdown drives the filename (`{sign_id}_{signer_id}_{ts}.webm`) and the train/test split. With ≥3 signers, [`build_manifest.py --signer-disjoint`](../ml/scripts/build_manifest.py) holds one entire signer out for the test set — this is the rubric-honest accuracy number.

Be **strict** about consistency: don't record under `signer_a` on Monday and `signer_b` on Tuesday for the same person. The model will learn signer-specific quirks and a held-out signer will appear to "fail" because they look subtly different.

## Quality checks during the session

- Every 30 clips, **play back one randomly** (find it in Downloads, double-click — Chrome plays webm natively). Confirm the sign is performed correctly and the face/hands are visible.
- If a sign repeatedly confuses the model after training, document the failure mode in the **Known limitations** section of [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md). Don't silently drop the sign — it stays in the trained set but the limitation is on the record.

## Resolving disagreements with the hint file

The hint files in [`content/hints/`](../content/hints/) drive both the capture-page reference and the practice-page failure hints. If you change your interpretation of a sign mid-session, **edit the hint file first**, then re-record from scratch — don't mix two different forms of the same sign in the training data.

To regenerate hints after editing the source dict in [`scripts/generate_hints.py`](../scripts/generate_hints.py):

```powershell
python scripts/generate_hints.py
```
