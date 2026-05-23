# Privacy � Camera and Video Handling

## Default pilot behavior

- **Raw video is not uploaded** during normal practice sessions.
- Camera frames are processed **in the browser** in memory for inference only.
- The ONNX model runs locally via ONNX Runtime Web; weights ship with the static bundle.

## Server-stored data

When logged in, the API stores **metadata only**:

- User id (Supabase auth subject)
- Sign id attempted, pass/fail outcome, confidence score, attempt count
- Session timestamps and mastery flags

No frame buffers, blobs, or video files are persisted server-side in the default configuration.

## Dataset capture mode

The optional `/capture` tool saves clips **locally** to the engineer machine via download or local filesystem APIs during collection. Upload to cloud storage is manual and out of band�not enabled in learner practice flow.

## Analytics

Optional lightweight events (retry count, camera errors) contain **no image data**.

## Future data collection

Any proposal to upload learner video for retraining requires:

- Explicit opt-in UI copy
- Separate consent record
- Security review

This is **not implemented** in the pilot build.

## HTTPS

Production deployments must serve the app over HTTPS so camera APIs are available.
