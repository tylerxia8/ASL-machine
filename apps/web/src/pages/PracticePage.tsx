import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { fetchSigns, fetchHint, recordAttempt, trackEvent, type SignMeta } from "../lib/api";
import { requestCamera, captureFramesAsync, framesToTensor, CameraError } from "../lib/camera";

const RECORD_MS = 2000;
import { downsampleForModel } from "../lib/clipFeatures";
import { loadModel, runInference, getLabels, ModelUnavailableError } from "../lib/inference";
import { evaluateAttempt, EvalOutcome } from "../lib/threshold";

type Phase = "prompt" | "recording" | "evaluating" | "result";
type SignReference = { handshape: string; movement: string; location: string };

const CAMERA_HELP: Record<string, string> = {
  denied: "Camera access was denied. Enable camera permission in browser settings and reload.",
  unsupported: "This browser does not support camera access. Use Chrome or Edge on desktop.",
  not_found: "No camera found. Connect a webcam and retry.",
  unknown: "Camera error. Check drivers and close other apps using the camera.",
};

export default function PracticePage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [signs, setSigns] = useState<SignMeta[]>([]);
  const [index, setIndex] = useState(0);
  const [phase, setPhase] = useState<Phase>("prompt");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [outcome, setOutcome] = useState<EvalOutcome | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [predicted, setPredicted] = useState("");
  const [hint, setHint] = useState<string | null>(null);
  const [modelVersion, setModelVersion] = useState("");
  const [modelError, setModelError] = useState<string | null>(null);
  const [reference, setReference] = useState<SignReference | null>(null);
  const [showReference, setShowReference] = useState(false);
  const [sessionLog, setSessionLog] = useState<{ sign: string; outcome: string }[]>([]);
  const sessionId = sessionStorage.getItem("practice_session_id") || undefined;
  const wave = Number(sessionStorage.getItem("practice_wave") || "1");

  const current = signs[index];

  useEffect(() => {
    const saved = sessionStorage.getItem("session_log");
    if (saved) setSessionLog(JSON.parse(saved));
    fetchSigns(wave >= 99 ? undefined : wave).then(setSigns).catch(console.error);
    loadModel()
      .then((m) => {
        setModelVersion(m.version);
        setModelError(null);
      })
      .catch((e: unknown) => {
        setModelVersion("unavailable");
        setModelError(
          e instanceof ModelUnavailableError
            ? e.message
            : "Recognition model failed to load. Refresh to retry."
        );
      });
  }, [wave]);

  useEffect(() => {
    if (!current) return;
    setReference(null);
    setShowReference(false);
    fetchHint(current.sign_id, "fail", userId)
      .then((h) => setReference({ handshape: h.handshape, movement: h.movement, location: h.location }))
      .catch(() => setReference(null));
  }, [current?.sign_id, userId]);

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await requestCamera();
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      trackEvent("camera_ok");
    } catch (e) {
      const code = (e as { code: CameraError }).code || "unknown";
      setCameraError(CAMERA_HELP[code] || CAMERA_HELP.unknown);
      trackEvent("camera_error", { code });
    }
  }, []);

  useEffect(() => {
    startCamera();
    return () => stopCamera();
  }, [startCamera]);

  const evaluate = async () => {
    if (!videoRef.current || !current) return;
    try {
      await loadModel();
    } catch (e) {
      setOutcome("fail");
      setHint(
        e instanceof ModelUnavailableError
          ? e.message
          : "Recognition model failed to load. Refresh to retry."
      );
      setPhase("result");
      trackEvent("inference_error", { error: String(e) });
      return;
    }
    const meta = getLabels();
    const captureSz = 160;
    const captureFrameCount = meta?.num_frames ?? 24;
    const modelT = meta?.num_frames ?? 8;
    const modelSz = meta?.frame_size ?? 32;
    const rawFrames = await captureFramesAsync(
      videoRef.current,
      captureFrameCount,
      captureSz,
      RECORD_MS,
      meta?.preprocess ?? "center_crop"
    );
    setPhase("evaluating");
    const tensor =
      meta?.input_type === "flat"
        ? downsampleForModel(rawFrames, modelT, modelSz, modelSz)
        : framesToTensor(rawFrames, captureFrameCount, captureSz, captureSz);
    try {
      const { predictedLabel, confidence: conf } = await runInference(tensor);
      const result = evaluateAttempt(current.sign_id, predictedLabel, conf);
      setOutcome(result.outcome);
      setConfidence(result.confidence);
      setPredicted(predictedLabel);

      const hintReason =
        result.outcome === "retry" ? "framing" : result.outcome === "fail" ? "fail" : "pass";
      if (result.outcome !== "pass") {
        const h = await fetchHint(current.sign_id, hintReason, userId);
        setHint(h.message);
      } else {
        setHint(null);
      }

      await recordAttempt(
        userId,
        {
          sign_id: current.sign_id,
          outcome: result.outcome,
          confidence: conf,
          predicted_label: predictedLabel,
          session_id: sessionId,
        },
        auth.session?.access_token
      );
      trackEvent("attempt", { sign_id: current.sign_id, outcome: result.outcome, confidence: conf });
      const log = [...sessionLog, { sign: current.sign_id, outcome: result.outcome }];
      setSessionLog(log);
      sessionStorage.setItem("session_log", JSON.stringify(log));
      setPhase("result");
    } catch (err) {
      setOutcome("fail");
      setHint("Model could not run. Ensure model files are synced and reload.");
      setPhase("result");
      trackEvent("inference_error", { error: String(err) });
    }
  };

  const recordAndEvaluate = () => {
    // captureFramesAsync (inside evaluate) waits RECORD_MS on its own;
    // calling evaluate() directly is correct. UI just needs to flip to "recording".
    setPhase("recording");
    void evaluate();
  };

  const nextSign = () => {
    setOutcome(null);
    setHint(null);
    setPhase("prompt");
    if (index + 1 < signs.length) setIndex(index + 1);
    else setIndex(0);
  };

  if (!current && signs.length === 0) {
    return (
      <div className="container">
        <p>Loading signs…</p>
      </div>
    );
  }

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1 style={{ marginTop: "0.5rem" }}>Practice</h1>
      <p>
        Sign {index + 1} of {signs.length}: <strong style={{ fontSize: "1.5rem" }}>{current?.gloss}</strong>
        {current && current.trained === false && (
          <span
            style={{
              marginLeft: "0.5rem",
              padding: "0.1rem 0.45rem",
              borderRadius: "999px",
              background: "var(--muted)",
              color: "white",
              fontSize: "0.7rem",
              verticalAlign: "middle",
            }}
            title="Not yet in the trained model. Use as reference only."
          >
            reference
          </span>
        )}
      </p>

      {cameraError ? (
        <div className="card status-fail">
          <p>{cameraError}</p>
          <button className="btn" onClick={startCamera}>
            Retry camera
          </button>
        </div>
      ) : (
        <div className="video-wrap" style={{ position: "relative" }}>
          <video ref={videoRef} muted playsInline />
          <div className="guide-box" title="Keep hands and face inside box" />
          {phase === "recording" && (
            <>
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  border: "6px solid #e23",
                  pointerEvents: "none",
                  boxSizing: "border-box",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: 8,
                  left: 8,
                  background: "#e23",
                  color: "white",
                  padding: "0.2rem 0.6rem",
                  borderRadius: "4px",
                  fontWeight: 700,
                  fontSize: "0.9rem",
                }}
              >
                ● REC
              </div>
            </>
          )}
        </div>
      )}

      {modelError && (
        <div className="card status-fail" style={{ marginTop: "1rem" }}>
          <strong>Recognition model unavailable</strong>
          <p style={{ margin: "0.25rem 0 0" }}>{modelError}</p>
        </div>
      )}

      <div className="card" style={{ marginTop: "1rem" }}>
        {phase === "prompt" && (
          <>
            {current?.trained === false ? (
              <p style={{ color: "var(--muted)" }}>
                This sign isn't in the trained model yet. Use the reference below to learn it, then skip to the next sign.
              </p>
            ) : (
              <p>When you click Record, perform the sign inside the box. Recording lasts {RECORD_MS / 1000} seconds.</p>
            )}
            <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
              {current?.trained !== false && (
                <button className="btn" disabled={!!cameraError || !!modelError} onClick={recordAndEvaluate}>
                  Record & evaluate
                </button>
              )}
              {reference && (
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => setShowReference((s) => !s)}
                >
                  {showReference ? "Hide reference" : "Show me the sign"}
                </button>
              )}
              {current?.trained === false && (
                <button className="btn" type="button" onClick={nextSign}>
                  Next sign
                </button>
              )}
            </div>
            {showReference && reference && (
              <div className="hint-panel" style={{ marginTop: "0.75rem" }}>
                <strong>Reference</strong>
                <p style={{ margin: "0.25rem 0 0" }}>
                  Handshape: {reference.handshape}<br />
                  Movement: {reference.movement}<br />
                  Location: {reference.location}
                </p>
              </div>
            )}
          </>
        )}
        {phase === "recording" && <p>Recording… hold your sign.</p>}
        {phase === "evaluating" && <p>Evaluating locally…</p>}
        {phase === "result" && outcome && (
          <>
            <p className={`status-${outcome}`} style={{ fontSize: "1.25rem", fontWeight: 600 }}>
              {outcome === "pass" ? "Pass" : outcome === "retry" ? "Try again (uncertain)" : "Fail"}
              {outcome !== "pass" && ` — ${(confidence * 100).toFixed(0)}% confidence`}
            </p>
            {predicted && outcome !== "pass" && (
              <p style={{ color: "var(--muted)" }}>Detected: {predicted}</p>
            )}
            {hint && (
              <div className="hint-panel">
                <strong>Hint</strong>
                <p>{hint}</p>
              </div>
            )}
            <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
              {outcome !== "pass" && (
                <button className="btn" onClick={() => { setPhase("prompt"); setOutcome(null); setHint(null); }}>
                  Retry
                </button>
              )}
              <button className="btn btn-secondary" onClick={nextSign}>
                {outcome === "pass" ? "Next sign" : "Skip to next"}
              </button>
            </div>
          </>
        )}
      </div>
      {sessionLog.length > 0 && (
        <div className="card" style={{ marginTop: "1rem", fontSize: "0.85rem" }}>
          <strong>Session log</strong>
          <ul style={{ margin: "0.25rem 0", paddingLeft: "1.2rem" }}>
            {sessionLog.slice(-8).map((e, i) => (
              <li key={i}>
                {e.sign}: <span className={`status-${e.outcome}`}>{e.outcome}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="footer-meta">Model: {modelVersion} · Inference runs locally · No video upload</p>
    </div>
  );
}
