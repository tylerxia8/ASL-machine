import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { fetchSigns, fetchHint, type SignMeta, type HintResponse } from "../lib/api";
import { requestCamera, recordVideo, CameraError } from "../lib/camera";
import { buildLearningPriorities } from "../lib/learningPlan";
import { loadRecognitionCalibration, type RecognitionCalibration } from "../lib/recognitionCalibration";
import { readRecognitionFeedback, summarizeRecognitionFeedback } from "../lib/recognitionFeedback";

const TARGET_PER_SIGN = 30;
const MIN_PER_SIGN = 10;
const RECORD_MS = 2000;
const COUNTDOWN_MS = 3000;
const INCOMING_REL = "ml\\data\\incoming";

type Phase = "idle" | "countdown" | "recording" | "saving";
type CaptureOrder = "weak_first" | "fewest_clips" | "default";

const PHASE_LABELS: Record<Phase, string> = {
  idle: `Record ${RECORD_MS / 1000}s clip`,
  countdown: "Get ready…",
  recording: "RECORDING",
  saving: "Saving…",
};

const CAMERA_HELP: Record<string, string> = {
  denied: "Camera access was denied. Enable camera permission in browser settings and reload.",
  unsupported: "This browser does not support camera access. Use Chrome or Edge on desktop.",
  not_found: "No camera found. Connect a webcam and retry.",
  unknown: "Camera error. Check drivers and close other apps using the camera.",
};

export default function CapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [signerId, setSignerId] = useState("signer_a");
  const [signs, setSigns] = useState<SignMeta[]>([]);
  const [index, setIndex] = useState(0);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [status, setStatus] = useState("");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [countdown, setCountdown] = useState(3);
  const lastDownloadUrlRef = useRef<string | null>(null);
  const [lastClip, setLastClip] = useState<{ filename: string; url: string; sizeKB: number } | null>(null);
  const [reference, setReference] = useState<HintResponse | null>(null);
  const [calibration, setCalibration] = useState<RecognitionCalibration | null>(null);
  const [captureOrder, setCaptureOrder] = useState<CaptureOrder>("weak_first");

  const feedbackSummary = useMemo(() => summarizeRecognitionFeedback(readRecognitionFeedback()), []);

  const orderedSigns = useMemo(() => {
    if (captureOrder === "default") return signs;
    if (captureOrder === "fewest_clips") {
      return [...signs].sort(
        (a, b) => (counts[a.sign_id] ?? 0) - (counts[b.sign_id] ?? 0) || a.gloss.localeCompare(b.gloss)
      );
    }
    return buildLearningPriorities(signs, calibration, feedbackSummary, counts).map((p) => p.sign);
  }, [calibration, captureOrder, counts, feedbackSummary.bySign, signs]);

  const current = orderedSigns[index];

  useEffect(() => {
    if (!current) return;
    setReference(null);
    fetchHint(current.sign_id, "fail", "capture-anon")
      .then(setReference)
      .catch(() => setReference(null));
  }, [current?.sign_id]);

  useEffect(() => {
    fetchSigns(1)
      .then((s) => setSigns(s))
      .catch(() => setStatus("Failed to load trained sign list from API."));
    loadRecognitionCalibration()
      .then(setCalibration)
      .catch(() => setCalibration(null));
  }, []);

  useEffect(() => {
    const raw = localStorage.getItem(`capture_counts_${signerId}`);
    setCounts(raw ? JSON.parse(raw) : {});
  }, [signerId]);

  useEffect(() => {
    requestCamera()
      .then((stream) => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // play() can reject under autoplay policies; surface it instead of
          // letting it disappear into an unhandled-promise-rejection.
          videoRef.current.play().catch((err) => {
            setCameraError(
              `Browser blocked autoplay: ${err?.name ?? "unknown"}. ` +
                "Click anywhere on the page, then refresh."
            );
          });
        }
      })
      .catch((e) => {
        const code = (e as { code: CameraError }).code || "unknown";
        setCameraError(CAMERA_HELP[code] || CAMERA_HELP.unknown);
      });
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (lastDownloadUrlRef.current) URL.revokeObjectURL(lastDownloadUrlRef.current);
    };
  }, []);

  const progress = useMemo(() => {
    const done = signs.filter((s) => (counts[s.sign_id] ?? 0) >= TARGET_PER_SIGN).length;
    const minDone = signs.filter((s) => (counts[s.sign_id] ?? 0) >= MIN_PER_SIGN).length;
    const totalClips = Object.values(counts).reduce((a, b) => a + b, 0);
    return { done, minDone, total: signs.length, totalClips };
  }, [counts, signs]);

  const startRecord = async () => {
    if (!current || !streamRef.current) return;
    setStatus("");
    setPhase("countdown");
    for (let n = 3; n >= 1; n--) {
      setCountdown(n);
      await new Promise((r) => setTimeout(r, COUNTDOWN_MS / 3));
    }
    setPhase("recording");
    let blob: Blob;
    try {
      blob = await recordVideo(streamRef.current, RECORD_MS);
    } catch (e) {
      setPhase("idle");
      setStatus((e as Error).message);
      return;
    }
    setPhase("saving");
    const filename = `${current.sign_id}_${signerId}_${Date.now()}.webm`;
    const url = URL.createObjectURL(blob);
    if (lastDownloadUrlRef.current) URL.revokeObjectURL(lastDownloadUrlRef.current);
    lastDownloadUrlRef.current = url;
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    setLastClip({ filename, url, sizeKB: blob.size / 1024 });

    const nextCount = (counts[current.sign_id] ?? 0) + 1;
    const updated = { ...counts, [current.sign_id]: nextCount };
    setCounts(updated);
    localStorage.setItem(`capture_counts_${signerId}`, JSON.stringify(updated));
    setStatus(`Saved ${filename} (${(blob.size / 1024).toFixed(0)} KB) to Downloads. Review below.`);
    setPhase("idle");

    if (nextCount >= TARGET_PER_SIGN && index < orderedSigns.length - 1) {
      setIndex(index + 1);
    }
  };

  const undoLast = () => {
    if (!current) return;
    const c = counts[current.sign_id] ?? 0;
    if (c <= 0) return;
    const updated = { ...counts, [current.sign_id]: c - 1 };
    setCounts(updated);
    localStorage.setItem(`capture_counts_${signerId}`, JSON.stringify(updated));
    const filename = lastClip?.filename;
    setStatus(
      `Decremented count for ${current.gloss}. ` +
        (filename
          ? `Delete ${filename} from your Downloads folder if you want to discard it.`
          : "The file is still in Downloads; delete it manually if you want to discard.")
    );
    setLastClip(null);
  };

  const recording = phase === "recording";
  const countingDown = phase === "countdown";
  const busy = recording || countingDown || phase === "saving";
  const currentThreshold = current ? calibration?.thresholds?.[current.sign_id] : null;
  const currentFeedback = current ? feedbackSummary.bySign[current.sign_id] : null;

  if (signs.length === 0) {
    return (
      <div className="container">
        <Link to="/lobby">← Lobby</Link>
        <h1>Wave 1 Dataset Capture</h1>
        <p>{status || "Loading trained sign list…"}</p>
      </div>
    );
  }

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1>Wave 1 Dataset Capture</h1>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <strong>After recording</strong>
        <ol style={{ margin: "0.5rem 0", paddingLeft: "1.25rem", color: "var(--muted)" }}>
          <li>Move <code>.webm</code> files from Downloads to <code>{INCOMING_REL}</code></li>
          <li>Run <code>scripts\continue-wave1.ps1</code> to import + retrain</li>
        </ol>
      </div>

      <p style={{ color: "var(--muted)" }}>
        Target: {TARGET_PER_SIGN}/sign · minimum {MIN_PER_SIGN}/sign to retrain · {RECORD_MS / 1000}s per clip.
      </p>

      <div className="card" style={{ marginBottom: "1rem", display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
        <label htmlFor="signer-select"><strong>Signer ID:</strong></label>
        <select
          id="signer-select"
          className="input"
          style={{ width: "12rem", marginBottom: 0 }}
          value={signerId}
          onChange={(e) => setSignerId(e.target.value)}
          disabled={busy}
        >
          <option value="signer_a">signer_a</option>
          <option value="signer_b">signer_b</option>
          <option value="signer_c">signer_c</option>
        </select>
        <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          <strong>{progress.totalClips}</strong> clips · {" "}
          <strong>{progress.minDone}</strong>/{progress.total} at {MIN_PER_SIGN}+ · {" "}
          <strong>{progress.done}</strong>/{progress.total} at {TARGET_PER_SIGN}+
        </span>
      </div>

      <div className="card" style={{ marginBottom: "1rem", display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
        <label htmlFor="capture-order">
          <strong>Capture order:</strong>
        </label>
        <select
          id="capture-order"
          className="input"
          style={{ width: "13rem", marginBottom: 0 }}
          value={captureOrder}
          onChange={(e) => {
            setCaptureOrder(e.target.value as CaptureOrder);
            setIndex(0);
          }}
          disabled={busy}
        >
          <option value="weak_first">Weak signs first</option>
          <option value="fewest_clips">Fewest clips first</option>
          <option value="default">Default sign order</option>
        </select>
        <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>
          Weak-sign order uses release metrics, local feedback, and current clip counts.
        </span>
      </div>

      {cameraError ? (
        <div className="card status-fail">
          <p>{cameraError}</p>
        </div>
      ) : (
        <div className="video-wrap" style={{ position: "relative" }}>
          <video ref={videoRef} muted playsInline />
          <div className="guide-box" />
          {countingDown && (
            <div
              style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "6rem",
                fontWeight: 700,
                color: "white",
                textShadow: "0 0 10px rgba(0,0,0,0.7)",
              }}
            >
              {countdown}
            </div>
          )}
          {recording && (
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

      <div className="card" style={{ marginTop: "1rem" }}>
        <p style={{ fontSize: "1.25rem", margin: 0 }}>
          Sign {index + 1}/{orderedSigns.length}: <strong>{current.gloss}</strong>{" "}
          <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>({current.sign_id})</span>
        </p>
        <p style={{ marginTop: "0.25rem" }}>
          Clips this sign: <strong>{counts[current.sign_id] ?? 0}</strong> / {TARGET_PER_SIGN}
        </p>
        {(currentThreshold || currentFeedback) && (
          <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginTop: "0.25rem" }}>
            {currentThreshold && (
              <>
                Model F1 {(((currentThreshold.f1 ?? 0) * 100)).toFixed(0)}% with{" "}
                {currentThreshold.support ?? 0} test clips.
              </>
            )}{" "}
            {currentFeedback && (
              <>
                Learner feedback: {currentFeedback.rejected}/{currentFeedback.total} marked wrong.
              </>
            )}
          </p>
        )}
        {reference && (
          <div className="hint-panel" style={{ marginTop: "0.5rem" }}>
            <strong>Reference (perform this exact form)</strong>
            <p style={{ margin: "0.25rem 0 0", fontSize: "0.9rem" }}>
              <strong>Handshape:</strong> {reference.handshape}
              <br />
              <strong>Movement:</strong> {reference.movement}
              <br />
              <strong>Location:</strong> {reference.location}
              <br />
              <strong>Framing:</strong> {reference.framing}
            </p>
          </div>
        )}
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
          <button className="btn" onClick={startRecord} disabled={busy || !!cameraError}>
            {PHASE_LABELS[phase]}
          </button>
          <button className="btn btn-secondary" onClick={undoLast} disabled={busy || (counts[current.sign_id] ?? 0) === 0}>
            Undo last
          </button>
          <button className="btn btn-secondary" disabled={index === 0 || busy} onClick={() => setIndex(index - 1)}>
            Previous sign
          </button>
          <button
            className="btn btn-secondary"
            disabled={index >= orderedSigns.length - 1 || busy}
            onClick={() => setIndex(index + 1)}
          >
            Next sign
          </button>
        </div>
        {status && <p style={{ marginTop: "0.75rem", color: "var(--muted)", fontSize: "0.9rem" }}>{status}</p>}
      </div>

      {/* Clip review: shown after a successful save. Lets the signer verify
          the take before committing to the next one. */}
      {lastClip && !busy && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <strong>Last clip review</strong>
          <p style={{ color: "var(--muted)", fontSize: "0.85rem", margin: "0.25rem 0" }}>
            {lastClip.filename} · {lastClip.sizeKB.toFixed(0)} KB
          </p>
          <video
            src={lastClip.url}
            controls
            playsInline
            style={{ width: "100%", maxWidth: "320px", borderRadius: "8px", background: "#000" }}
          />
          <p style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--muted)" }}>
            Looks good? Hit <strong>Record</strong> again. Bad take? <strong>Undo last</strong> decrements the count;
            delete the file from your Downloads folder.
          </p>
        </div>
      )}
    </div>
  );
}
