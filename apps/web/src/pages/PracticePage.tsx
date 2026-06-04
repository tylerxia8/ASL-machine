import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import ReferenceVideo from "../components/ReferenceVideo";
import { fetchSigns, fetchHint, recordAttempt, trackEvent, type SignMeta } from "../lib/api";
import {
  attachCameraStream,
  requestCamera,
  captureFramesAsync,
  framesToTensor,
  recordVideo,
  streamIsLive,
  waitForVideoReady,
  CameraError,
} from "../lib/camera";
import { captureHandLandmarkWindows, getHandTrackingRatio } from "../lib/handLandmarks";
import { downsampleForModel } from "../lib/clipFeatures";
import { loadModel, runInference, runInferenceBatch, getLabels, ModelUnavailableError } from "../lib/inference";
import { confusionHint, loadRecognitionCalibration, reliabilityFor, thresholdsFor, type RecognitionCalibration } from "../lib/recognitionCalibration";
import { buildConfusionDrillSigns, buildLearningPriorities } from "../lib/learningPlan";
import { personalizeThresholds } from "../lib/personalCalibration";
import { readRecognitionFeedback, summarizeRecognitionFeedback } from "../lib/recognitionFeedback";
import { evaluateAttempt, EvalOutcome } from "../lib/threshold";

type Phase = "prompt" | "recording" | "selfCheck" | "evaluating" | "result";
type PracticeMode = "guided" | "recognition";
type PracticeOrder = "weak_first" | "confusions" | "default" | "shuffle";
type SignReference = { handshape: string; movement: string; location: string };
type CorrectionClip = {
  filename: string;
  sign_id: string;
  predicted_label: string;
  confidence: number;
  agreement?: number;
  tracking_ratio?: number | null;
  captured_at: string;
};

const RECORD_MS = 2000;
const MIN_HAND_TRACKING_RATIO = 0.35;
const MULTI_WINDOW_CAPTURE_MS = 2400;
const MIN_WINDOW_AGREEMENT_FOR_PASS = 0.67;
const CORRECTION_RECORD_MS = 2200;
const CORRECTION_MANIFEST_KEY = "recognition_correction_clips";
const PRACTICE_MODE_KEY = "practice_mode";
const PRACTICE_ORDER_KEY = "practice_order";
const PRACTICE_UNIT_FILTER_KEY = "practice_unit_filter";
const WATCHLIST_SELF_CHECK_HINT =
  "This sign is on the model watchlist, so recognition asks for your self-check instead of auto-passing it.";

function readPracticeMode(): PracticeMode {
  try {
    return localStorage.getItem(PRACTICE_MODE_KEY) === "recognition" ? "recognition" : "guided";
  } catch {
    return "guided";
  }
}

function readPracticeOrder(): PracticeOrder {
  try {
    const value = localStorage.getItem(PRACTICE_ORDER_KEY);
    return value === "default" || value === "shuffle" || value === "confusions" ? value : "weak_first";
  } catch {
    return "weak_first";
  }
}

function stableShuffle(items: SignMeta[]) {
  const rank = (signId: string) => {
    let hash = 0;
    for (const char of signId) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    return Math.sin(hash);
  };
  return [...items].sort((a, b) => {
    return rank(a.sign_id) - rank(b.sign_id);
  });
}

function outcomeLabel(outcome: string) {
  if (outcome === "pass") return "matched";
  if (outcome === "retry") return "review";
  return "try again";
}

function liveTrackingLabel(ratio: number | null) {
  if (ratio === null) return "tracking...";
  if (ratio >= 0.75) return "hands visible";
  if (ratio >= MIN_HAND_TRACKING_RATIO) return "tracking okay";
  return "move into frame";
}

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
  const [cameraNeedsStart, setCameraNeedsStart] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [outcome, setOutcome] = useState<EvalOutcome | null>(null);
  const [practiceMode, setPracticeMode] = useState<PracticeMode>(readPracticeMode);
  const [practiceOrder, setPracticeOrder] = useState<PracticeOrder>(readPracticeOrder);
  const [confidence, setConfidence] = useState(0);
  const [predicted, setPredicted] = useState("");
  const [topPredictions, setTopPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [windowPredictions, setWindowPredictions] = useState<{ label: string; confidence: number }[]>([]);
  const [windowAgreement, setWindowAgreement] = useState<number | null>(null);
  const [routeInfo, setRouteInfo] = useState<{
    routedBy: string;
    primaryLabel: string;
    primaryConfidence: number;
    specialistLabel: string;
    specialistConfidence: number;
  } | null>(null);
  const [trackingRatio, setTrackingRatio] = useState<number | null>(null);
  const [liveTrackingRatio, setLiveTrackingRatio] = useState<number | null>(null);
  const [calibration, setCalibration] = useState<RecognitionCalibration | null>(null);
  const [recognitionFeedbackSaved, setRecognitionFeedbackSaved] = useState(false);
  const [hint, setHint] = useState<string | null>(null);
  const [modelVersion, setModelVersion] = useState("");
  const [modelError, setModelError] = useState<string | null>(null);
  const [reference, setReference] = useState<SignReference | null>(null);
  const [showReference, setShowReference] = useState(false);
  const [sessionLog, setSessionLog] = useState<{ sign: string; outcome: string }[]>([]);
  const [recognitionFeedback, setRecognitionFeedback] = useState(readRecognitionFeedback);
  const [correctionStatus, setCorrectionStatus] = useState("");
  const [correctionClips, setCorrectionClips] = useState<CorrectionClip[]>([]);
  const [selfCheckUrl, setSelfCheckUrl] = useState<string | null>(null);
  const sessionId = sessionStorage.getItem("practice_session_id") || undefined;
  const wave = Number(sessionStorage.getItem("practice_wave") || "1");
  const unitFilter = useMemo(() => {
    try {
      return (localStorage.getItem(PRACTICE_UNIT_FILTER_KEY) || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
    } catch {
      return [];
    }
  }, []);

  const feedbackSummary = useMemo(() => summarizeRecognitionFeedback(recognitionFeedback), [recognitionFeedback]);
  const orderedSigns = useMemo(() => {
    const filteredSigns = unitFilter.length > 0
      ? signs.filter((sign) => unitFilter.includes(sign.sign_id))
      : signs;
    if (practiceOrder === "default") return filteredSigns;
    if (practiceOrder === "shuffle") return stableShuffle(filteredSigns);
    if (practiceOrder === "confusions") return buildConfusionDrillSigns(filteredSigns, calibration);
    return buildLearningPriorities(filteredSigns, calibration, feedbackSummary).map((p) => p.sign);
  }, [calibration, feedbackSummary, practiceOrder, signs, unitFilter]);
  const current = orderedSigns[index];
  const currentReliability = current ? reliabilityFor(calibration?.thresholds?.[current.sign_id]) : "watch";
  const currentFeedback = current ? feedbackSummary.bySign[current.sign_id] : null;
  const currentThresholds = current ? thresholdsFor(calibration, current.sign_id) : null;
  const currentConfusions = useMemo(() => {
    if (!current) return [];
    return Object.entries(calibration?.confusions ?? {})
      .map(([pair, row]) => {
        const [prompt, confusedWith] = pair.split("->");
        return { prompt, confusedWith, count: row.count ?? 0, message: row.message };
      })
      .filter((row) => row.prompt === current.sign_id)
      .sort((a, b) => b.count - a.count)
      .slice(0, 2);
  }, [calibration, current]);

  useEffect(() => {
    const saved = sessionStorage.getItem("session_log");
    if (saved) setSessionLog(JSON.parse(saved));
    try {
      setCorrectionClips(JSON.parse(localStorage.getItem(CORRECTION_MANIFEST_KEY) || "[]"));
    } catch {
      setCorrectionClips([]);
    }
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
    loadRecognitionCalibration().then(setCalibration).catch(() => setCalibration(null));
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
    setCameraReady(false);
    if (selfCheckUrl) URL.revokeObjectURL(selfCheckUrl);
  };

  const resetRecognitionDetails = () => {
    setTopPredictions([]);
    setWindowPredictions([]);
    setWindowAgreement(null);
    setRouteInfo(null);
    setTrackingRatio(null);
    setRecognitionFeedbackSaved(false);
  };

  const clearSelfCheckClip = () => {
    if (selfCheckUrl) URL.revokeObjectURL(selfCheckUrl);
    setSelfCheckUrl(null);
  };

  const startCamera = useCallback(async () => {
    setCameraError(null);
    setCameraNeedsStart(false);
    setCameraReady(false);
    try {
      const stream = streamRef.current ?? await requestCamera();
      streamRef.current = stream;
      if (videoRef.current) {
        try {
          await attachCameraStream(videoRef.current, stream);
          setCameraReady(true);
        } catch (err) {
          setCameraNeedsStart(true);
          setCameraError(null);
          return;
        }
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

  useEffect(() => {
    if (practiceMode !== "recognition" || cameraError || phase === "recording" || phase === "evaluating") {
      setLiveTrackingRatio(null);
      return;
    }
    let stopped = false;
    let timer: number | undefined;
    const tick = async () => {
      if (!videoRef.current || stopped) return;
      try {
        const ratio = await getHandTrackingRatio(videoRef.current);
        if (!stopped) setLiveTrackingRatio(ratio);
      } catch {
        if (!stopped) setLiveTrackingRatio(null);
      }
      if (!stopped) timer = window.setTimeout(tick, 700);
    };
    timer = window.setTimeout(tick, 250);
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [cameraError, phase, practiceMode]);

  const saveOutcome = async (
    result: EvalOutcome,
    conf: number | undefined,
    predictedLabel: string,
    source: "self_check" | "model",
    routeDetails = routeInfo
  ) => {
    if (!current) return;
    await recordAttempt(
      userId,
      {
        sign_id: current.sign_id,
        outcome: result,
        confidence: conf,
        predicted_label: predictedLabel,
        session_id: sessionId,
      },
      auth.session?.access_token
    );
    trackEvent("attempt", {
      sign_id: current.sign_id,
      outcome: result,
      confidence: conf ?? null,
      source,
      routed_by: routeDetails?.routedBy,
    });
    const log = [...sessionLog, { sign: current.sign_id, outcome: result }];
    setSessionLog(log);
    sessionStorage.setItem("session_log", JSON.stringify(log));
  };

  const updatePracticeMode = (mode: PracticeMode) => {
    try {
      localStorage.setItem(PRACTICE_MODE_KEY, mode);
    } catch {
      // Storage can be blocked in hardened/private browser contexts.
    }
    setPracticeMode(mode);
    setPhase("prompt");
    setOutcome(null);
    setHint(null);
    setPredicted("");
    setCorrectionStatus("");
    resetRecognitionDetails();
    setConfidence(0);
  };

  const updatePracticeOrder = (order: PracticeOrder) => {
    try {
      localStorage.setItem(PRACTICE_ORDER_KEY, order);
    } catch {
      // Storage can be blocked in hardened/private browser contexts.
    }
    setPracticeOrder(order);
    setIndex(0);
    setPhase("prompt");
    setOutcome(null);
    setHint(null);
    setCorrectionStatus("");
    resetRecognitionDetails();
  };

  const startSelfCheck = async () => {
    if (!streamIsLive(streamRef.current)) {
      await startCamera();
    }
    if (!streamIsLive(streamRef.current)) {
      setHint("Camera preview is not active yet. Click Start camera preview, allow permission, then try again.");
      return;
    }
    const stream = streamRef.current;
    if (!stream) return;
    setPhase("recording");
    clearSelfCheckClip();
    try {
      const blob = await recordVideo(stream, RECORD_MS);
      setSelfCheckUrl(URL.createObjectURL(blob));
      setOutcome(null);
      setConfidence(0);
      setPredicted("");
      resetRecognitionDetails();
      setHint(null);
      setCorrectionStatus("");
      setPhase("selfCheck");
    } catch (err) {
      setOutcome("retry");
      setHint(`Recording failed: ${(err as Error).message}`);
      setPhase("result");
    }
  };

  const completeSelfCheck = async (result: EvalOutcome) => {
    if (!current) return;
    setOutcome(result);
    setConfidence(0);
    setPredicted("self_check");
    if (result !== "pass") {
      try {
        const h = await fetchHint(current.sign_id, "framing", userId);
        setHint(h.message);
      } catch {
        setHint("Review the reference, then try the sign again slowly inside the guide box.");
      }
    } else {
      setHint(null);
    }
    try {
      await saveOutcome(result, undefined, "self_check", "self_check");
    } catch (err) {
      trackEvent("attempt_record_error", { error: String(err), source: "self_check" });
    }
    setPhase("result");
  };

  const evaluate = async () => {
    if (!videoRef.current || !current) return;
    if (!streamIsLive(streamRef.current) || !cameraReady) {
      await startCamera();
      if (!videoRef.current || !streamIsLive(streamRef.current)) {
        setOutcome("retry");
        setHint("Camera preview is not active yet. Click Start camera preview, allow permission, then try again.");
        setPhase("result");
        return;
      }
      try {
        await waitForVideoReady(videoRef.current);
        setCameraReady(true);
      } catch {
        setOutcome("retry");
        setHint("Camera opened, but the video preview is not visible yet. Wait a moment, then try again.");
        setPhase("result");
        return;
      }
    }
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
    try {
      let modelInput: Float32Array;
      let modelResult: Awaited<ReturnType<typeof runInference>> | null = null;
      if (meta?.input_type === "hand_landmarks") {
        const samples = await captureHandLandmarkWindows(videoRef.current, 3, captureFrameCount, MULTI_WINDOW_CAPTURE_MS);
        const bestSample = samples.reduce((best, sample) =>
          sample.detectedFrameRatio > best.detectedFrameRatio ? sample : best
        );
        modelInput = bestSample.tensor;
        const averageTracking = samples.reduce((sum, sample) => sum + sample.detectedFrameRatio, 0) / samples.length;
        setTrackingRatio(averageTracking);
        if (averageTracking < MIN_HAND_TRACKING_RATIO) {
          const ratioPct = Math.round(averageTracking * 100);
          setOutcome("fail");
          setConfidence(0);
          setPredicted("low_tracking");
          resetRecognitionDetails();
          setTrackingRatio(averageTracking);
          setHint(
            `I only tracked hands in ${ratioPct}% of the capture. Move closer, keep both hands inside the guide box, and try again with brighter lighting.`
          );
          try {
            await saveOutcome("fail", undefined, "low_tracking", "model");
          } catch (err) {
            trackEvent("attempt_record_error", { error: String(err), source: "model" });
          }
          setPhase("result");
          return;
        }
        setPhase("evaluating");
        modelResult = await runInferenceBatch(samples.map((sample) => sample.tensor));
      } else {
        const rawFrames = await captureFramesAsync(
          videoRef.current,
          captureFrameCount,
          captureSz,
          RECORD_MS,
          meta?.preprocess ?? "center_crop"
        );
        modelInput =
          meta?.input_type === "flat"
            ? downsampleForModel(rawFrames, modelT, modelSz, modelSz)
            : framesToTensor(rawFrames, captureFrameCount, captureSz, captureSz);
      }
      setPhase("evaluating");
      modelResult = modelResult ?? await runInference(modelInput);
      const { predictedLabel, confidence: conf, topPredictions: top } = modelResult;
      const nextRouteInfo = modelResult.routedBy && modelResult.primaryPrediction && modelResult.specialistPrediction
        ? {
            routedBy: modelResult.routedBy,
            primaryLabel: modelResult.primaryPrediction.predictedLabel,
            primaryConfidence: modelResult.primaryPrediction.confidence,
            specialistLabel: modelResult.specialistPrediction.predictedLabel,
            specialistConfidence: modelResult.specialistPrediction.confidence,
          }
        : null;
      const signThresholds = personalizeThresholds(
        thresholdsFor(calibration, current.sign_id),
        current.sign_id,
        feedbackSummary
      );
      let result = evaluateAttempt(
        current.sign_id,
        predictedLabel,
        conf,
        signThresholds.passThreshold,
        signThresholds.retryThreshold
      );
      const agreement = modelResult.agreement ?? 1;
      if (result.outcome === "pass" && agreement < MIN_WINDOW_AGREEMENT_FOR_PASS) {
        result = { ...result, outcome: "retry" };
      }
      setOutcome(result.outcome);
      setConfidence(result.confidence);
      setPredicted(predictedLabel);
      setTopPredictions(top);
      setWindowPredictions(modelResult.windowPredictions ?? []);
      setWindowAgreement(modelResult.agreement ?? null);
      setRouteInfo(nextRouteInfo);

      const watchlistSelfCheck =
        reliabilityFor(calibration?.thresholds?.[current.sign_id]) === "weak" &&
        predictedLabel === current.sign_id &&
        result.outcome !== "pass";
      const hintReason =
        result.outcome === "retry" ? "framing" : result.outcome === "fail" ? "fail" : "pass";
      if (result.outcome !== "pass") {
        const targetedHint = agreement < MIN_WINDOW_AGREEMENT_FOR_PASS
          ? "The model only saw that result in part of the recording. Try again with a slower, centered sign so all capture windows agree."
          : watchlistSelfCheck ? null : confusionHint(calibration, current.sign_id, predictedLabel);
        if (targetedHint) {
          setHint(targetedHint);
        } else if (watchlistSelfCheck) {
          setHint(null);
        } else {
          const h = await fetchHint(current.sign_id, hintReason, userId);
          setHint(h.message);
        }
      } else {
        setHint(null);
      }

      try {
        await saveOutcome(result.outcome, conf, predictedLabel, "model", nextRouteInfo);
      } catch (err) {
        trackEvent("attempt_record_error", { error: String(err), source: "model" });
      }
      setPhase("result");
    } catch (err) {
      setOutcome("fail");
      setConfidence(0);
      setPredicted("");
      resetRecognitionDetails();
      setHint("Model could not run. Ensure model files are synced and reload.");
      setPhase("result");
      trackEvent("inference_error", { error: String(err) });
    }
  };

  const recordAndEvaluate = () => {
    setRecognitionFeedbackSaved(false);
    setCorrectionStatus("");
    resetRecognitionDetails();
    // The model-specific capture path inside evaluate() waits for its capture window.
    // UI just needs to flip to "recording".
    setPhase("recording");
    void evaluate();
  };

  const nextSign = () => {
    setOutcome(null);
    setHint(null);
    setCorrectionStatus("");
    clearSelfCheckClip();
    resetRecognitionDetails();
    setPhase("prompt");
    if (index + 1 < orderedSigns.length) setIndex(index + 1);
    else setIndex(0);
  };

  const saveRecognitionFeedback = (correct: boolean) => {
    if (!current || !predicted || predicted === "low_tracking") return;
      const entry = {
      signId: current.sign_id,
      sign_id: current.sign_id,
      predictedLabel: predicted,
      predicted_label: predicted,
      confidence,
      accepted: correct,
      correct,
      top_predictions: topPredictions,
      window_predictions: windowPredictions,
      agreement: windowAgreement ?? undefined,
      tracking_ratio: trackingRatio,
      model_version: modelVersion,
      routed_by: routeInfo?.routedBy,
      primary_predicted_label: routeInfo?.primaryLabel,
      primary_confidence: routeInfo?.primaryConfidence,
      specialist_predicted_label: routeInfo?.specialistLabel,
      specialist_confidence: routeInfo?.specialistConfidence,
      ts: Date.now(),
    };
    try {
      const key = "recognition_feedback";
      const existing = JSON.parse(localStorage.getItem(key) || "[]") as unknown[];
      const next = [...existing.slice(-199), entry];
      localStorage.setItem(key, JSON.stringify(next));
      setRecognitionFeedback(readRecognitionFeedback());
    } catch {
      // Feedback still gets sent as an analytics event below.
    }
    trackEvent("recognition_feedback", entry);
    setRecognitionFeedbackSaved(true);
  };

  const recordCorrectionClip = async () => {
    if (!current || !predicted || predicted === "low_tracking") return;
    if (!streamIsLive(streamRef.current)) {
      await startCamera();
    }
    if (!streamIsLive(streamRef.current)) {
      setCorrectionStatus("Camera preview is not active yet. Click Start camera preview, allow permission, then try again.");
      return;
    }
    const stream = streamRef.current;
    if (!stream) return;
    setCorrectionStatus("Recording correction clip...");
    try {
      const blob = await recordVideo(stream, CORRECTION_RECORD_MS);
      const filename = `${current.sign_id}_correction_pred-${predicted}_conf-${Math.round(confidence * 100)}_${Date.now()}.webm`;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      const clip: CorrectionClip = {
        filename,
        sign_id: current.sign_id,
        predicted_label: predicted,
        confidence,
        agreement: windowAgreement ?? undefined,
        tracking_ratio: trackingRatio,
        captured_at: new Date().toISOString(),
      };
      const next = [...correctionClips.slice(-99), clip];
      setCorrectionClips(next);
      localStorage.setItem(CORRECTION_MANIFEST_KEY, JSON.stringify(next));
      trackEvent("correction_clip_recorded", clip);
      setCorrectionStatus(`Saved correction clip ${filename} to Downloads.`);
    } catch (err) {
      setCorrectionStatus(`Correction clip failed: ${(err as Error).message}`);
    }
  };

  const exportCorrectionManifest = () => {
    const payload = {
      exported_at: new Date().toISOString(),
      incoming_dir: "ml\\data\\incoming",
      clips: correctionClips,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `recognition_corrections_${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
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
        Sign {index + 1} of {orderedSigns.length}: <strong style={{ fontSize: "1.5rem" }}>{current?.gloss}</strong>
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
      {unitFilter.length > 0 && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <strong>Unit practice</strong>
          <p style={{ color: "var(--muted)", marginBottom: 0 }}>
            Practicing {unitFilter.length} signs from your selected Learn unit. Return to the lobby to start a full session.
          </p>
        </div>
      )}
      {practiceMode === "recognition" && current && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <strong>Recognition readiness</strong>
          <div className="metric-grid" style={{ marginTop: "0.75rem", marginBottom: 0 }}>
            <div>
              <span className="metric-label">Model reliability</span>
              <strong className={currentReliability === "strong" ? "status-pass" : currentReliability === "weak" ? "status-fail" : "status-retry"}>
                {currentReliability}
              </strong>
            </div>
            <div>
              <span className="metric-label">Pass threshold</span>
              <strong>{currentThresholds ? `${Math.round(currentThresholds.passThreshold * 100)}%` : "n/a"}</strong>
            </div>
            <div>
              <span className="metric-label">Local feedback</span>
              <strong>
                {currentFeedback ? `${currentFeedback.rejected}/${currentFeedback.total} wrong` : "none yet"}
              </strong>
            </div>
          </div>
          {currentConfusions.length > 0 && (
            <p style={{ color: "var(--muted)", fontSize: "0.9rem", marginBottom: 0 }}>
              Often confused with: {currentConfusions.map((row) => `${row.confusedWith} (${row.count})`).join(", ")}.
            </p>
          )}
        </div>
      )}

      <div className="card" style={{ marginTop: "1rem", display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
        <strong>Mode</strong>
        <button
          className={practiceMode === "guided" ? "btn" : "btn btn-secondary"}
          type="button"
          aria-pressed={practiceMode === "guided"}
          onClick={() => updatePracticeMode("guided")}
        >
          Guided self-check
        </button>
        <button
          className={practiceMode === "recognition" ? "btn" : "btn btn-secondary"}
          type="button"
          aria-pressed={practiceMode === "recognition"}
          onClick={() => updatePracticeMode("recognition")}
        >
          Recognition demo
        </button>
        {practiceMode === "recognition" && (
          <span style={{ color: "var(--retry)", fontSize: "0.85rem" }}>
            Personalized with local correctness feedback after 3+ labels/sign.
          </span>
        )}
      </div>

      <div className="card" style={{ marginTop: "1rem", display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
        <strong>Order</strong>
        <button
          className={practiceOrder === "weak_first" ? "btn" : "btn btn-secondary"}
          type="button"
          aria-pressed={practiceOrder === "weak_first"}
          onClick={() => updatePracticeOrder("weak_first")}
        >
          Weak first
        </button>
        <button
          className={practiceOrder === "shuffle" ? "btn" : "btn btn-secondary"}
          type="button"
          aria-pressed={practiceOrder === "shuffle"}
          onClick={() => updatePracticeOrder("shuffle")}
        >
          Mixed review
        </button>
        <button
          className={practiceOrder === "confusions" ? "btn" : "btn btn-secondary"}
          type="button"
          aria-pressed={practiceOrder === "confusions"}
          onClick={() => updatePracticeOrder("confusions")}
        >
          Confusion drill
        </button>
        <button
          className={practiceOrder === "default" ? "btn" : "btn btn-secondary"}
          type="button"
          aria-pressed={practiceOrder === "default"}
          onClick={() => updatePracticeOrder("default")}
        >
          Catalog
        </button>
      </div>

      {cameraError ? (
        <div className="card status-fail">
          <p>{cameraError}</p>
          <button className="btn" onClick={startCamera}>
            {cameraNeedsStart ? "Start camera" : "Retry camera"}
          </button>
        </div>
      ) : (
        <div className="video-wrap" style={{ position: "relative" }}>
          <video
            ref={videoRef}
            muted
            playsInline
            autoPlay
            onLoadedMetadata={() => setCameraReady(true)}
            onCanPlay={() => setCameraReady(true)}
          />
          <div className="guide-box" title="Keep hands and face inside box" />
          {(!cameraReady || cameraNeedsStart) && phase !== "recording" && phase !== "evaluating" && (
            <div className="camera-overlay">
              <strong>{cameraNeedsStart ? "Camera needs a click" : "Camera preview not visible yet"}</strong>
              <button className="btn" type="button" onClick={() => void startCamera()}>
                Start camera preview
              </button>
            </div>
          )}
          {practiceMode === "recognition" && phase !== "recording" && phase !== "evaluating" && (
            <div
              style={{
                position: "absolute",
                right: 8,
                top: 8,
                background:
                  liveTrackingRatio === null
                    ? "rgba(15, 20, 25, 0.82)"
                    : liveTrackingRatio >= 0.75
                      ? "rgba(34, 197, 94, 0.88)"
                      : liveTrackingRatio >= MIN_HAND_TRACKING_RATIO
                        ? "rgba(245, 158, 11, 0.9)"
                        : "rgba(239, 68, 68, 0.9)",
                color: "white",
                padding: "0.25rem 0.6rem",
                borderRadius: "6px",
                fontWeight: 700,
                fontSize: "0.82rem",
              }}
            >
              {liveTrackingLabel(liveTrackingRatio)}
            </div>
          )}
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

      {modelError && practiceMode === "recognition" && (
        <div className="card status-fail" style={{ marginTop: "1rem" }}>
          <strong>Recognition model unavailable</strong>
          <p style={{ margin: "0.25rem 0 0" }}>{modelError}</p>
        </div>
      )}

      <div className="card" style={{ marginTop: "1rem" }}>
        {phase === "prompt" && (
          <>
            {practiceMode === "guided" ? (
              <p>Record your sign, compare handshape, movement, and location with the reference, then log how it went.</p>
            ) : current?.trained === false ? (
              <p style={{ color: "var(--muted)" }}>
                This sign isn't in the trained model yet. Use the reference below to learn it, then skip to the next sign.
              </p>
            ) : (
              <p>When you click Record, perform the sign inside the box. Recording lasts about {MULTI_WINDOW_CAPTURE_MS / 1000} seconds.</p>
            )}
            <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", flexWrap: "wrap" }}>
              {practiceMode === "guided" && (
                <button className="btn" disabled={!!cameraError || !cameraReady} onClick={() => void startSelfCheck()}>
                  Record & self-check
                </button>
              )}
              {practiceMode === "recognition" && current?.trained !== false && (
                <button className="btn" disabled={!!cameraError || !!modelError || !cameraReady} onClick={recordAndEvaluate}>
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
              {practiceMode === "recognition" && current?.trained === false && (
                <button className="btn" type="button" onClick={nextSign}>
                  Next sign
                </button>
              )}
            </div>
            {showReference && reference && (
              <div className="hint-panel" style={{ marginTop: "0.75rem" }}>
                <strong>Reference</strong>
                <ReferenceVideo signId={current.sign_id} />
                <p style={{ margin: "0.25rem 0 0" }}>
                  Handshape: {reference.handshape}<br />
                  Movement: {reference.movement}<br />
                  Location: {reference.location}
                </p>
              </div>
            )}
          </>
        )}
        {phase === "recording" && <p>Recording... keep your hands visible inside the guide box.</p>}
        {phase === "selfCheck" && (
          <>
            <p>Compare your sign with the reference, then log the attempt.</p>
            {reference && (
              <div className="hint-panel" style={{ marginBottom: "0.75rem" }}>
                <strong>Reference</strong>
                <p style={{ margin: "0.25rem 0 0" }}>
                  Handshape: {reference.handshape}<br />
                  Movement: {reference.movement}<br />
                  Location: {reference.location}
                </p>
              </div>
            )}
            {selfCheckUrl && (
              <div className="review-grid" style={{ marginBottom: "0.75rem" }}>
                <div>
                  <strong>Your recording</strong>
                  <video
                    src={selfCheckUrl}
                    controls
                    playsInline
                    style={{ width: "100%", marginTop: "0.5rem", borderRadius: 8, background: "#000" }}
                  />
                </div>
                <div>
                  <strong>Reference video</strong>
                  <ReferenceVideo signId={current.sign_id} />
                </div>
              </div>
            )}
            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <button className="btn" onClick={() => void completeSelfCheck("pass")}>
                Matched it
              </button>
              <button className="btn btn-secondary" onClick={() => void completeSelfCheck("retry")}>
                Needs practice
              </button>
            </div>
          </>
        )}
        {phase === "evaluating" && <p>Evaluating locally...</p>}
        {phase === "result" && outcome && (
          <>
            <p className={`status-${outcome}`} style={{ fontSize: "1.25rem", fontWeight: 600 }}>
              {outcome === "pass" ? "Matched" : outcome === "retry" ? "Review once more" : "Try again"}
              {practiceMode === "recognition" && predicted !== "low_tracking" && outcome !== "pass" && ` - ${(confidence * 100).toFixed(0)}% confidence`}
            </p>
            {practiceMode === "recognition" && predicted && predicted !== "low_tracking" && (
              <p style={{ color: "var(--muted)" }}>
                Detected: {predicted} ({(confidence * 100).toFixed(0)}%)
              </p>
            )}
            {practiceMode === "recognition" && routeInfo && (
              <div className="hint-panel" style={{ marginTop: "0.75rem" }}>
                <strong>Specialist model used</strong>
                <p style={{ margin: "0.25rem 0 0" }}>
                  {routeInfo.routedBy} changed the result from{" "}
                  <code>{routeInfo.primaryLabel}</code> ({(routeInfo.primaryConfidence * 100).toFixed(0)}%) to{" "}
                  <code>{routeInfo.specialistLabel}</code> ({(routeInfo.specialistConfidence * 100).toFixed(0)}%).
                </p>
              </div>
            )}
            {practiceMode === "recognition" && currentReliability === "weak" && predicted === current?.sign_id && (
              <div className="hint-panel" style={{ marginTop: "0.75rem" }}>
                <strong>Model watchlist</strong>
                <p>{WATCHLIST_SELF_CHECK_HINT}</p>
              </div>
            )}
            {practiceMode === "recognition" && trackingRatio !== null && (
              <p style={{ color: "var(--muted)" }}>
                Hand tracking: {Math.round(trackingRatio * 100)}% of frames
              </p>
            )}
            {practiceMode === "recognition" && windowAgreement !== null && windowPredictions.length > 1 && (
              <div className="hint-panel" style={{ marginTop: "0.75rem" }}>
                <strong>Recognition consistency</strong>
                <p style={{ margin: "0.25rem 0 0" }}>
                  {Math.round(windowAgreement * 100)}% agreement across capture windows
                  {windowAgreement < MIN_WINDOW_AGREEMENT_FOR_PASS ? ". Try once more before trusting this result." : "."}
                </p>
                <ol style={{ margin: "0.35rem 0 0", paddingLeft: "1.4rem" }}>
                  {windowPredictions.map((p, i) => (
                    <li key={`${p.label}-${i}`}>
                      Window {i + 1}: {p.label} ({(p.confidence * 100).toFixed(0)}%)
                    </li>
                  ))}
                </ol>
              </div>
            )}
            {practiceMode === "recognition" && topPredictions.length > 0 && (
              <div style={{ marginTop: "0.5rem" }}>
                <strong>Top matches</strong>
                <ol style={{ margin: "0.35rem 0 0", paddingLeft: "1.4rem" }}>
                  {topPredictions.map((p) => (
                    <li key={p.label}>
                      {p.label}: {(p.confidence * 100).toFixed(0)}%
                    </li>
                  ))}
                </ol>
              </div>
            )}
            {hint && (
              <div className="hint-panel">
                <strong>Hint</strong>
                <p>{hint}</p>
              </div>
            )}
            {practiceMode === "recognition" && predicted && predicted !== "low_tracking" && (
              <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap", marginTop: "1rem" }}>
                <span style={{ color: "var(--muted)", fontSize: "0.9rem" }}>Was recognition right?</span>
                <button className="btn btn-secondary" type="button" onClick={() => saveRecognitionFeedback(true)}>
                  Yes
                </button>
                <button className="btn btn-secondary" type="button" onClick={() => saveRecognitionFeedback(false)}>
                  No
                </button>
                {recognitionFeedbackSaved && <span className="status-pass">Saved</span>}
              </div>
            )}
            {practiceMode === "recognition" && predicted && predicted !== "low_tracking" && outcome !== "pass" && (
              <div className="hint-panel" style={{ marginTop: "0.75rem" }}>
                <strong>Improve this sign</strong>
                <p style={{ margin: "0.25rem 0 0" }}>
                  Record a clean correction clip now. It will download locally and can be imported into the next training run.
                </p>
                <div className="button-row">
                  <button className="btn" type="button" onClick={() => void recordCorrectionClip()}>
                    Record correction clip
                  </button>
                  <button className="btn btn-secondary" type="button" disabled={correctionClips.length === 0} onClick={exportCorrectionManifest}>
                    Export correction manifest
                  </button>
                </div>
                {correctionStatus && <p style={{ color: "var(--muted)", fontSize: "0.9rem" }}>{correctionStatus}</p>}
              </div>
            )}
            <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
              {outcome !== "pass" && (
                <button
                  className="btn"
                  onClick={() => {
                    setPhase("prompt");
                    setOutcome(null);
                    setHint(null);
                    setCorrectionStatus("");
                    resetRecognitionDetails();
                  }}
                >
                  Retry
                </button>
              )}
              <button className="btn btn-secondary" onClick={nextSign}>
                {outcome === "pass" ? "Next sign" : "Keep going"}
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
                {e.sign}: <span className={`status-${e.outcome}`}>{outcomeLabel(e.outcome)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="footer-meta">Model: {modelVersion} · Inference runs locally · No video upload</p>
    </div>
  );
}
