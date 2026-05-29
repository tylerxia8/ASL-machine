import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const NUM_FRAMES = 24;
const HAND_FEATURES = 132;
const LOCAL_HAND_MODEL_URL = "/mediapipe/hand_landmarker.task";
const LOCAL_WASM_URL = "/mediapipe/wasm";
const REMOTE_HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const REMOTE_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

let landmarkerPromise: Promise<HandLandmarker> | null = null;

export type HandLandmarkCapture = {
  tensor: Float32Array;
  detectedFrameCount: number;
  detectedFrameRatio: number;
};

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = createLandmarker(LOCAL_WASM_URL, LOCAL_HAND_MODEL_URL).catch(() =>
      createLandmarker(REMOTE_WASM_URL, REMOTE_HAND_MODEL_URL)
    );
  }
  return landmarkerPromise;
}

async function createLandmarker(wasmUrl: string, modelAssetPath: string) {
  const vision = await FilesetResolver.forVisionTasks(wasmUrl);
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath,
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.35,
    minHandPresenceConfidence: 0.35,
    minTrackingConfidence: 0.4,
  });
}

function handVector(landmarks: { x: number; y: number; z: number }[]) {
  const out = new Float32Array(66);
  if (landmarks.length === 0) return out;
  const wrist = landmarks[0];
  out[0] = wrist.x;
  out[1] = wrist.y;
  out[2] = wrist.z;
  let idx = 3;
  for (const lm of landmarks) {
    out[idx++] = lm.x - wrist.x;
    out[idx++] = lm.y - wrist.y;
    out[idx++] = lm.z - wrist.z;
  }
  return out;
}

function frameFeatures(result: {
  landmarks?: { x: number; y: number; z: number }[][];
  handednesses?: { categoryName?: string }[][];
}) {
  const out = new Float32Array(HAND_FEATURES);
  const hands = result.landmarks ?? [];
  hands.forEach((landmarks, i) => {
    const label = result.handednesses?.[i]?.[0]?.categoryName?.toLowerCase() ?? "";
    const vec = handVector(landmarks);
    let offset = 0;
    if (label.startsWith("right")) offset = 66;
    else if (!label.startsWith("left")) offset = vec[0] < 0.5 ? 0 : 66;
    out.set(vec, offset);
  });
  return out;
}

export async function captureHandLandmarkTensor(
  video: HTMLVideoElement,
  count = NUM_FRAMES,
  durationMs = 2000
): Promise<Float32Array> {
  return (await captureHandLandmarkSample(video, count, durationMs)).tensor;
}

export async function captureHandLandmarkSample(
  video: HTMLVideoElement,
  count = NUM_FRAMES,
  durationMs = 2000
): Promise<HandLandmarkCapture> {
  const landmarker = await getLandmarker();
  const frames: Float32Array[] = [];
  let detectedFrameCount = 0;
  const step = durationMs / Math.max(count - 1, 1);
  const start = performance.now();
  for (let i = 0; i < count; i++) {
    const targetT = start + i * step;
    const wait = targetT - performance.now();
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    const result = landmarker.detectForVideo(video, performance.now());
    if ((result.landmarks ?? []).length > 0) detectedFrameCount += 1;
    frames.push(frameFeatures(result));
  }

  // Model input shape is [1, HAND_FEATURES, NUM_FRAMES].
  const out = new Float32Array(HAND_FEATURES * count);
  let idx = 0;
  for (let f = 0; f < HAND_FEATURES; f++) {
    for (let t = 0; t < count; t++) {
      out[idx++] = frames[t][f] ?? 0;
    }
  }
  return {
    tensor: out,
    detectedFrameCount,
    detectedFrameRatio: detectedFrameCount / Math.max(count, 1),
  };
}

export async function getHandTrackingRatio(video: HTMLVideoElement, frames = 4, durationMs = 300) {
  const landmarker = await getLandmarker();
  let detected = 0;
  const step = durationMs / Math.max(frames - 1, 1);
  const start = performance.now();
  for (let i = 0; i < frames; i++) {
    const targetT = start + i * step;
    const wait = targetT - performance.now();
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    const result = landmarker.detectForVideo(video, performance.now());
    if ((result.landmarks ?? []).length > 0) detected += 1;
  }
  return detected / Math.max(frames, 1);
}

export async function captureHandLandmarkWindows(
  video: HTMLVideoElement,
  windows = 3,
  count = NUM_FRAMES,
  durationMs = 2400
): Promise<HandLandmarkCapture[]> {
  const landmarker = await getLandmarker();
  const totalFrames = Math.max(count + windows - 1, count);
  const frames: Float32Array[] = [];
  const detected: boolean[] = [];
  const step = durationMs / Math.max(totalFrames - 1, 1);
  const start = performance.now();
  for (let i = 0; i < totalFrames; i++) {
    const targetT = start + i * step;
    const wait = targetT - performance.now();
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    const result = landmarker.detectForVideo(video, performance.now());
    detected.push((result.landmarks ?? []).length > 0);
    frames.push(frameFeatures(result));
  }

  const samples: HandLandmarkCapture[] = [];
  for (let w = 0; w < windows; w++) {
    const startIdx = Math.round((w * (totalFrames - count)) / Math.max(windows - 1, 1));
    const selected = frames.slice(startIdx, startIdx + count);
    const selectedDetected = detected.slice(startIdx, startIdx + count);
    const tensor = new Float32Array(HAND_FEATURES * count);
    let idx = 0;
    for (let f = 0; f < HAND_FEATURES; f++) {
      for (let t = 0; t < count; t++) {
        tensor[idx++] = selected[t]?.[f] ?? 0;
      }
    }
    const detectedFrameCount = selectedDetected.filter(Boolean).length;
    samples.push({
      tensor,
      detectedFrameCount,
      detectedFrameRatio: detectedFrameCount / Math.max(count, 1),
    });
  }
  return samples;
}
