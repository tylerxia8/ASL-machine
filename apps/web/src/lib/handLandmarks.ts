import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

const NUM_FRAMES = 24;
const HAND_FEATURES = 132;
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";

let landmarkerPromise: Promise<HandLandmarker> | null = null;

async function getLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = FilesetResolver.forVisionTasks(WASM_URL).then((vision) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: HAND_MODEL_URL,
          delegate: "CPU",
        },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence: 0.35,
        minTrackingConfidence: 0.4,
      })
    );
  }
  return landmarkerPromise;
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
  const landmarker = await getLandmarker();
  const frames: Float32Array[] = [];
  const step = durationMs / Math.max(count - 1, 1);
  const start = performance.now();
  for (let i = 0; i < count; i++) {
    const targetT = start + i * step;
    const wait = targetT - performance.now();
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
    const result = landmarker.detectForVideo(video, performance.now());
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
  return out;
}
