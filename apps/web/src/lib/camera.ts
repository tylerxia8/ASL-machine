export type CameraError = "denied" | "unsupported" | "not_found" | "unknown";

export async function requestCamera(): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw { code: "unsupported" as CameraError };
  }
  try {
    return await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    const err = e as DOMException;
    if (err.name === "NotAllowedError") throw { code: "denied" as CameraError };
    if (err.name === "NotFoundError") throw { code: "not_found" as CameraError };
    throw { code: "unknown" as CameraError, message: err.message };
  }
}

type Planes = [number[], number[], number[]];

function grabFrame(
  video: HTMLVideoElement,
  ctx: CanvasRenderingContext2D,
  size: number
): Planes {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2;
  const sy = (vh - side) / 2;
  ctx.drawImage(video, sx, sy, side, side, 0, 0, size, size);
  const img = ctx.getImageData(0, 0, size, size);
  const n = size * size;
  const r = new Array<number>(n);
  const g = new Array<number>(n);
  const b = new Array<number>(n);
  for (let p = 0, i = 0; i < n; p += 4, i++) {
    r[i] = img.data[p] / 255;
    g[i] = img.data[p + 1] / 255;
    b[i] = img.data[p + 2] / 255;
  }
  return [r, g, b];
}

/**
 * Capture `count` frames over `durationMs` real-time milliseconds. Spaces samples
 * evenly so the resulting clip has actual motion. Use this for practice-time
 * evaluation; recording for training should use `recordVideo` instead.
 */
export async function captureFramesAsync(
  video: HTMLVideoElement,
  count: number,
  size: number,
  durationMs: number
): Promise<Planes[]> {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const frames: Planes[] = [];
  const step = durationMs / Math.max(count - 1, 1);
  const start = performance.now();
  for (let i = 0; i < count; i++) {
    const targetT = start + i * step;
    const wait = targetT - performance.now();
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    frames.push(grabFrame(video, ctx, size));
  }
  return frames;
}

/**
 * Legacy sync grabber — DO NOT use for new code. All 24 frames return identical
 * pixels because the video element doesn't advance between calls in one tick.
 * Kept temporarily until existing call sites migrate to captureFramesAsync.
 * @deprecated
 */
export function captureFrames(
  video: HTMLVideoElement,
  count: number,
  size: number
): Planes[] {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;
  const frames: Planes[] = [];
  for (let i = 0; i < count; i++) frames.push(grabFrame(video, ctx, size));
  return frames;
}

/**
 * Record the given camera stream for `durationMs` and return a webm Blob.
 * Throws if MediaRecorder is unsupported.
 */
export async function recordVideo(stream: MediaStream, durationMs: number): Promise<Blob> {
  if (typeof MediaRecorder === "undefined") {
    throw new Error("MediaRecorder is not supported in this browser.");
  }
  const mimeCandidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  const mime = mimeCandidates.find((m) => MediaRecorder.isTypeSupported(m)) || "";
  const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
  const chunks: BlobPart[] = [];
  return new Promise<Blob>((resolve, reject) => {
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };
    rec.onerror = (e) => reject(e);
    rec.onstop = () => resolve(new Blob(chunks, { type: rec.mimeType || "video/webm" }));
    rec.start();
    setTimeout(() => rec.state !== "inactive" && rec.stop(), durationMs);
  });
}

/** Flatten [1,3,T,H,W] tensor for sklearn ONNX models */
export function flattenTensor(tensor: Float32Array, nFeatures?: number): Float32Array {
  if (nFeatures && tensor.length >= nFeatures) return tensor.subarray(0, nFeatures);
  return tensor;
}

/** Stack frames into tensor [1, 3, T, H, W] */
export function framesToTensor(frames: number[][][], t: number, h: number, w: number): Float32Array {
  const selected = resampleFrames(frames, t);
  const out = new Float32Array(1 * 3 * t * h * w);
  let idx = 0;
  for (let c = 0; c < 3; c++) {
    for (let fi = 0; fi < t; fi++) {
      const plane = selected[fi][c];
      for (let i = 0; i < h * w; i++) {
        out[idx++] = plane[i] ?? 0;
      }
    }
  }
  return out;
}

function resampleFrames(frames: number[][][], target: number): number[][][] {
  if (frames.length === target) return frames;
  const out: number[][][] = [];
  for (let i = 0; i < target; i++) {
    const src = frames[Math.floor((i / target) * frames.length)] ?? frames[frames.length - 1];
    out.push(src);
  }
  return out;
}
