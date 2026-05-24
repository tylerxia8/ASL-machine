import { flattenTensor } from "./camera";
import { resolveSelectedSource } from "./modelSource";

export type LabelsFile = {
  sign_ids: string[];
  label_to_idx: Record<string, number>;
  model_version: string;
  input_type?: "flat" | "3d";
  n_features?: number;
  num_frames?: number;
  frame_size?: number;
};

export class ModelUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ModelUnavailableError";
  }
}

let labels: LabelsFile | null = null;
let session: import("onnxruntime-web").InferenceSession | null = null;
let inputName = "input";
let loadPromise: Promise<{ version: string; numClasses: number }> | null = null;

export function getLabels() {
  return labels;
}

async function doLoad(): Promise<{ version: string; numClasses: number }> {
  const source = await resolveSelectedSource();
  const labelsRes = await fetch(source.labelsUrl);
  if (!labelsRes.ok) {
    throw new ModelUnavailableError(
      `labels.json not found at ${source.labelsUrl}. Run npm run sync-model after training, or pick a different model source in the Lobby.`
    );
  }
  labels = await labelsRes.json();

  // Releases redirect through release-assets.githubusercontent.com on download
  // — HEAD doesn't always return 200 there, so for non-bundled sources skip the
  // HEAD check and let ort.InferenceSession.create report any failure.
  if (source.id === "bundled") {
    const head = await fetch(source.modelUrl, { method: "HEAD" });
    if (!head.ok) {
      throw new ModelUnavailableError(
        `model.onnx not found at ${source.modelUrl}. Export the model and run npm run sync-model.`
      );
    }
  }

  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(source.modelUrl, {
    executionProviders: ["wasm"],
  });
  inputName = session.inputNames[0] ?? "input";

  return { version: labels!.model_version, numClasses: labels!.sign_ids.length };
}

export function resetLoadCache() {
  loadPromise = null;
  session = null;
  labels = null;
}

export async function loadModel() {
  if (!loadPromise) loadPromise = doLoad();
  return loadPromise;
}

export async function runInference(tensorData: Float32Array) {
  if (!session || !labels) {
    await loadModel();
  }
  if (!session || !labels) {
    throw new ModelUnavailableError("Inference session not initialized.");
  }

  const ort = await import("onnxruntime-web");
  const inputData =
    labels.input_type === "flat" ? flattenTensor(tensorData, labels.n_features) : tensorData;
  const shape: number[] =
    labels.input_type === "flat"
      ? [1, inputData.length]
      : [1, 3, labels.num_frames ?? 24, labels.frame_size ?? 160, labels.frame_size ?? 160];

  const input = new ort.Tensor("float32", inputData, shape);
  const results = await session.run({ [inputName]: input });
  const outKey = session.outputNames[0];
  const logits = results[outKey].data as Float32Array;
  const probs = softmax(Array.from(logits));

  let bestIdx = 0;
  let best = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > best) {
      best = probs[i];
      bestIdx = i;
    }
  }
  return {
    predictedLabel: labels.sign_ids[bestIdx],
    confidence: best,
    probs,
  };
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
