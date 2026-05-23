import { flattenTensor } from "./camera";

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
  const labelsRes = await fetch("/models/labels.json");
  if (!labelsRes.ok) {
    throw new ModelUnavailableError(
      "labels.json not found at /models/labels.json. Run npm run sync-model after training."
    );
  }
  labels = await labelsRes.json();

  const head = await fetch("/models/model.onnx", { method: "HEAD" });
  if (!head.ok) {
    throw new ModelUnavailableError(
      "model.onnx not found at /models/model.onnx. Export the model and run npm run sync-model."
    );
  }

  const ort = await import("onnxruntime-web");
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create("/models/model.onnx", {
    executionProviders: ["wasm"],
  });
  inputName = session.inputNames[0] ?? "input";

  return { version: labels!.model_version, numClasses: labels!.sign_ids.length };
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
