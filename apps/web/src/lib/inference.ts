import { flattenTensor } from "./camera";
import { BUNDLED_SOURCE, releaseTagToModelSource, resolveSelectedSource, type ModelSource } from "./modelSource";
import { routeSpecialistPrediction, type EnsembleConfig, type SpecialistRoute } from "./ensembleRouting";

export type LabelsFile = {
  sign_ids: string[];
  label_to_idx: Record<string, number>;
  model_version: string;
  input_type?: "flat" | "3d" | "hand_landmarks";
  n_features?: number;
  num_frames?: number;
  frame_size?: number;
  preprocess?: "center_crop" | "letterbox";
  pretrained_detector?: string;
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
let primarySourceId = BUNDLED_SOURCE.id;
let ensembleConfig: EnsembleConfig | null = null;
let specialistRuntime: ModelRuntime | null = null;

type ModelRuntime = {
  route: SpecialistRoute;
  source: ModelSource;
  labels: LabelsFile;
  session: import("onnxruntime-web").InferenceSession;
  inputName: string;
};

export type InferenceSummary = {
  predictedLabel: string;
  confidence: number;
  margin: number;
  topPredictions: { label: string; confidence: number }[];
  probs: number[];
  modelVersion?: string;
};

export function getLabels() {
  return labels;
}

async function doLoad(): Promise<{ version: string; numClasses: number }> {
  const source = await resolveSelectedSource();
  primarySourceId = source.id;
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
  await loadEnsembleConfig();

  return { version: labels!.model_version, numClasses: labels!.sign_ids.length };
}

export function resetLoadCache() {
  loadPromise = null;
  session = null;
  labels = null;
  ensembleConfig = null;
  specialistRuntime = null;
}

export async function loadModel() {
  if (!loadPromise) loadPromise = doLoad();
  return loadPromise;
}

export async function runInference(tensorData: Float32Array) {
  return runInferenceBatch([tensorData]);
}

export async function runInferenceBatch(tensorDataRows: Float32Array[]) {
  if (!session || !labels) {
    await loadModel();
  }
  if (!session || !labels) {
    throw new ModelUnavailableError("Inference session not initialized.");
  }

  const ort = await import("onnxruntime-web");
  const primaryProbs = await runRuntimeBatch({
    ort,
    session,
    inputName,
    labels,
    tensorDataRows,
  });
  const primary = summarizeProbabilitiesForLabels(primaryProbs, labels);
  const specialist = await maybeRunSpecialist({ ort, tensorDataRows, primary });
  return specialist ?? primary;
}

export function summarizeProbabilities(probs: number[]) {
  if (!labels) {
    throw new ModelUnavailableError("Labels not initialized.");
  }
  return summarizeProbabilitiesForLabels(probs, labels);
}

export function summarizeProbabilitiesForLabels(probs: number[], sourceLabels: LabelsFile): InferenceSummary {
  let bestIdx = 0;
  let best = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > best) {
      best = probs[i];
      bestIdx = i;
    }
  }
  const topPredictions = probs
    .map((confidence, idx) => ({ label: sourceLabels.sign_ids[idx], confidence }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 3);
  const margin = topPredictions[0].confidence - (topPredictions[1]?.confidence ?? 0);
  return {
    predictedLabel: sourceLabels.sign_ids[bestIdx],
    confidence: best,
    margin,
    topPredictions,
    probs,
    modelVersion: sourceLabels.model_version,
  };
}

async function loadEnsembleConfig() {
  if (primarySourceId !== BUNDLED_SOURCE.id) return;
  try {
    const res = await fetch("/models/ensemble_config.json", { cache: "no-store" });
    if (!res.ok) return;
    ensembleConfig = await res.json();
  } catch {
    ensembleConfig = null;
  }
}

async function loadSpecialistRuntime(route: SpecialistRoute): Promise<ModelRuntime | null> {
  if (specialistRuntime?.route.id === route.id) return specialistRuntime;
  try {
    const source = releaseTagToModelSource(route.releaseTag, route.label);
    const labelsRes = await fetch(source.labelsUrl);
    if (!labelsRes.ok) return null;
    const specialistLabels = (await labelsRes.json()) as LabelsFile;
    if (!labels || specialistLabels.input_type !== labels.input_type) return null;
    if (specialistLabels.sign_ids.join("|") !== labels.sign_ids.join("|")) return null;

    const ort = await import("onnxruntime-web");
    const specialistSession = await ort.InferenceSession.create(source.modelUrl, {
      executionProviders: ["wasm"],
    });
    specialistRuntime = {
      route,
      source,
      labels: specialistLabels,
      session: specialistSession,
      inputName: specialistSession.inputNames[0] ?? "input",
    };
    return specialistRuntime;
  } catch {
    return null;
  }
}

async function maybeRunSpecialist({
  ort,
  tensorDataRows,
  primary,
}: {
  ort: typeof import("onnxruntime-web");
  tensorDataRows: Float32Array[];
  primary: InferenceSummary;
}) {
  if (!ensembleConfig?.enabled || ensembleConfig.primarySourceId !== primarySourceId) return null;
  const route = ensembleConfig.specialists?.[0];
  if (!route || !labels) return null;
  const allowed = new Set(route.allowedSigns);
  if (!allowed.has(primary.predictedLabel) && primary.confidence > (route.maxPrimaryConfidence ?? 1)) {
    return null;
  }
  const runtime = await loadSpecialistRuntime(route);
  if (!runtime) return null;
  const specialistProbs = await runRuntimeBatch({
    ort,
    session: runtime.session,
    inputName: runtime.inputName,
    labels: runtime.labels,
    tensorDataRows,
  });
  const specialist = summarizeProbabilitiesForLabels(specialistProbs, runtime.labels);
  return routeSpecialistPrediction(primary, specialist, route);
}

async function runRuntimeBatch({
  ort,
  session,
  inputName,
  labels,
  tensorDataRows,
}: {
  ort: typeof import("onnxruntime-web");
  session: import("onnxruntime-web").InferenceSession;
  inputName: string;
  labels: LabelsFile;
  tensorDataRows: Float32Array[];
}) {
  const rows = await Promise.all(
    tensorDataRows.map(async (tensorData) => {
      const inputData =
        labels.input_type === "flat" ? flattenTensor(tensorData, labels.n_features) : tensorData;
      const shape: number[] =
        labels.input_type === "flat"
          ? [1, inputData.length]
          : labels.input_type === "hand_landmarks"
            ? [1, labels.n_features ?? 132, labels.num_frames ?? 24]
            : [1, 3, labels.num_frames ?? 24, labels.frame_size ?? 160, labels.frame_size ?? 160];

      const input = new ort.Tensor("float32", inputData, shape);
      const results = await session.run({ [inputName]: input });
      const outKey = session.outputNames[0];
      const logits = results[outKey].data as Float32Array;
      return softmax(Array.from(logits));
    })
  );
  return averageProbabilities(rows);
}

function averageProbabilities(rows: number[][]) {
  if (rows.length === 0) return [];
  const out = new Array(rows[0].length).fill(0);
  rows.forEach((row) => row.forEach((p, i) => (out[i] += p)));
  return out.map((p) => p / rows.length);
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}
