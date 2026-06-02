import { resolveSelectedSource } from "./modelSource";

export type SignThresholds = {
  passThreshold: number;
  retryThreshold: number;
  f1?: number;
  support?: number;
};

export type RecognitionCalibration = {
  model_version?: string;
  accuracy?: number;
  thresholds?: Record<string, SignThresholds>;
  confusions?: Record<string, { count?: number; message: string }>;
};

export type SignReliability = "strong" | "watch" | "weak";

let calibrationPromise: Promise<RecognitionCalibration> | null = null;

export function resetRecognitionCalibration() {
  calibrationPromise = null;
}

export async function loadRecognitionCalibration(): Promise<RecognitionCalibration> {
  if (!calibrationPromise) {
    calibrationPromise = resolveSelectedSource()
      .then(async (source) => {
        const url = source.calibrationUrl ?? "/models/recognition_calibration.json";
        const res = await fetch(url);
        if (!res.ok) return {};
        return (await res.json()) as RecognitionCalibration;
      })
      .catch(() => ({}));
  }
  return calibrationPromise;
}

export function thresholdsFor(
  calibration: RecognitionCalibration | null,
  signId: string
): Required<Pick<SignThresholds, "passThreshold" | "retryThreshold">> {
  const row = calibration?.thresholds?.[signId];
  const base = row ?? { passThreshold: 0.9, retryThreshold: 0.7 };
  if (reliabilityFor(row) === "weak") {
    return { passThreshold: 1.01, retryThreshold: Math.min(base.retryThreshold, 0.82) };
  }
  return { passThreshold: base.passThreshold, retryThreshold: base.retryThreshold };
}

export function confusionHint(
  calibration: RecognitionCalibration | null,
  promptLabel: string,
  predictedLabel: string
) {
  return calibration?.confusions?.[`${promptLabel}->${predictedLabel}`]?.message ?? null;
}

export function reliabilityFor(row: SignThresholds | undefined): SignReliability {
  if (!row) return "watch";
  const f1 = row.f1 ?? 1;
  const support = row.support ?? 999;
  if (f1 < 0.5 || support < 10) return "weak";
  if (f1 < 0.75 || support < 25) return "watch";
  return "strong";
}
