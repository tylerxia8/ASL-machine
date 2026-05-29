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
  return calibration?.thresholds?.[signId] ?? { passThreshold: 0.9, retryThreshold: 0.7 };
}

export function confusionHint(
  calibration: RecognitionCalibration | null,
  promptLabel: string,
  predictedLabel: string
) {
  return calibration?.confusions?.[`${promptLabel}->${predictedLabel}`]?.message ?? null;
}
