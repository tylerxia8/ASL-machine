import type { InferenceSummary } from "./inference";

export type SpecialistRoute = {
  id: string;
  label?: string;
  releaseTag: string;
  allowedSigns: string[];
  minConfidence?: number;
  minMargin?: number;
  maxPrimaryConfidence?: number;
};

export type EnsembleConfig = {
  enabled?: boolean;
  primarySourceId?: string;
  specialists?: SpecialistRoute[];
};

export type RoutedSummary = InferenceSummary & {
  routedBy?: string;
  primaryPrediction?: InferenceSummary;
  specialistPrediction?: InferenceSummary;
};

export function routeSpecialistPrediction(
  primary: InferenceSummary,
  specialist: InferenceSummary,
  route: SpecialistRoute
): RoutedSummary {
  const allowed = new Set(route.allowedSigns);
  const minConfidence = route.minConfidence ?? 0.5;
  const minMargin = route.minMargin ?? 0;
  const maxPrimaryConfidence = route.maxPrimaryConfidence ?? 1;
  const canRoute =
    allowed.has(specialist.predictedLabel) &&
    specialist.confidence >= minConfidence &&
    specialist.margin >= minMargin &&
    primary.confidence <= maxPrimaryConfidence;

  if (!canRoute) {
    return primary;
  }

  return {
    ...specialist,
    routedBy: route.id,
    primaryPrediction: primary,
    specialistPrediction: specialist,
  };
}
