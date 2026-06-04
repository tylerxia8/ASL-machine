import type { Mastery, ProgressSummary } from "./api";
import type { CourseUnit } from "./coursePlan";
import { unitProgress } from "./coursePlan";
import type { RecognitionFeedbackSummary } from "./recognitionFeedback";

const DAY_MS = 1000 * 60 * 60 * 24;

export function practiceStreak(mastery: Mastery[], now = Date.now()) {
  const days = new Set(
    mastery
      .map((row) => row.last_practiced_at)
      .filter(Boolean)
      .map((value) => new Date(value as string).toDateString())
  );
  let streak = 0;
  for (let offset = 0; offset < 60; offset++) {
    const day = new Date(now - offset * DAY_MS).toDateString();
    if (!days.has(day)) break;
    streak += 1;
  }
  return streak;
}

export function recommendedUnit(units: CourseUnit[], mastery: Mastery[]) {
  return [...units]
    .map((unit) => ({ unit, progress: unitProgress(unit, mastery) }))
    .sort((a, b) => a.progress.pct - b.progress.pct || b.progress.attempted - a.progress.attempted)[0]?.unit ?? units[0];
}

export function todayMissions({
  summary,
  feedback,
  dueCount,
}: {
  summary: ProgressSummary | null;
  feedback: RecognitionFeedbackSummary;
  dueCount: number;
}) {
  const attempts = summary?.total_attempts ?? 0;
  return [
    {
      id: "review",
      title: "Review signs",
      detail: dueCount > 0 ? `${Math.min(dueCount, 8)} signs due` : "Keep your streak warm",
      cta: "Start review",
      href: "/learn",
      tone: dueCount > 0 ? "retry" : "pass",
    },
    {
      id: "phrase",
      title: "Sign a phrase",
      detail: "Practice a short class-ready sequence",
      cta: "Open phrases",
      href: "/phrases",
      tone: "accent",
    },
    {
      id: "feedback",
      title: "Tune recognition",
      detail: feedback.rejected > 0 ? `${feedback.rejected} labeled misses` : `${attempts} attempts logged`,
      cta: "Practice weak signs",
      href: "/practice",
      tone: feedback.rejected > 0 ? "retry" : "accent",
    },
  ];
}
