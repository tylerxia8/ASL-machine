import type { SignMeta } from "./api";
import type { Mastery } from "./api";
import type { RecognitionFeedbackSummary } from "./recognitionFeedback";

export type CourseUnit = {
  id: string;
  title: string;
  week: string;
  goal: string;
  signs: string[];
  phrases: string[];
  cultureCardIds: string[];
};

export type CultureCard = {
  id: string;
  title: string;
  body: string;
};

export const COURSE_UNITS: CourseUnit[] = [
  {
    id: "week1-greetings",
    title: "Greetings and Introductions",
    week: "Week 1",
    goal: "Start a basic ASL interaction with clear eye contact, greeting, and name signs.",
    signs: ["hello", "goodbye", "name", "nice", "meet", "deaf"],
    phrases: ["hello_name", "nice_meet"],
    cultureCardIds: ["eye-contact", "facial-expression"],
  },
  {
    id: "week2-courtesy",
    title: "Classroom Courtesy",
    week: "Week 2",
    goal: "Use common polite signs and ask for help during class practice.",
    signs: ["please", "thank_you", "sorry", "help", "friend"],
    phrases: ["thank_you", "please_help"],
    cultureCardIds: ["visual-attention", "not-english"],
  },
  {
    id: "week3-questions",
    title: "WH Questions",
    week: "Week 3",
    goal: "Recognize and produce simple question signs with beginner-friendly pacing.",
    signs: ["who", "what", "where", "how", "yes", "no"],
    phrases: ["where_help", "who_deaf", "what_name"],
    cultureCardIds: ["question-face", "turn-taking"],
  },
  {
    id: "week4-numbers-needs",
    title: "Numbers and Everyday Needs",
    week: "Week 4",
    goal: "Practice numbers 1-5 and common daily signs used in beginner dialogues.",
    signs: ["one", "two", "three", "four", "five", "water", "eat", "sleep"],
    phrases: ["water_please", "eat_sleep"],
    cultureCardIds: ["clarity-over-speed", "self-correction"],
  },
];

export const CULTURE_CARDS: CultureCard[] = [
  {
    id: "eye-contact",
    title: "Eye Contact",
    body: "In ASL, looking at your conversation partner is part of listening. Watch the face and upper body, not only the hands.",
  },
  {
    id: "facial-expression",
    title: "Facial Expression",
    body: "Facial expression carries grammar and meaning. For beginners, practice matching the reference face as well as the hand movement.",
  },
  {
    id: "visual-attention",
    title: "Getting Attention",
    body: "Use visual attention respectfully, such as a small wave in the signing space. Do not shout or grab someone.",
  },
  {
    id: "not-english",
    title: "ASL Is Its Own Language",
    body: "ASL is not English on the hands. Word order, facial grammar, and spatial structure can differ from spoken English.",
  },
  {
    id: "question-face",
    title: "Question Face",
    body: "WH questions often use brows lowered and a held final question sign. The face helps mark the sentence as a question.",
  },
  {
    id: "turn-taking",
    title: "Turn Taking",
    body: "ASL conversations are visual. Pause, keep your hands visible, and make sure your partner is looking before you continue.",
  },
  {
    id: "clarity-over-speed",
    title: "Clarity Before Speed",
    body: "Intro ASL students should prioritize clear handshape, location, and movement. Speed comes later.",
  },
  {
    id: "self-correction",
    title: "Self Correction",
    body: "It is normal to restart a sign or phrase. Clean self-correction is better than rushing through an unclear sign.",
  },
];

export function unitProgress(unit: CourseUnit, mastery: Mastery[]) {
  const bySign = new Map(mastery.map((row) => [row.sign_id, row]));
  const mastered = unit.signs.filter((signId) => bySign.get(signId)?.mastered).length;
  const attempted = unit.signs.filter((signId) => (bySign.get(signId)?.total_attempts ?? 0) > 0).length;
  return {
    mastered,
    attempted,
    total: unit.signs.length,
    pct: unit.signs.length ? mastered / unit.signs.length : 0,
  };
}

export function signsForUnit(unit: CourseUnit, signs: SignMeta[]) {
  const byId = new Map(signs.map((sign) => [sign.sign_id, sign]));
  return unit.signs.map((signId) => byId.get(signId)).filter(Boolean) as SignMeta[];
}

export function dueReviewSigns(units: CourseUnit[], mastery: Mastery[], feedback: RecognitionFeedbackSummary) {
  const bySign = new Map(mastery.map((row) => [row.sign_id, row]));
  const seen = new Set<string>();
  const due: string[] = [];
  for (const unit of units) {
    for (const signId of unit.signs) {
      if (seen.has(signId)) continue;
      seen.add(signId);
      const row = bySign.get(signId);
      const local = feedback.bySign[signId];
      const hasMisses = (local?.rejected ?? 0) > 0;
      const notMastered = !row?.mastered;
      const stale =
        row?.last_practiced_at &&
        Date.now() - new Date(row.last_practiced_at).getTime() > 1000 * 60 * 60 * 24 * 2;
      if (hasMisses || notMastered || stale) due.push(signId);
    }
  }
  return due.slice(0, 12);
}
