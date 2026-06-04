import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import ReferenceVideo from "../components/ReferenceVideo";
import { useAuth, getUserId } from "../lib/auth";
import { fetchMastery, fetchSigns, type Mastery, type SignMeta } from "../lib/api";
import {
  COURSE_UNITS,
  CULTURE_CARDS,
  dueReviewSigns,
  signsForUnit,
  unitProgress,
  type CultureCard,
  type CourseUnit,
} from "../lib/coursePlan";
import { readRecognitionFeedback, summarizeRecognitionFeedback } from "../lib/recognitionFeedback";
import { readPhraseLog, summarizePhraseLog } from "../lib/phraseLog";

export default function LearnPage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [signs, setSigns] = useState<SignMeta[]>([]);
  const [mastery, setMastery] = useState<Mastery[]>([]);
  const [selectedUnitId, setSelectedUnitId] = useState(COURSE_UNITS[0].id);
  const [flashIndex, setFlashIndex] = useState(0);

  useEffect(() => {
    fetchSigns(1).then(setSigns).catch(() => setSigns([]));
    fetchMastery(userId, auth.session?.access_token).then(setMastery).catch(() => setMastery([]));
  }, [auth.session?.access_token, userId]);

  const feedbackSummary = useMemo(() => summarizeRecognitionFeedback(readRecognitionFeedback()), []);
  const phraseSummary = useMemo(() => summarizePhraseLog(readPhraseLog()), []);
  const selectedUnit = COURSE_UNITS.find((unit) => unit.id === selectedUnitId) ?? COURSE_UNITS[0];
  const selectedSigns = signsForUnit(selectedUnit, signs);
  const currentFlash = selectedSigns[flashIndex % Math.max(selectedSigns.length, 1)];
  const dueSigns = dueReviewSigns(COURSE_UNITS, mastery, feedbackSummary);
  const cultureCards = selectedUnit.cultureCardIds
    .map((id) => CULTURE_CARDS.find((card) => card.id === id))
    .filter((card): card is CultureCard => Boolean(card));

  const startUnitPractice = (unit: CourseUnit, order: "weak_first" | "shuffle" = "weak_first") => {
    sessionStorage.setItem("practice_wave", "1");
    localStorage.setItem("practice_order", order);
    localStorage.setItem("practice_unit_filter", unit.signs.join(","));
    window.location.href = "/practice";
  };

  return (
    <div className="container">
      <Link to="/lobby">{"<-"} Lobby</Link>
      <h1>Intro ASL Learn</h1>
      <p style={{ color: "var(--muted)" }}>
        A college-friendly path for weekly review: learn signs, practice short phrases, revisit missed signs, and keep culture notes close.
      </p>

      <div className="metric-grid">
        <div>
          <span className="metric-label">Course units</span>
          <strong>{COURSE_UNITS.length}</strong>
        </div>
        <div>
          <span className="metric-label">Due for review</span>
          <strong>{dueSigns.length}</strong>
        </div>
        <div>
          <span className="metric-label">Phrase attempts</span>
          <strong>{phraseSummary.total}</strong>
        </div>
      </div>

      {dueSigns.length > 0 && (
        <div className="card" style={{ marginBottom: "1rem", borderColor: "var(--retry)" }}>
          <strong>Spaced review</strong>
          <p style={{ color: "var(--muted)" }}>
            These signs are not mastered yet, were missed locally, or have not been practiced recently.
          </p>
          <div className="phrase-strip">
            {dueSigns.map((signId) => (
              <span key={signId} className="tag-chip">
                {signId}
              </span>
            ))}
          </div>
          <div className="button-row">
            <Link to="/practice" className="btn">
              Review now
            </Link>
            <Link to="/progress" className="btn btn-secondary">
              See details
            </Link>
          </div>
        </div>
      )}

      <div className="learn-grid">
        <div>
          {COURSE_UNITS.map((unit) => {
            const progress = unitProgress(unit, mastery);
            return (
              <button
                key={unit.id}
                className={`unit-row ${unit.id === selectedUnit.id ? "unit-row-active" : ""}`}
                type="button"
                onClick={() => {
                  setSelectedUnitId(unit.id);
                  setFlashIndex(0);
                }}
              >
                <span>
                  <strong>{unit.week}</strong>
                  <span>{unit.title}</span>
                </span>
                <span>{progress.mastered}/{progress.total}</span>
              </button>
            );
          })}
        </div>

        <div className="card">
          <p style={{ color: "var(--muted)", marginTop: 0 }}>{selectedUnit.week}</p>
          <h2 style={{ marginTop: 0 }}>{selectedUnit.title}</h2>
          <p>{selectedUnit.goal}</p>
          <div className="button-row">
            <button className="btn" type="button" onClick={() => startUnitPractice(selectedUnit)}>
              Practice this unit
            </button>
            <button className="btn btn-secondary" type="button" onClick={() => startUnitPractice(selectedUnit, "shuffle")}>
              Quiz me
            </button>
            <Link to="/phrases" className="btn btn-secondary">
              Phrase practice
            </Link>
          </div>

          <h3>Flashcard</h3>
          {currentFlash ? (
            <div>
              <div className="flashcard-title">
                <code>{currentFlash.sign_id}</code>
                <span>{currentFlash.gloss}</span>
              </div>
              <ReferenceVideo signId={currentFlash.sign_id} />
              <div className="button-row">
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => setFlashIndex((i) => Math.max(0, i - 1))}
                  disabled={flashIndex === 0}
                >
                  Previous
                </button>
                <button
                  className="btn"
                  type="button"
                  onClick={() => setFlashIndex((i) => (i + 1) % selectedSigns.length)}
                  disabled={selectedSigns.length <= 1}
                >
                  Next card
                </button>
              </div>
            </div>
          ) : (
            <p style={{ color: "var(--muted)" }}>Loading unit signs...</p>
          )}

          <h3>Signs</h3>
          <div className="phrase-strip">
            {selectedUnit.signs.map((signId) => (
              <span key={signId} className="tag-chip">
                {signId}
              </span>
            ))}
          </div>

          <h3>Phrase goals</h3>
          <div className="phrase-strip">
            {selectedUnit.phrases.map((phrase) => (
              <span key={phrase} className="tag-chip">
                {phrase} ({phraseSummary.byPhrase[phrase]?.total ?? 0})
              </span>
            ))}
          </div>

          <h3>Culture notes</h3>
          <div className="review-grid">
            {cultureCards.map((card) => (
              <div key={card.id} className="mini-card">
                <strong>{card.title}</strong>
                <p>{card.body}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
