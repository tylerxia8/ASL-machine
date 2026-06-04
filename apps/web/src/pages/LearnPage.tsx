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
    <div className="container wide-container">
      <Link to="/lobby">{"<-"} Lobby</Link>
      <section className="page-header">
        <p className="eyebrow">Course path</p>
        <h1>Intro ASL Learn</h1>
        <p>
          Weekly units, flashcards, phrases, and culture notes for a first college ASL course.
        </p>
      </section>

      <div className="metric-grid dashboard-metrics">
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
          <p className="eyebrow">Units</p>
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
                  <small>{progress.mastered}/{progress.total} mastered</small>
                </span>
                <span className="unit-progress-dot">{Math.round(progress.pct * 100)}%</span>
              </button>
            );
          })}
        </div>

        <div>
          <div className="card unit-focus-card">
            <p className="eyebrow">{selectedUnit.week}</p>
            <h2>{selectedUnit.title}</h2>
            <p>{selectedUnit.goal}</p>
            <div className="progress-track">
              <span style={{ width: `${Math.round(unitProgress(selectedUnit, mastery).pct * 100)}%` }} />
            </div>
            <div className="mode-grid compact-modes">
              <button className="mode-card mode-card-button" type="button" onClick={() => startUnitPractice(selectedUnit)}>
                <span className="mode-icon">A</span>
                <strong>Practice</strong>
                <small>Coach mode with camera feedback</small>
              </button>
              <button className="mode-card mode-card-button" type="button" onClick={() => startUnitPractice(selectedUnit, "shuffle")}>
                <span className="mode-icon">B</span>
                <strong>Quiz</strong>
                <small>Mixed order for recall</small>
              </button>
              <Link to="/phrases" className="mode-card">
                <span className="mode-icon">C</span>
                <strong>Phrases</strong>
                <small>Practice short sequences</small>
              </Link>
            </div>
          </div>

          <div className="card flashcard-panel">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Flashcards</p>
              <h3>Watch, recall, sign</h3>
            </div>
            <span className="muted-small">{flashIndex + 1}/{Math.max(selectedSigns.length, 1)}</span>
          </div>
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
          </div>

          <div className="card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Unit checklist</p>
                <h3>Signs and phrases</h3>
              </div>
            </div>
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
          </div>

          <div className="card">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Culture</p>
                <h3>Classroom notes</h3>
              </div>
            </div>
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
    </div>
  );
}
