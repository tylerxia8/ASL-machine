import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useAuth, getUserId } from "../lib/auth";
import { fetchHint, trackEvent, type HintResponse } from "../lib/api";
import ReferenceVideo from "../components/ReferenceVideo";

type Phrase = {
  id: string;
  label: string;
  signs: string[];
};

const PHRASES: Phrase[] = [
  { id: "hello_name", label: "Hello, my name", signs: ["hello", "name"] },
  { id: "thank_you", label: "Thank you", signs: ["thank_you"] },
  { id: "where_help", label: "Where help?", signs: ["where", "help"] },
  { id: "nice_meet", label: "Nice to meet you", signs: ["nice", "meet"] },
  { id: "who_deaf", label: "Who is deaf?", signs: ["who", "deaf"] },
];
const PHRASE_LOG_KEY = "phrase_practice_log";

function readPhraseLog() {
  try {
    const raw = localStorage.getItem(PHRASE_LOG_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed as { phrase: string; outcome: string; ts?: number }[] : [];
  } catch {
    return [];
  }
}

export default function PhrasePage() {
  const auth = useAuth();
  const userId = getUserId(auth);
  const [index, setIndex] = useState(0);
  const [step, setStep] = useState(0);
  const [references, setReferences] = useState<Record<string, HintResponse>>({});
  const [log, setLog] = useState(readPhraseLog);

  const phrase = PHRASES[index];
  const currentSign = phrase.signs[step];
  const referenceList = useMemo(
    () => phrase.signs.map((sign) => references[sign]).filter(Boolean),
    [phrase.signs, references]
  );

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      phrase.signs.map((sign) =>
        fetchHint(sign, "phrase", userId).then((hint) => [sign, hint] as const).catch(() => null)
      )
    ).then((rows) => {
      if (cancelled) return;
      setReferences((prev) => {
        const next = { ...prev };
        rows.forEach((row) => {
          if (row) next[row[0]] = row[1];
        });
        return next;
      });
    });
    return () => {
      cancelled = true;
    };
  }, [phrase.id, phrase.signs, userId]);

  const mark = (outcome: "pass" | "retry") => {
    const nextLog = [...log.slice(-99), { phrase: phrase.id, outcome, ts: Date.now() }];
    setLog(nextLog);
    try {
      localStorage.setItem(PHRASE_LOG_KEY, JSON.stringify(nextLog));
    } catch {
      // Phrase practice still works if storage is blocked.
    }
    trackEvent("phrase_attempt", { phrase_id: phrase.id, outcome });
    setIndex((i) => (i + 1) % PHRASES.length);
    setStep(0);
  };

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1>Phrase Practice</h1>

      <div className="card">
        <p style={{ color: "var(--muted)", marginTop: 0 }}>Phrase {index + 1} of {PHRASES.length}</p>
        <h2 style={{ marginTop: 0 }}>{phrase.label}</h2>
        <div className="phrase-strip">
          {phrase.signs.map((sign, i) => (
            <button
              key={sign}
              className={i === step ? "btn" : "btn btn-secondary"}
              type="button"
              onClick={() => setStep(i)}
            >
              {sign}
            </button>
          ))}
        </div>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <strong>Current sign: <code>{currentSign}</code></strong>
        <ReferenceVideo signId={currentSign} />
        {references[currentSign] ? (
          <div className="hint-panel">
            <p>
              <strong>Handshape:</strong> {references[currentSign].handshape}<br />
              <strong>Movement:</strong> {references[currentSign].movement}<br />
              <strong>Location:</strong> {references[currentSign].location}<br />
              <strong>Framing:</strong> {references[currentSign].framing}
            </p>
          </div>
        ) : (
          <p style={{ color: "var(--muted)" }}>Loading reference...</p>
        )}
        <div className="button-row">
          <button className="btn btn-secondary" disabled={step === 0} onClick={() => setStep(step - 1)}>
            Previous sign
          </button>
          <button className="btn" disabled={step >= phrase.signs.length - 1} onClick={() => setStep(step + 1)}>
            Next sign
          </button>
        </div>
      </div>

      <div className="card" style={{ marginTop: "1rem" }}>
        <strong>Full sequence</strong>
        <ol>
          {referenceList.map((hint) => (
            <li key={hint.sign_id}>
              <code>{hint.sign_id}</code>: {hint.movement}
            </li>
          ))}
        </ol>
        <div className="button-row">
          <button className="btn" onClick={() => mark("pass")}>
            I signed the phrase
          </button>
          <button className="btn btn-secondary" onClick={() => mark("retry")}>
            Needs another pass
          </button>
        </div>
      </div>

      {log.length > 0 && (
        <div className="card" style={{ marginTop: "1rem" }}>
          <strong>Phrase log</strong>
          <ul>
            {log.slice(-6).map((row, i) => (
              <li key={`${row.phrase}-${i}`}>
                {row.phrase}: <span className={`status-${row.outcome}`}>{row.outcome}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
