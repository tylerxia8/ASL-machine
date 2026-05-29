import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { fetchSigns, type SignMeta } from "../lib/api";
import { downloadText } from "../lib/recognitionFeedback";

type ClipReview = {
  id: string;
  file: File;
  url: string;
  sign_id: string;
  signer_id: string;
  accepted: boolean;
};

function inferClip(file: File) {
  const stem = file.name.replace(/\.[^.]+$/, "");
  const match = stem.match(/^(.+)_(signer_[a-z0-9]+)_\d+$/i);
  return {
    sign_id: match?.[1] ?? "unknown",
    signer_id: match?.[2] ?? "unknown",
  };
}

export default function ReviewCapturesPage() {
  const [clips, setClips] = useState<ClipReview[]>([]);
  const [signs, setSigns] = useState<SignMeta[]>([]);
  const accepted = clips.filter((clip) => clip.accepted);
  const grouped = useMemo(() => {
    const rows: Record<string, number> = {};
    accepted.forEach((clip) => {
      rows[clip.sign_id] = (rows[clip.sign_id] ?? 0) + 1;
    });
    return Object.entries(rows).sort((a, b) => a[0].localeCompare(b[0]));
  }, [accepted]);

  useEffect(() => {
    return () => clips.forEach((clip) => URL.revokeObjectURL(clip.url));
  }, [clips]);

  useEffect(() => {
    fetchSigns(1).then(setSigns).catch(() => setSigns([]));
  }, []);

  const addFiles = (files: FileList | null) => {
    if (!files) return;
    const next = Array.from(files)
      .filter((file) => /\.(webm|mp4|mov|mkv|avi)$/i.test(file.name))
      .map((file) => {
        const inferred = inferClip(file);
        return {
          id: `${file.name}-${file.lastModified}-${file.size}`,
          file,
          url: URL.createObjectURL(file),
          accepted: true,
          ...inferred,
        };
      });
    setClips((current) => [...current, ...next]);
  };

  const toggle = (id: string) => {
    setClips((current) => current.map((clip) => clip.id === id ? { ...clip, accepted: !clip.accepted } : clip));
  };

  const updateClip = (id: string, patch: Partial<Pick<ClipReview, "sign_id" | "signer_id">>) => {
    setClips((current) => current.map((clip) => clip.id === id ? { ...clip, ...patch } : clip));
  };

  const exportManifest = () => {
    downloadText(
      `reviewed_capture_manifest_${Date.now()}.json`,
      JSON.stringify(
        {
          exported_at: new Date().toISOString(),
          accepted_count: accepted.length,
          rejected_count: clips.length - accepted.length,
          clips: accepted.map((clip) => ({
            filename: clip.file.name,
            sign_id: clip.sign_id,
            signer_id: clip.signer_id,
            size_bytes: clip.file.size,
          })),
        },
        null,
        2
      ),
      "application/json"
    );
  };

  return (
    <div className="container">
      <Link to="/lobby">← Lobby</Link>
      <h1>Review Captures</h1>

      <div className="card">
        <input
          className="input"
          type="file"
          accept="video/webm,video/mp4,video/quicktime,.mkv,.avi"
          multiple
          onChange={(event) => addFiles(event.target.files)}
        />
        <p style={{ color: "var(--muted)", margin: 0 }}>
          Select downloaded capture videos, reject bad takes, then export a clean manifest for retraining.
        </p>
      </div>

      {clips.length > 0 && (
        <>
          <div className="card" style={{ marginTop: "1rem" }}>
            <strong>{accepted.length}/{clips.length} clips accepted</strong>
            {grouped.length > 0 && (
              <p style={{ color: "var(--muted)" }}>
                {grouped.map(([sign, count]) => `${sign}: ${count}`).join(" · ")}
              </p>
            )}
            <button className="btn" onClick={exportManifest} disabled={accepted.length === 0}>
              Export clean manifest
            </button>
          </div>

          <div className="review-grid">
            {clips.map((clip) => (
              <div className="card" key={clip.id}>
                <video src={clip.url} controls playsInline style={{ width: "100%", background: "#000" }} />
                <p style={{ marginBottom: "0.25rem" }}>
                  <code>{clip.sign_id}</code> · {clip.signer_id}
                </p>
                <p style={{ color: "var(--muted)", fontSize: "0.8rem" }}>{clip.file.name}</p>
                <label>
                  <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Sign</span>
                  <select
                    className="input"
                    value={clip.sign_id}
                    onChange={(event) => updateClip(clip.id, { sign_id: event.target.value })}
                  >
                    {clip.sign_id === "unknown" && <option value="unknown">unknown</option>}
                    {signs.map((sign) => (
                      <option key={sign.sign_id} value={sign.sign_id}>
                        {sign.gloss}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span style={{ color: "var(--muted)", fontSize: "0.8rem" }}>Signer</span>
                  <input
                    className="input"
                    value={clip.signer_id}
                    onChange={(event) => updateClip(clip.id, { signer_id: event.target.value })}
                  />
                </label>
                <button className={clip.accepted ? "btn" : "btn btn-secondary"} onClick={() => toggle(clip.id)}>
                  {clip.accepted ? "Accepted" : "Rejected"}
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
