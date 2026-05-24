/**
 * Manages which trained model the web app loads. By default uses the model
 * bundled at /models/model.onnx (the file synced from ml/exports/). Lets the
 * user temporarily point at a different release without re-downloading +
 * re-syncing locally.
 *
 * Persists the selection in localStorage so a hard refresh remembers it.
 */

const STORAGE_KEY = "asl_model_source";
const RELEASES_API = "https://api.github.com/repos/tylerxia8/ASL-machine/releases?per_page=20";
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export type ModelSource = {
  id: string;
  label: string;
  modelUrl: string;
  labelsUrl: string;
  metaUrl: string;
};

export const BUNDLED_SOURCE: ModelSource = {
  id: "bundled",
  label: "Bundled (apps/web/public/models)",
  modelUrl: "/models/model.onnx",
  labelsUrl: "/models/labels.json",
  metaUrl: "/models/model_meta.json",
};

// release-assets.githubusercontent.com doesn't send CORS headers, so the
// browser can't fetch model files directly from the GitHub Release URL.
// Route through the API's /model_proxy/{tag}/{file} endpoint instead, which
// streams the file with our own CORS-allowed origin.
function releaseToSource(tag: string, assets: { name: string }[]): ModelSource | null {
  const names = new Set(assets.map((a) => a.name));
  if (!names.has("model.onnx") || !names.has("labels.json")) return null;
  const proxy = (file: string) => `${API_URL}/model_proxy/${encodeURIComponent(tag)}/${file}`;
  return {
    id: `release:${tag}`,
    label: `Release: ${tag}`,
    modelUrl: proxy("model.onnx"),
    labelsUrl: proxy("labels.json"),
    metaUrl: names.has("model_meta.json") ? proxy("model_meta.json") : "/models/model_meta.json",
  };
}

let cachedReleases: ModelSource[] | null = null;

export async function listReleaseSources(): Promise<ModelSource[]> {
  if (cachedReleases) return cachedReleases;
  try {
    const res = await fetch(RELEASES_API);
    if (!res.ok) {
      cachedReleases = [];
      return cachedReleases;
    }
    const releases = (await res.json()) as { tag_name: string; assets: { name: string; browser_download_url: string }[] }[];
    const sources = releases
      .map((r) => releaseToSource(r.tag_name, r.assets))
      .filter((s): s is ModelSource => s !== null);
    cachedReleases = sources;
    return sources;
  } catch {
    cachedReleases = [];
    return cachedReleases;
  }
}

export function getSelectedSourceId(): string {
  return localStorage.getItem(STORAGE_KEY) || BUNDLED_SOURCE.id;
}

export function setSelectedSourceId(id: string) {
  localStorage.setItem(STORAGE_KEY, id);
}

export async function resolveSelectedSource(): Promise<ModelSource> {
  const id = getSelectedSourceId();
  if (id === BUNDLED_SOURCE.id) return BUNDLED_SOURCE;
  const releases = await listReleaseSources();
  return releases.find((r) => r.id === id) ?? BUNDLED_SOURCE;
}
