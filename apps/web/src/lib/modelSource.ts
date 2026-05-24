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

function releaseToSource(tag: string, assets: { name: string; browser_download_url: string }[]): ModelSource | null {
  const m = assets.find((a) => a.name === "model.onnx");
  const l = assets.find((a) => a.name === "labels.json");
  const meta = assets.find((a) => a.name === "model_meta.json");
  if (!m || !l) return null;
  return {
    id: `release:${tag}`,
    label: `Release: ${tag}`,
    modelUrl: m.browser_download_url,
    labelsUrl: l.browser_download_url,
    metaUrl: meta?.browser_download_url ?? "/models/model_meta.json",
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
