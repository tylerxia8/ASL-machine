import { copyFileSync, mkdirSync, existsSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const webRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = join(webRoot, "../..");
const src = join(repoRoot, "ml/exports");
const dest = join(webRoot, "public/models");

mkdirSync(dest, { recursive: true });
for (const f of ["model.onnx", "labels.json", "model_meta.json"]) {
  const s = join(src, f);
  if (existsSync(s)) {
    copyFileSync(s, join(dest, f));
    console.log(`Copied ${f}`);
  } else {
    console.warn(`Missing ${s}`);
  }
}
