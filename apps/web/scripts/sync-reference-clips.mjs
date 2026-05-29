import { copyFile, mkdir, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, "../../..");
const sourceDir = path.join(repoRoot, "ml/data/learner_samples/signer_a");
const outDir = path.join(repoRoot, "apps/web/public/references");

await mkdir(outDir, { recursive: true });
const files = await readdir(sourceDir);
let copied = 0;

for (const file of files) {
  if (!file.endsWith(".webm")) continue;
  const signId = file.split("_signer_", 1)[0];
  if (!signId) continue;
  await copyFile(path.join(sourceDir, file), path.join(outDir, `${signId}.webm`));
  copied += 1;
}

console.log(`Synced ${copied} reference clip(s) to ${path.relative(repoRoot, outDir)}`);
