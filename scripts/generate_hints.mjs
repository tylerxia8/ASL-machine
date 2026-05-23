import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const vocab = readFileSync(join(root, "content/vocabulary.csv"), "utf8");
const lines = vocab.trim().split("\n").slice(1);
const hintsDir = join(root, "content/hints");
mkdirSync(hintsDir, { recursive: true });

const TEMPLATES = {
  greetings: ["relaxed handshape", "one clear movement", "chest-level neutral space"],
  numbers: ["correct number handshape", "hold steady", "center frame chest height"],
  colors: ["color-specific handshape", "correct shake/brush", "chin or neutral zone"],
  default: ["check handshape", "correct movement path", "correct spatial zone"],
};

const index = {};
for (const line of lines) {
  const [sign_id, gloss, category] = line.split(",");
  const t = TEMPLATES[category] || TEMPLATES.default;
  const hint = {
    sign_id,
    gloss,
    handshape: t[0],
    movement: t[1],
    location: t[2],
    orientation: "Match palm orientation to reference.",
    framing: "Hold sign still inside guide box for one second.",
    common_confusions: `If confused with similar ${category} signs, slow down and emphasize handshape.`,
  };
  writeFileSync(join(hintsDir, `${sign_id}.json`), JSON.stringify(hint, null, 2));
  index[sign_id] = hint;
}
writeFileSync(join(hintsDir, "_index.json"), JSON.stringify(index, null, 2));
console.log(`Generated ${lines.length} hints`);
