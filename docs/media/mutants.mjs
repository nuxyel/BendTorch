// Builds the README's mutation-testing figure as SVG, in the banner's visual language (warm paper, the embedded Bend
// ML Mono font): one square per fault that scripts/mutate.py plants, green when a law or a test catches it, amber with
// a halo when it survives by design. The counts and the amber squares' places (their order in MUTANTS) are read from
// scripts/mutate.py, so a new mutant only needs these three commands again:
//   node docs/media/mutants.mjs   -> docs/media/mutants.svg
//   brave --headless --screenshot=docs/media/mutants.png --window-size=1000,H docs/media/mutants.svg
//     (H is the height this script prints; the window must match the SVG's size)
//   magick docs/media/mutants.png -quality 90 docs/media/mutants.webp && rm docs/media/mutants.png
// (or: ffmpeg -i docs/media/mutants.png -quality 90 docs/media/mutants.webp). Then update the README's alt text.
// Without a browser: rsvg-convert -w 1000 -h H docs/media/mutants.svg -o docs/media/mutants.png, with the two fonts
// decompressed (woff2_decompress) into a directory fontconfig reads; librsvg ignores the embedded @font-face.
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const mutate = path.join(here, "..", "..", "scripts", "mutate.py");

// the last field of each MUTANTS tuple: "caught", or the reason the mutant survives
const PARSE = `
import ast, json, sys
tree = ast.parse(open(sys.argv[1]).read())
for n in tree.body:
    if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "MUTANTS" for t in n.targets):
        print(json.dumps([el.elts[-1].value for el in n.value.elts]))
`;
const expected = JSON.parse(execFileSync("python3", ["-I", "-c", PARSE, mutate], { encoding: "utf8" }));
const survives = expected.map((e, i) => (e === "caught" ? -1 : i)).filter((i) => i >= 0);
const N = expected.length, S = survives.length, K = N - S;
if (expected.some((e) => e !== "caught" && !e.startsWith("surviv"))) throw new Error("an expected field is neither caught nor survives");

const b64 = (f) => fs.readFileSync(path.join(here, "fonts", f)).toString("base64");
const FONT = `@font-face{font-family:"Bend ML Mono";font-weight:400;src:url(data:font/woff2;base64,${b64("BendMLMono-Regular.woff2")}) format("woff2")}
@font-face{font-family:"Bend ML Mono";font-weight:700;src:url(data:font/woff2;base64,${b64("BendMLMono-Bold.woff2")}) format("woff2")}`;

const C = { bg: "#f2eee7", edge: "#e9e3de", b1: "#b3afaa", b02: "#4d4a44", green: "#7e985d", amber: "#c3835b" };
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const T = (x, y, s, { size = 15, fill = C.b02, weight = 400 } = {}) =>
  `<text x="${x}" y="${y}" font-size="${size}" fill="${fill}" font-weight="${weight}" text-anchor="middle">${esc(s)}</text>`;
const WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
const word = (n) => WORDS[n] ?? String(n);

// the grid: the column count from 14 to 20 that leaves the fewest empty places in the last row (149 -> 15 x 10)
let cols = 14;
for (let c = 14; c <= 20; c++) if ((c - (N % c)) % c < (cols - (N % cols)) % cols) cols = c;
const rows = Math.ceil(N / cols), cell = 32, step = 40;
const W = 1000, gx = (W - (cols * step - (step - cell))) / 2, gy = 84;
const gridEnd = gy + rows * step - (step - cell);
const H = gridEnd + 160;

let squares = "";
for (let i = 0; i < N; i++) {
  const x = gx + (i % cols) * step, y = gy + Math.floor(i / cols) * step, live = expected[i] !== "caught";
  if (live) squares += `<rect x="${x - 6}" y="${y - 6}" width="${cell + 12}" height="${cell + 12}" rx="9" fill="none" stroke="${C.amber}" stroke-opacity="0.45" stroke-width="2"/>`;
  squares += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" rx="6" fill="${live ? C.amber : C.green}"/>`;
}

const TITLE = `${N} planted faults: ${K} caught, ${S} survive by design`;
const tally = `<text x="${W / 2}" y="${gridEnd + 46}" font-size="20" font-weight="700" text-anchor="middle" xml:space="preserve">` +
  `<tspan fill="${C.green}">${K} caught</tspan><tspan fill="${C.b1}"> · </tspan><tspan fill="${C.amber}">${S} survive by design</tspan></text>`;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(TITLE)}">
<title>${esc(TITLE)}</title>
<style>${FONT}
text{font-family:"Bend ML Mono",ui-monospace,Menlo,"SF Mono",Consolas,"Liberation Mono",monospace}</style>
<defs><radialGradient id="paper" cx="50%" cy="50%" r="75%"><stop offset="60%" stop-color="${C.bg}"/><stop offset="100%" stop-color="${C.edge}"/></radialGradient></defs>
<rect width="${W}" height="${H}" fill="url(#paper)"/>
${T(W / 2, 56, `${N} planted faults.`, { size: 38, weight: 700 })}
${squares}
${tally}
${T(W / 2, gridEnd + 96, "Every one behaves as expected.", { size: 32, fill: C.green, weight: 700 })}
${T(W / 2, gridEnd + 134, `one square, one fault planted on purpose · the ${word(S)} amber ones still compute the right result`, { size: 16, fill: C.b1 })}
</svg>
`;
fs.writeFileSync(path.join(here, "mutants.svg"), svg);
console.log(`mutants ok: ${N} faults, ${K} caught, ${S} survive (places ${survives.join(", ")}); ${W} x ${H}`);
