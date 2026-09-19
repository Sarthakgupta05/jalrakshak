/*
 * Parity test for the live demo.
 *
 * The claim on the demo page is that the browser runs the *same* model as
 * Python. This test proves it: it loads the built docs/index.html, extracts
 * the inlined model + fleet and the page's own featureVector() and predict()
 * source, executes them in Node, and compares the result to predictions
 * produced by scikit-learn.
 *
 * Run:  node tests/test_demo_parity.js
 */
const fs = require("fs");

const html = fs.readFileSync("docs/index.html", "utf8");
const grab = (id) => {
  const m = html.match(new RegExp(`<script id="${id}"[^>]*>([\\s\\S]*?)</script>`));
  if (!m) throw new Error("missing blob: " + id);
  return JSON.parse(m[1]);
};

const M = grab("modeldata");
const FLEET = grab("fleetdata");
const expected = JSON.parse(fs.readFileSync("tests/expected_predictions.json", "utf8"));

// pull the two functions verbatim out of the page so we test shipped code
const src = html.match(/\/\* ---- feature engineering[\s\S]*?\n}\n\/\* ---- ensemble/)[0]
  .replace(/\/\* ---- ensemble$/, "")
  + html.match(/function predict\(r\)\{[\s\S]*?\n}/)[0];

const COL = M.columns;
const IDX = {};
COL.forEach((c, i) => (IDX[c] = i));
const { featureVector, predict } = new Function(
  "M", "COL", "IDX", src + "\nreturn {featureVector, predict};"
)(M, COL, IDX);

let maxErr = 0;
const byId = {};
FLEET.forEach((r) => (byId[r.waterpoint_id] = r));
expected.ids.forEach((id, i) => {
  const got = predict(byId[id]);
  maxErr = Math.max(maxErr, Math.abs(got - expected.p[i]));
});

const TOL = 1e-6; // JSON stores thresholds at 9dp; error is ~1e-9 in practice
console.log(`compared ${expected.ids.length} water points`);
console.log(`max |javascript - scikit-learn| probability = ${maxErr.toExponential(3)}`);
if (maxErr > TOL) {
  console.error(`FAIL: exceeds tolerance ${TOL}`);
  process.exit(1);
}
console.log("PASS: the browser demo reproduces the Python model exactly.");
