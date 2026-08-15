# Proof-space pilot

One-day pilot for the "empirical geometry of proof space" project
(Conway–Shipman extreme proofs, measured).

**Question:** when we embed 200 LLM-generated proofs of the infinitude of
primes — 5 techniques × 4 languages × 2 styles × 5 samples — does the
latent space organize by *mathematical technique* or by *surface form*?

## Run it

```bash
cd proofspace
python -m venv .venv && source .venv/bin/activate
pip install anthropic sentence-transformers umap-learn scikit-learn matplotlib

export ANTHROPIC_API_KEY=sk-ant-...
python generate.py          # ~200 API calls, roughly a dollar or two,
                            # resumable if interrupted
python analyze.py           # embeds locally on the M4, makes pilot_umap.png
```

First `analyze.py` run downloads the embedding model (~1 GB). Everything
runs locally on Apple silicon; no GPU rental needed.

## Reading the results

- `pilot_umap.png` — one projection, colored by technique / language /
  style. The eyeball test.
- Console scores — the real test, computed in full-dimensional space:
  ARI/NMI (does k-means recover the labeling?), silhouette (how separated
  is each labeling geometrically?), probe accuracy (is the label linearly
  decodable?).

| Outcome | Meaning | Next step |
|---|---|---|
| technique ≫ language | space encodes math | scale up: more theorems, tagging, Lean sample |
| language ≫ technique | space encodes words | try `--model BAAI/bge-m3`; if it persists, that's the negative-result paper |
| both high | entangled | the disentanglement version of the paper — arguably the most interesting |

## Deliberate pilot shortcuts (fix before scaling)

- No correctness filtering — some generated proofs may be wrong.
- Technique labels come from the *prompt*, not from reading the proof;
  the model may have ignored instructions. Spot-check a handful.
- Single generator model (circularity unaddressed at pilot scale).
- No control theorem yet.