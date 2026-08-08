# P5 — Generative AI: GB Electricity Generation

Conditional spatial *simulation* of GB electricity generation, not forecasting: a probabilistic Transformer learns a distribution over half-hourly output conditioned only on a site's metadata (location, fuel type, rough capacity) and its position in the calendar year — no real time-series history is ever fed in as input. Sampling repeatedly produces synthetic generation traces; sampling many times produces an ensemble. See `Generative_AI_Analysis_Report.pdf` for the full write-up (dataset, model design, evaluation, ethics, limitations) — this README does not duplicate it.

## Repo structure

```
notebooks/
  exploration.ipynb        historical — early live look at the Elexon/OSUKED APIs, superseded by extraction.ipynb, not part of the pipeline proper
  extraction.ipynb         full resumable Elexon pull + OSUKED join + hull-based spatial train/val/test split
  dead_site_sweep.ipynb    fleet-wide detection/exclusion of "registered but not delivering" dead series, all splits
  granularity_eda.ipynb    half-hourly vs daily resolution EDA, train-split only — confirms half-hourly as the modelling resolution
  generative_model.ipynb   the graded artefact: model build + training (Tasks 1-3), generation + evaluation (Task 4), summary (Task 5)
  archive/                 pre-tightening originals of the five notebooks above, untouched — reference only, not part of the reproduction path

data/
  raw/generation/          per-BMU raw CSV pulls from Elexon (gitignored — source of truth, but too large/numerous for git)
  interim/
    extraction_manifest.csv        one row per (BMU, year) chunk pulled, with status — the resumability record for the full pull
    site_generation/               per-(site, fuel type) aggregated CSVs (gitignored, regenerable from raw/)
    site_generation_consolidated.parquet   consolidated training dataset (gitignored, regenerable — disposable cache, not source of truth)
  reference/                small tracked CSVs: OSUKED dictionary/fuel-type/location lookups, the Elexon BMU list, and the
                            pipeline's own outputs (site_fuel_type_coverage.csv, site_split_assignment.csv — the authoritative
                            train/val/test split)

models/
  generative_model/                 the model in use — best_checkpoint.pt (gitignored) + training_log.csv (tracked)
  generative_model_run1_baseline/   preserved diagnostic run (the overfitting run) — checkpoint.pt (gitignored) + training_log.csv (tracked)
```

## Environment

```
pip install -r requirements.txt
```

Python 3.12. A GPU is assumed for training (a full run is impractical on CPU) but not required for generation/evaluation alone — `generative_model.ipynb` falls back to CPU automatically (`torch.cuda.is_available()`) and a single forward pass per site is cheap either way.

## Reproduction

**Only the from-scratch path below is reproducible from a fresh clone.** The consolidated parquet and the trained checkpoints are deliberately gitignored (regenerable caches, not source of truth) — only the small reference CSVs and training logs are tracked. If you already have a local checkout with `data/interim/` and `models/` populated (e.g. this machine), you can skip straight to `generative_model.ipynb`; there is no first-class "fast path" for anyone starting from `git clone`.

**Full path, run order:**
1. `extraction.ipynb` — pulls the full Elexon BMU universe and joins the OSUKED dictionary. **Slow**: a full pull is multi-hour depending on network conditions, and resumable via `data/interim/extraction_manifest.csv` (safe to interrupt and re-run — already-`success` rows are skipped). Produces `site_generation_consolidated.parquet` and `data/reference/site_split_assignment.csv`.
2. `dead_site_sweep.ipynb` — must run immediately after any `extraction.ipynb` run, even a resumed one: extraction has no knowledge of this notebook's exclusions, so re-running extraction can resurrect previously-excluded dead series. Idempotent — safe to re-run any time.
3. `granularity_eda.ipynb` — EDA only, no file writes; informs the half-hourly resolution decision baked into `generative_model.ipynb`. Optional for pure reproduction, included for completeness.
4. `generative_model.ipynb` — builds, trains, generates, and evaluates the model.

`exploration.ipynb` is historical/superseded and not part of this run order.

## Data access

- **Elexon Insights Solution API** — public, no key required. Base URL: `https://data.elexon.co.uk/bmrs/api/v1`
- **OSUKED Power-Station-Dictionary** — public GitHub repo: https://github.com/OSUKED/Power-Station-Dictionary

Both are third-party public services outside this project's control — exact reproduction depends on their continued availability and data stability (e.g. new BMUs registering, historical revisions).

## Exact reproduction notes

- **The trained model is `models/generative_model/best_checkpoint.pt`** — the best-validation-loss epoch (165), not `checkpoint.pt` (the latest/resumable epoch, kept for training continuation only). `generative_model.ipynb` loads `best_checkpoint.pt` explicitly for all generation/evaluation.
- **Random seed 42** is used throughout for anything that needs to be reproducible rather than freshly random: the chunk-level train/holdout split for training diagnostics, ensemble sampling at generation time, and representative-site selection (fan charts, Step 4 week-level comparison). Re-running generation against `best_checkpoint.pt` with these seeds should reproduce the same figures, not a fresh random draw.

## What this README is not

Not a substitute for `Generative_AI_Analysis_Report.pdf`, and not itself part of the graded rubric — a practical aid for anyone (including future-you) wanting to rerun this.
