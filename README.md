# Synthetic Low-Data Industrial O&M Benchmark

This repository contains the code and reproducibility materials for:

**A Synthetic Low-Data Benchmark for Trustworthy AI-Based Industrial Operation and Maintenance Decision Support**

The study targets a generic industrial operation and maintenance (O&M) setting. It uses only self-created synthetic data and does not contain real industrial, personal, proprietary, or institution-owned operational records.

## Scope

The benchmark evaluates AI-based industrial O&M decision support under:

- scarce labels and rare maintenance/fault positives;
- noisy structured sensor and operational observations;
- sparse, imperfect technician-style reports;
- missing structured values and missing reports;
- cross-site domain shift across six simulated plants;
- calibrated human-in-the-loop routing with validation-selected thresholds.

The work is a synthetic benchmark and evaluation protocol. It does not claim real industrial deployment validation.

## Repository Structure

```text
run_pipeline.py
src/
  generator.py       Synthetic data generator and leakage audit
  train.py           Preprocessing, model training, calibration
  routing.py         Validation-set threshold optimization
  evaluate.py        Experiments, metrics, tables, and figures
  explain.py         SHAP and TF-IDF coefficient utilities
outputs/
  dataset.csv
  dataset.parquet
  train.parquet
  val.parquet
  test.parquet
  random_splits/
  site_splits/
  label_scarcity_splits/
  table_*.csv
  table_*.md
  figure_*.png
  generation_config.json
  generation_metadata.json
  split_manifest.csv
  label_scarcity_manifest.csv
manuscript/
  manuscript.md
AUDIT_AND_REPAIR_NOTES.md
PUBLIC_REPOSITORY_PLAN.md
DATASET_CARD.md
REPOSITORY_RELEASE_CHECKLIST.md
requirements.txt
LICENSE
CITATION.cff
```

## What Not to Upload

Do not upload the local `literature_review/` folder to a public repository. It contains Consensus reports, extracted report text, and downloaded scholarly PDFs that are local review materials, not redistributable benchmark artifacts.

The `.gitignore` file excludes those files, local caches, stale DOCX exports, and latent simulator internals that are not model inputs.

## Environment

Tested package versions are recorded in `requirements.txt`.

Install dependencies:

```bash
pip install -r requirements.txt
```

The MiniLM baseline uses the real Hugging Face model `sentence-transformers/all-MiniLM-L6-v2` as a frozen encoder. The code fails loudly by default if this model cannot be loaded. The deterministic mock fallback is only for explicitly labeled offline smoke tests:

```bash
set ALLOW_MOCK_MINILM=1
```

Do not use mock MiniLM outputs for manuscript results.

## Reproduce the Benchmark

Run the full pipeline:

```bash
python run_pipeline.py --days 365 --seed 42 --output-dir outputs
```

This regenerates:

- synthetic dataset and split files;
- learning-curve results;
- random-split and held-out-site comparisons;
- robustness results;
- validation-selected routing thresholds;
- alpha sensitivity analysis for routing;
- SHAP and TF-IDF explanation artifacts;
- tables and figures.

Verify the leakage audit:

```bash
python src/generator.py --test-leakage
```

Expected behavior: the test deliberately injects a forbidden latent column and the audit fails loudly.

## Data Availability Statement

Use this statement only after the public repository exists:

> The synthetic dataset generation code, benchmark configuration files, generated synthetic dataset, train/validation/test split files, label-scarcity split files, site-held-out split files, model-training scripts, threshold-selection scripts, evaluation scripts, result tables, and figure-generation artifacts are openly available at [GitHub repository URL], release [release tag or commit hash]. The dataset is fully synthetic and does not contain real industrial, personal, proprietary, or institution-owned operational data.

## GenAI Disclosure

During the preparation of this manuscript/study, the author used ChatGPT and Gemini for research planning, code drafting, manuscript structuring, and language refinement. The author reviewed and edited all outputs, verified the methods, results, and references, and takes full responsibility for the content of the publication.

## Current Release Status

This folder is intended to be released at:

```text
https://github.com/purohit0208/synthetic-low-data-om-benchmark
```

The final manuscript should cite release tag `v1.0.0-mdpi-ai-submission` or the exact commit hash.

The final public release is not complete until:

- a release tag or commit hash is frozen;
- the manuscript Data Availability Statement is updated with that URL/tag;
- the corrected Markdown manuscript is regenerated into the journal template.
