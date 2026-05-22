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
manuscript/
  build_mdpi_submission.py
  manuscript.docx
submission/
  mdpi_ai_2026_05_22/
    FINAL_MANUSCRIPT_MDPI_AI.docx
    COVER_LETTER_MDPI_AI.docx
    SUBMISSION_METADATA.md
    SUBMISSION_PACKAGE_CHECKLIST.md
    figures_high_res/
    figures_high_res.zip
    supplementary/
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
  local_shap_contributions.csv
  figure_*.png
  generation_config.json
  generation_metadata.json
  split_manifest.csv
  label_scarcity_manifest.csv
DATASET_CARD.md
requirements.txt
LICENSE
CITATION.cff
```

## What Not to Upload

Do not upload the local `literature_review/` folder to a public repository. It contains Consensus reports, extracted report text, and downloaded scholarly PDFs that are local review materials, not redistributable benchmark artifacts.

The `.gitignore` file excludes those files, local caches, stale DOCX exports, and latent simulator internals that are not model inputs.

## Manuscript and Submission Package

The repository includes the current MDPI AI submission package under:

```text
submission/mdpi_ai_2026_05_22/
```

The final manuscript is:

```text
submission/mdpi_ai_2026_05_22/FINAL_MANUSCRIPT_MDPI_AI.docx
```

The same manuscript is also copied to:

```text
manuscript/manuscript.docx
```

The manuscript builder is:

```text
manuscript/build_mdpi_submission.py
```

The submitted references are grounded in the local literature-review paper set used during drafting. The `literature_review/` folder itself is intentionally not uploaded because it contains downloaded third-party PDFs and Consensus discovery reports.

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

The synthetic dataset generation code, benchmark configuration files, generated synthetic dataset, train/validation/test split files, label-scarcity split files, site-held-out split files, model-training scripts, threshold-selection scripts, evaluation scripts, result tables, and figure-generation artifacts are openly available at:

```text
https://github.com/purohit0208/synthetic-low-data-om-benchmark
```

The dataset is fully synthetic and does not contain real industrial, personal, proprietary, or institution-owned operational data.

## GenAI Disclosure

During the preparation of this manuscript/study, the author used ChatGPT and Gemini for research planning, code drafting, manuscript structuring, and language refinement. The author reviewed and edited all outputs, verified the methods, results, and references, and takes full responsibility for the content of the publication.

## Repository Status

This public reproducibility and submission package is hosted at:

```text
https://github.com/purohit0208/synthetic-low-data-om-benchmark
```

The repository intentionally contains benchmark code, generated synthetic data, split files, result tables, figures, and the current MDPI AI manuscript/submission package. It does not contain Consensus reports, downloaded scholarly PDFs, or local render-QA folders.
