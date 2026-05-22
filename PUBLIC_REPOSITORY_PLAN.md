# Public Repository Plan

Date: 2026-05-21

Target journal: MDPI `AI`

Target special issue: `AI for Industrial Operation and Maintenance: Recognition Challenges with Limited Data Condition`

Repository URL selected on 2026-05-22:

`https://github.com/purohit0208/synthetic-low-data-om-benchmark`

## Repository Strategy

Use one public GitHub repository for the reproducibility package. Zenodo is optional, not required for the current plan.

MDPI `AI` requires enough experimental detail for reproducibility and asks authors to make full datasets available where possible. MDPI also recommends depositing data/code in trusted repositories and requires a Data Availability Statement. A public GitHub repository can satisfy the code/software part and can host this synthetic dataset package if the repository remains public and stable.

## What to Upload

Upload:

- `README.md`
- `AUDIT_AND_REPAIR_NOTES.md`
- `run_pipeline.py`
- `src/`
- `outputs/dataset.parquet`
- `outputs/data_dictionary.csv`
- `outputs/generation_config.json`
- `outputs/generation_metadata.json`
- `outputs/split_manifest.csv`
- `outputs/label_scarcity_manifest.csv`
- `outputs/train.parquet`
- `outputs/val.parquet`
- `outputs/test.parquet`
- `outputs/random_splits/`
- `outputs/site_splits/`
- `outputs/label_scarcity_splits/`
- `outputs/table_*.csv`
- `outputs/table_*.md`
- `outputs/exp*_*.csv`
- `outputs/random_split_performance.csv`
- `outputs/routing_sensitivity.csv`
- `outputs/global_feature_importance.csv`
- `outputs/tfidf_word_importance.csv`
- `outputs/local_case_indices.csv`
- `outputs/figure_*.png`
- `requirements.txt` or `environment.yml`
- `LICENSE`
- `CITATION.cff`
- `.gitignore`

Optional:

- `manuscript/` only after placeholder repository/DOI claims and stale result claims are removed.

## What Not to Upload

Do not upload:

- `literature_review/01.pdf` through `13.pdf` Consensus reports.
- Consensus report extracted text files.
- Downloaded full paper PDFs unless the license explicitly permits redistribution.
- Any credentials, local caches, `.venv`, `__pycache__`, model cache, or personal files.
- Placeholder Zenodo DOI or fake repository links.

## GitHub Stability Rules

Before submission:

1. Create a clean public repository at `https://github.com/purohit0208/synthetic-low-data-om-benchmark`.
2. Push a complete reproducibility package.
3. Create a release tag, for example `v1.0.0-mdpi-ai-submission`.
4. In the manuscript, cite the repository URL and the release tag or commit hash.
5. Do not rewrite the release history after submission.

## Draft Data Availability Statement

Use this only after the public repository exists:

> The synthetic dataset generation code, benchmark configuration files, generated synthetic dataset, train/validation/test split files, label-scarcity split files, site-held-out split files, model-training scripts, threshold-selection scripts, evaluation scripts, result tables, and figure-generation artifacts are openly available at [GitHub repository URL], release [release tag or commit hash]. The dataset is fully synthetic and does not contain real industrial, personal, proprietary, or institution-owned operational data.

## Repository Size Check

As of 2026-05-21, the local package is approximately 228 MB. No individual file is over 100 MB. This is technically compatible with GitHub, but the public repository should exclude literature reports and third-party paper PDFs.

Update 2026-05-22:

- A local Git repository has been initialized for audit only.
- Candidate public files after `.gitignore` are approximately 222.63 MB across 90 files.
- No candidate file is larger than 100 MB.
- The following are ignored and should remain excluded from public release: `literature_review/`, Python caches, stale/local DOCX exports, render QA folders, and `outputs/latent_database.parquet`.
