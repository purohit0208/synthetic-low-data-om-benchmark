# Public Repository Release Checklist

Use this checklist before publishing the GitHub repository or inserting a repository link into the manuscript.

## Required Before Public Release

- [x] Confirm final repository name and owner: `purohit0208/synthetic-low-data-om-benchmark`.
- [x] Choose a final license for code and generated synthetic data: MIT.
- [x] Confirm that `literature_review/` is not committed.
- [x] Confirm that `manuscript/*.docx` is not committed unless regenerated from the corrected source.
- [x] Confirm that `outputs/latent_database.parquet` is not committed.
- [x] Run `python src/generator.py --test-leakage`.
- [x] Run `python run_pipeline.py --days 365 --seed 42 --output-dir outputs`.
- [x] Check that `outputs/generation_metadata.json` reports 175,200 rows, 480 assets, six sites, and report coverage inside the intended 20% to 40% range.
- [x] Check that `outputs/routing_sensitivity.csv` contains all seven models.
- [x] Create a public GitHub repository.
- [x] Push the cleaned package.
- [ ] Create a release tag, for example `v1.0.0-mdpi-ai-submission`.
- [x] Insert the real repository URL and release tag or commit hash in the manuscript Data Availability Statement.
- [ ] Regenerate the manuscript DOCX using the selected journal template.
- [ ] Perform a final formal reference metadata check against PDFs, DOI pages, publisher pages, and official documentation.

## Files Expected in Public Repository

- `README.md`
- `requirements.txt`
- `run_pipeline.py`
- `src/`
- `outputs/dataset.csv`
- `outputs/dataset.parquet`
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
- `outputs/data_dictionary.csv`
- `outputs/generation_config.json`
- `outputs/generation_metadata.json`
- `outputs/split_manifest.csv`
- `outputs/label_scarcity_manifest.csv`
- `manuscript/manuscript.md`
- `AUDIT_AND_REPAIR_NOTES.md`
- `PUBLIC_REPOSITORY_PLAN.md`
- `DATASET_CARD.md`
- `REPOSITORY_RELEASE_CHECKLIST.md`

## Files Not for Public Repository

- `literature_review/`
- `__pycache__/`
- `.venv/`
- `outputs/latent_database.parquet`
- stale DOCX exports
- local credentials, caches, or personal files
