# Audit and Repair Notes

Date: 2026-05-21

## Current Verdict

The `D:\ExtraPapers` package is now more reproducible than the original Gemini output, but the manuscript is still not submission-ready. The code/results have been repaired and regenerated; the manuscript still needs to be rewritten against the regenerated artifacts and manually verified references.

## Repairs Completed

- Fixed the leakage-audit CLI bug in `src/generator.py`.
- De-duplicated leakage-audit findings.
- Added `outputs/generation_config.json`.
- Added `outputs/generation_metadata.json`.
- Added `outputs/split_manifest.csv`.
- Added `outputs/label_scarcity_manifest.csv`.
- Added fixed label-scarcity split files under `outputs/label_scarcity_splits/`.
- Added stratified random split files under `outputs/random_splits/`.
- Adjusted technician-report generation defaults; regenerated dataset has 24.67% report coverage.
- Changed MiniLM handling so real `sentence-transformers/all-MiniLM-L6-v2` is required by default. Mock embeddings require explicit `ALLOW_MOCK_MINILM=1`.
- Updated evaluation to use saved label-scarcity splits.
- Added random-split baseline output: `outputs/random_split_performance.csv`.
- Updated cross-site Table 6 to compare random-split performance against site-held-out performance.
- Expanded robustness outputs to all seven models; `outputs/exp3_robustness_results.csv` and Table 7 now contain 112 rows.
- Regenerated `outputs/local_case_indices.csv` with TP, FP, FN, and TN examples using a 0.5 diagnostic threshold.

## Verification Completed

- `python -m py_compile run_pipeline.py src\generator.py src\train.py src\routing.py src\evaluate.py src\explain.py`
- `python src\generator.py --test-leakage`
- Full rerun:

```powershell
python run_pipeline.py --days 365 --seed 42 --output-dir outputs
```

The full rerun completed successfully and used the real cached MiniLM model.

Regenerated dataset summary:

- Rows: 175,200
- Sites: 6
- Assets: 480
- `maintenance_required_7d` positive rate: 3.50%
- Technician-report coverage: 24.67%

## Remaining Problems

- `manuscript/manuscript.md` is stale relative to the regenerated outputs.
- The manuscript still contains placeholder public repository and Zenodo DOI claims.
- The references have not been manually verified.
- Related-work claims should be rebuilt from authentic sources only.
- The local explanation prose in the manuscript must be rewritten from actual saved case artifacts and, if needed, regenerated SHAP/local explanation outputs.
- The Data Availability, GenAI disclosure, author contribution, and conflict-of-interest statements must be checked before submission.

## Evidence Boundary

All results are synthetic-data results only. Do not claim real industrial deployment validation, certified safety assurance, production readiness, or actual maintenance-supervisor validation.
