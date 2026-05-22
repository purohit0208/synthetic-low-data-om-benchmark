# Dataset Card

## Dataset Name

Synthetic Low-Data Industrial O&M Benchmark

## Dataset Type

Fully synthetic tabular and short-text benchmark dataset.

## Intended Use

This dataset is intended for reproducible evaluation of AI-based industrial operation and maintenance decision-support models under label scarcity, rare positives, noisy sensors, imperfect technician reports, missing data, calibration stress, and cross-site domain shift.

## Not Intended For

This dataset is not intended to validate real industrial deployment, certify safety, estimate real plant risk, or train a production maintenance model without external validation on real site data.

## Generation Summary

The generator simulates:

- six industrial sites;
- 480 assets;
- 365 days of operation;
- 175,200 shift-level records;
- latent asset degradation hidden from model features;
- structured sensor and operational observations;
- sparse technician-style reports;
- binary maintenance-required labels;
- optional fault-type and maintenance-priority labels.

## Generated Files

Core files:

- `outputs/dataset.csv`
- `outputs/dataset.parquet`
- `outputs/train.parquet`
- `outputs/val.parquet`
- `outputs/test.parquet`
- `outputs/data_dictionary.csv`
- `outputs/generation_config.json`
- `outputs/generation_metadata.json`
- `outputs/split_manifest.csv`
- `outputs/label_scarcity_manifest.csv`

Split files:

- `outputs/random_splits/`
- `outputs/site_splits/`
- `outputs/label_scarcity_splits/`

## Labels

Main label:

- `maintenance_required_7d`

Auxiliary labels:

- `fault_type`
- `maintenance_priority`

## Leakage Boundary

The generator uses latent degradation internally, but latent degradation is not a model input. The feature pipeline includes a leakage-audit function that fails loudly if forbidden latent or future-information columns are present before training.

The local file `outputs/latent_database.parquet` is excluded from public release by `.gitignore` because it contains simulator internals and is not part of the model-feature benchmark dataset.

## Ethical and Privacy Notes

No real industrial data, personal data, confidential project data, proprietary institutional material, or third-party operational records were used. All records are synthetically generated for controlled methodological evaluation.

## Known Limitations

- The degradation process is simplified.
- Technician reports are generated from templates and are not real technician language.
- Human-in-the-loop routing is simulated using labels, not validated with real maintenance supervisors.
- Results may not transfer to real plants without external validation.
- MiniLM is a general-purpose sentence encoder, not a domain-specific industrial maintenance language model.
