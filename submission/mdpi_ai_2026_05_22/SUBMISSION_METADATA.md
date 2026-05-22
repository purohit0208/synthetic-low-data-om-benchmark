# Submission Metadata

Use this file as a consistency aid when completing the MDPI submission form. Verify all personal and institutional details before submission.

Journal: AI (MDPI)

Special issue: AI for Industrial Operation and Maintenance: Recognition Challenges with Limited Data Condition

Article type: Article

Title: A Synthetic Low-Data Benchmark for Trustworthy AI-Based Industrial Operation and Maintenance Decision Support

Author: Parth Purohit

Role: Wiss. Mitarbeiter | Research Associate

Affiliation: Technische Universität Braunschweig, Institut für Flugführung | Institute of Flight Guidance

Address: Hermann-Blenk-Str. 27, 38108 Braunschweig, Germany

ORCID: https://orcid.org/0009-0005-1547-8992

Corresponding author email: parth-yogeshbhai.purohit@tu-braunschweig.de

Website: https://www.tu-braunschweig.de/iff

Keywords: predictive maintenance; industrial operation and maintenance; synthetic benchmark; low-data learning; domain shift; uncertainty calibration; human-in-the-loop decision support; technician reports; explainable AI; reproducibility

Abstract:

Industrial operation and maintenance (O&M) AI is difficult to evaluate when labeled failures are scarce, observations are noisy, technician reports are incomplete, and deployment sites differ from training sites. This article presents a reproducible synthetic benchmark for these conditions in a generic multi-site industrial setting. The generator simulates 480 assets across six plants over 365 days, producing 175,200 shift-level records with hidden degradation, noisy structured observations, imperfect technician reports, leakage-audited features, chronological splits, random splits, and site-held-out splits. Seven calibrated baselines are evaluated: structured-only classifiers, TF-IDF plus Logistic Regression, frozen MiniLM sentence embeddings, and structured-plus-text fusion. Random Forest obtains the highest chronological-test PR-AUC (0.398), but at the default 0.5 threshold still misses 58.5% of positives. Random-split evaluation overstates generalization: Random Forest PR-AUC falls from 0.708 under random splitting to 0.439 under held-out-site testing, and false negative rate increases from 0.471 to 0.815. Validation-set routing thresholds can reduce workload, but only for models whose low-risk calibration tails satisfy the missed-fault constraint. The study is fully synthetic and does not claim real industrial deployment validation.

Funding statement: This research received no external funding. The study was conducted as independent research by the author.

Data availability statement: The synthetic dataset generation code, benchmark configuration files, generated synthetic dataset, train/validation/test split files, label-scarcity split files, site-held-out split files, model-training scripts, threshold-selection scripts, evaluation scripts, result tables, and figure-generation artifacts are available at https://github.com/purohit0208/synthetic-low-data-om-benchmark. The dataset is fully synthetic and does not contain real industrial, personal, proprietary, or institution-owned operational data.

Conflict of interest statement: The author declares no conflicts of interest.

Generative AI disclosure: During the preparation of this manuscript/study, the author used ChatGPT and Gemini for research planning, code drafting, manuscript structuring, and language refinement. The author reviewed and edited all outputs, verified the methods, results, and references, and takes full responsibility for the content of this publication.

Synthetic-data statement: No real industrial data, personal data, confidential project data, proprietary institutional material, or third-party operational records were used. All records were synthetically generated for controlled methodological evaluation.
