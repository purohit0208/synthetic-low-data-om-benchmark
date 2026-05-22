# A Synthetic Low-Data Benchmark for Trustworthy AI-Based Industrial Operation and Maintenance Decision Support

**Target Journal:** *AI* (MDPI)  
**Special Issue:** *AI for Industrial Operation and Maintenance: Recognition Challenges with Limited Data Condition*

---

## Abstract

Industrial operation and maintenance (O&M) AI is difficult to evaluate when labeled faults are scarce, fault events are rare, sensor observations are noisy, technician reports are incomplete, and deployment sites differ from training sites. This paper presents a reproducible synthetic benchmark for studying these conditions in a generic multi-site industrial setting. The benchmark simulates 480 assets across six plants over 365 days, producing 175,200 shift-level records with latent degradation, noisy structured observations, imperfect technician reports, maintenance labels, chronological splits, random splits, site-held-out splits, and leakage-audited feature matrices. We evaluate seven calibrated models spanning structured-only classifiers, TF-IDF plus Logistic Regression, frozen MiniLM sentence embeddings, and structured-plus-text fusion. Results on the synthetic benchmark show that structured Random Forest reaches the highest chronological-test PR-AUC (0.398), while text-only models remain weak (PR-AUC below 0.09). Random-split evaluation substantially overstates generalization for tree and fusion models: Random Forest PR-AUC drops from 0.708 under a random split to 0.439 under held-out-site evaluation, and its false negative rate increases from 0.471 to 0.815. Test-time sensor noise reduces PR-AUC sharply, whereas report perturbations have limited impact on fusion models because the trained classifiers rely primarily on structured sensor and maintenance-history features. A validation-set threshold optimizer selects routing thresholds before test evaluation; at $\alpha=0.05$, calibrated Random Forest auto-clears 71.9% of test cases while routing 0.35% of positives to auto-clear. The study is limited to synthetic data and does not claim real industrial deployment validation.

---

## 1. Introduction

Industrial systems operate under complex physical and environmental conditions, where asset wear and tear is inevitable. Predictive maintenance (PdM) leverages machine learning (ML) to identify early symptoms of component degradation, allowing operators to plan interventions before catastrophic failures occur. Despite its promise, deploying AI for industrial operation and maintenance (O&M) is hindered by significant practical limitations:

1. **Label Scarcity**: Real-world industrial sites are engineered for reliability. Consequently, critical failures are extremely rare, resulting in highly imbalanced datasets where positive (fault) labels constitute a tiny fraction of the data.
2. **Domain Shift**: Sensors and operational contexts vary across plants due to differences in ambient conditions, load factors, duty cycles, and installation setups. Models trained on historical data from one plant often experience severe performance degradation when deployed to another.
3. **Noisy and Unstructured Observations**: O&M datasets are inherently multimodal, combining numerical sensor streams (vibration, acoustic, temperature) with unstructured text reports logged by maintenance technicians. These inputs are frequently contaminated by sensor drift, missing data, and ambiguous report language.
4. **Safety and Trust Constraints**: Industrial operators need evidence about model reliability before using predictions in maintenance workflows. Missing a critical fault (a false negative) can create operational and safety risk. Thus, AI decision support should be calibrated, risk-bounded, and explainable.

To address these challenges, this study addresses six primary Research Questions (RQs):
* **RQ1**: How does label scarcity (1% to 100% labeled training data) impact the performance and calibration of structured, textual, and multimodal fusion models?
* **RQ2**: To what extent do site-specific domain shifts degrade model generalization and increase false negative rates on held-out plants?
* **RQ3**: How robust are structured and multimodal models to sensor noise, missing data, and technician report ambiguity?
* **RQ4**: Can modern sentence-transformer embeddings outperform classical TF-IDF text features in data-scarce and noisy O&M settings?
* **RQ5**: How does validation-set optimized threshold routing perform on the test set, and what workload reduction is achievable under strict safety constraints ($\alpha$)?
* **RQ6**: Are post-hoc explanations (SHAP and TF-IDF coefficients) operationally plausible and consistent with the simulated data-generation logic?

To resolve these questions, we establish a standardized synthetic benchmark featuring 480 assets across 6 plant sites simulated over 365 days (~175,200 records). The dataset undergoes a strict data leakage audit to prevent latent physical parameters from leaking into model features. We train a diverse classifier stack, calibrate model outputs using Platt scaling, and route decisions dynamically.

---

## 2. Related Work

### 2.1. Low-Data Predictive Maintenance and Synthetic Evaluation
Predictive maintenance commonly suffers from scarce run-to-failure examples, severe class imbalance, and temporal dependence. Hakami reports an extreme production-plant setting with very few failure observations relative to healthy observations, motivating methods that explicitly address data scarcity and imbalance rather than relying on accuracy alone [1]. Rare-event manufacturing studies similarly motivate data enrichment, imbalance-aware evaluation, and careful reporting of recall-oriented metrics [2]. Simulation and digital-twin work offers one response to scarce fault labels: synthetic or twin-assisted data can support controlled fault-diagnosis experiments, but simulated data must be treated with a clear validation boundary because simulated-to-real mismatch remains a central concern [3,4].

### 2.2. Domain Shift and Benchmark Splits
Industrial fault-diagnosis models can degrade when training and deployment conditions differ. Domain-generalization and domain-adaptation studies in bearing and machine fault diagnosis show that unseen operating conditions require evaluation protocols beyond random splits [5,6]. Calibration can also degrade under domain shift, especially when confidence estimates drive pseudo-labeling or decisions in fault-diagnosis pipelines [7]. Benchmark-design studies further warn that common fault-diagnosis datasets and split choices can overstate generalization if experimental protocols do not reflect deployment conditions [8]. This motivates the held-out-site protocol used in this study.

### 2.3. Technician Reports and Textual Maintenance Data
Maintenance text can contain useful operational information, but it is noisy, abbreviated, domain-specific, and often imbalanced. Cadavid et al. show that free-form maintenance text can support predictive-maintenance tasks, while also requiring imbalance handling and interpretability [9]. Sundaram and Zeid emphasize that maintenance work orders use technical language, shorthand, and domain-specific terminology that may limit generic NLP models [10]. Recent multimodal fault-diagnosis work also studies the integration of time-series and text, supporting the relevance of structured-plus-text baselines while leaving open how much text helps under low-label synthetic O&M conditions [11].

### 2.4. Calibration, Reject Options, and Human Routing
When predictions support operational decisions, probability quality matters in addition to ranking performance. Calibration surveys describe how to assess and improve predicted probabilities using metrics such as Brier score and calibration error [12]. Reject-option and selective-classification surveys provide the methodological basis for routing uncertain cases to human review rather than forcing every prediction into an automatic binary decision [13,14]. For safety-relevant cyber-physical systems, design-time decisions about thresholds, uncertainty, and human oversight must be explicit and validated for the intended operating context [15]. The routing layer in this paper follows this logic by choosing thresholds only on the validation set under a predefined missed-fault constraint.

### 2.5. Explainability and Reproducibility
Explainable predictive-maintenance surveys discuss SHAP, LIME, and related techniques as tools for model diagnostics and operational interpretation, while also warning that explanations are not causal proof [16,17]. Reproducible benchmark work emphasizes that datasets, splits, configurations, code, and evaluation scripts should be released together so that results can be audited and rerun [18,19]. This study therefore treats the generator, leakage audit, split manifests, trained-model protocol, threshold-selection code, and output tables as core research artifacts rather than incidental implementation details.

---

## 3. Synthetic Benchmark Design

We simulate a multi-site manufacturing environment to generate a benchmark dataset that captures selected O&M complexities while maintaining ground-truth control.

### 3.1. Physical Simulation and Latent Degradation
We model 480 distinct assets distributed across 6 sites (Plants 1 to 6). Each asset belongs to one of 6 types (e.g., Pump, Compressor, Gearbox) and contains specific components (e.g., Bearing, Seal, Valve). 

To simulate continuous wear, the health of each component is governed by a latent degradation variable $D_t \in [0, 1]$, modeled as a stochastic wear process with continuous drift and discrete shocks, following established condition-based maintenance and deterioration-modeling ideas [20,21]. The proposed physical degradation model is formulated as:
$$D_t = D_{t-1} + \Delta d_{base} \cdot \gamma_{site} \cdot (1 + \beta \cdot \text{load}) + \eta_t + \theta_t$$
where $\Delta d_{base}$ is the baseline wear rate, $\gamma_{site}$ is a site-specific scaling factor, $\beta$ represents the sensitivity to asset load, $\eta_t \sim \mathcal{N}(0, \sigma^2)$ is continuous drift noise, and $\theta_t$ represents discrete shock events modeled as a Poisson process. When $D_t \ge 0.75$, a physical fault is triggered. If maintenance is performed, $D_t$ resets to 0. This formulation is introduced specifically in this benchmark study to model multi-site environmental variability.

### 3.2. Observed Modalities and Technician Report Generation
Instead of observing $D_t$ directly, the system outputs noisy sensor measurements (vibration RMS, temperature deviation, motor current, acoustic level). We also simulate qualitative technician reports using template-based generation. The report text reflects the underlying fault state with injected noise: 15% of reports contain template ambiguity or omission, and 5% contain misleading report flags (e.g., reporting "normal operations" when a fault is occurring).

### 3.3. Leakage Auditing and Chronological Splits
A strict data leakage audit is executed before training to ensure that latent variables (such as $D_t$ or future maintenance flags) are not accessible to the models. The dataset is partitioned chronologically into training (60%), validation (20%), and test (20%) sets to mimic forward-time evaluation. Additionally, we generate site-held-out splits where one plant is completely omitted from training and validation, and used exclusively for testing.

The complete simulated dataset schema is detailed in **Table 1**.

| Feature Group | Column Name | Type | Description |
|:---|:---|:---|:---|
| Metadata | `date` | DateTime | Timestamp of record |
| | `asset_id` | Categorical | Unique asset identifier |
| | `site_id` | Categorical | Plant location (Plant 1 to 6) |
| Asset Details | `asset_type` | Categorical | e.g., Pump, Compressor, Gearbox |
| | `component_type` | Categorical | e.g., Bearing, Seal, Valve |
| | `age` | Numeric | Asset age in days since installation |
| | `operating_hours` | Numeric | Cumulative asset operating hours |
| Sensors | `vibration_rms` | Numeric | Vibration Root Mean Square (sensor value) |
| | `vibration_kurtosis` | Numeric | Vibration Kurtosis (sensor value) |
| | `ambient_temp` | Numeric | Ambient plant temperature |
| | `temp_deviation` | Numeric | Temperature deviation from ambient |
| | `acoustic_level` | Numeric | Acoustic noise level in dB (sensor value) |
| | `motor_current` | Numeric | Motor current draw in Amperes (sensor value) |
| | `pressure_deviation` | Numeric | Pressure deviation in bar (sensor value) |
| | `flow_rate_deviation` | Numeric | Fluid flow rate deviation in m^3/h |
| Operational | `load_factor` | Numeric | Daily load factor (0.5 to 1.1) |
| | `duty_cycle` | Numeric | Daily shift duration (hours per day) |
| | `previous_fault_count`| Numeric | Cumulative count of historical asset faults |
| | `time_since_last_maintenance`| Numeric | Days since last maintenance intervention |
| | `shift_type` | Categorical | Active shift type (Day, Swing, Night) |
| Textual | `technician_report`| Text | Unstructured qualitative maintenance logs |
| Target Label | `maintenance_required_7d` | Binary | 1 if maintenance occurs in next 7 days, else 0 |
| Latent Targets | `fault_type` | Categorical | Multiclass fault type (Normal, Bearing Wear, etc.) |
| | `maintenance_priority` | Categorical | Ordinal priority (Normal, Low, Medium, High, Critical) |

**Table 1:** Synthetic Dataset Schema and Feature Groups.

### 3.4. Target Variable and Class Distribution
The target variable is `maintenance_required_7d` (1 if maintenance is scheduled within the next 7 days, indicating an active or imminent fault). The generated dataset consists of 175,200 records. The class distributions are shown in **Table 2**.

| Fault Type | Count | Percentage (%) | Priority Label | Count | Percentage (%) |
|:---|---:|---:|:---|---:|---:|
| Normal | 174,701 | 99.715% | Normal | 111,079 | 63.401% |
| Bearing Wear | 140 | 0.080% | Low | 57,199 | 32.648% |
| Overheating | 117 | 0.067% | Medium | 6,423 | 3.666% |
| Misalignment | 90 | 0.051% | High | 476 | 0.272% |
| Seal Leakage | 75 | 0.043% | Critical | 23 | 0.013% |
| Blockage | 50 | 0.029% | | | |
| Sensor Drift | 27 | 0.015% | | | |

**Table 2:** Fault Types and Maintenance Priority Class Distribution.

#### Clarifying Note on Fault Rates and Targets
A comparison of Table 2 with the main target label reveals an apparent discrepancy: Table 2 lists 499 active fault records (0.285% of the dataset), while the `maintenance_required_7d` positive label has an overall rate of approximately 3.50% in the full generated dataset and 6.51% in the chronological test split. This difference is intentional and represents a core characteristic of preventive maintenance systems:
1. The **Fault Type** column represents the instantaneous presence of a severe physical fault ($D_t \ge 0.75$) where component failure has already occurred.
2. The **Binary Target** (`maintenance_required_7d`) is a predictive window. Maintenance is scheduled preemptively when degradation is detected early ($D_t \ge 0.50$ or high wear symptoms), preventing severe faults before they occur. Thus, the prediction target captures a wider, proactive operational window than the instantaneous severe-fault label.

---

## 4. Experimental Scenarios and Controlled Variables

To thoroughly stress-test the trustworthy AI stack, we define four distinct experimental scenarios as shown in **Table 3**.

| Experimental Scenario | Controlled Variable(s) | Evaluation Metrics |
|:---|:---|:---|
| **Experiment 1:** Label Scarcity Learning Curve | Training label fraction: 1%, 5%, 10%, 25%, 50%, 100% | PR-AUC, ROC-AUC, Brier score, ECE |
| **Experiment 2:** Cross-Site Domain Shift | Held-out plant site for testing: Plant_1 to Plant_6 | Drop in PR-AUC, FNR, ECE |
| **Experiment 3:** Robustness under Sensor & Report Noise | Sensor noise multiplier (1.0x to 3.0x), missing sensors (0% to 40%), missing reports (0% to 75%), ambiguity level | PR-AUC degradation rate, False Negative Rate (FNR) inflation |
| **Experiment 4:** Constrained Human-in-the-Loop Routing | Routing safety constraint limit $\alpha$: 0.01, 0.03, 0.05, 0.10 | Auto-clear rate, Workload reduction, Missed Critical Fault Rate |

**Table 3:** Experimental Scenarios and Controlled Variables.

---

## 5. Predictive Modeling and Calibrated Routing Methods

We evaluate seven models across structured, textual, and multimodal fusion architectures. The model stack and modalities are detailed in **Table 4**.
TF-IDF features are extracted with scikit-learn's `TfidfVectorizer` [22]. MiniLM text representations use the frozen `sentence-transformers/all-MiniLM-L6-v2` sentence encoder, which maps sentences to 384-dimensional embeddings [23]. The structured baselines include Random Forest [24] and XGBoost [25], and post-hoc global feature analysis uses SHAP [26]. Platt-style sigmoid calibration is applied on validation data [27].

| Model Identifier | Model Type | Input Modalities | Calibration Method |
|:---|:---|:---|:---|
| `lr_struct` | Logistic Regression | Structured sensor & asset metadata | Platt Scaling (Sigmoid Val-fit) |
| `rf_struct` | Random Forest | Structured sensor & asset metadata | Platt Scaling (Sigmoid Val-fit) |
| `xgb_struct` | XGBoost Classifier | Structured sensor & asset metadata | Platt Scaling (Sigmoid Val-fit) |
| `lr_tfidf` | Logistic Regression | Raw technician reports (TF-IDF tokens) | Platt Scaling (Sigmoid Val-fit) |
| `lr_minilm` | Logistic Regression | Raw technician reports (MiniLM embeddings) | Platt Scaling (Sigmoid Val-fit) |
| `fusion_tfidf` | XGBoost Classifier | Concatenated Structured & TF-IDF features | Platt Scaling (Sigmoid Val-fit) |
| `fusion_minilm`| XGBoost Classifier | Concatenated Structured & MiniLM features | Platt Scaling (Sigmoid Val-fit) |

**Table 4:** Model Stack and Input Modalities.

### 5.1. Probability Calibration using Platt Scaling
Machine learning classifiers often generate uncalibrated outputs that do not represent true probabilities. In safety-critical O&M, probability calibration is essential. We apply Platt scaling (sigmoid calibration) on the validation set:
$$P(y=1 | f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$
where $f(x)$ is the uncalibrated prediction output of the model, and $A$ and $B$ are scalar parameters fitted using maximum likelihood on the validation set.

### 5.2. Safety-Constrained Threshold Routing Optimization
Calibrated probabilities are used to route maintenance decisions. Instead of a single static threshold, we define a lower threshold $t_{low}$ and an upper threshold $t_{high}$ ($t_{low} < t_{high}$):
1. **Auto-Clear ($P(y=1|x) \le t_{low}$)**: The asset is classified as healthy; maintenance is deferred.
2. **Human Review ($t_{low} < P(y=1|x) < t_{high}$)**: The prediction is ambiguous; case is routed to human review.
3. **Urgent Inspection ($P(y=1|x) \ge t_{high}$)**: High probability of fault; triggers immediate inspection.

The Missed Critical Fault Rate (MCFR) is defined as the fraction of true positive cases that are incorrectly auto-cleared:
$$\text{MCFR} = \frac{\sum [y_i = 1 \land P(y=1|x_i) \le t_{low}]}{\sum [y_i = 1]}$$
We optimize $t_{low}$ and $t_{high}$ on the validation set to maximize Workload Reduction (fraction of cases auto-cleared) subject to a safety ceiling $\alpha$:
$$\max_{t_{low}, t_{high}} \frac{\sum [P(y=1|x_i) \le t_{low}]}{N} \quad \text{subject to} \quad \text{MCFR} \le \alpha$$
Once optimized, the thresholds are frozen and evaluated on the test set.

---

## 6. Trustworthy AI Benchmark Architecture

The orchestrating pipeline maps simulation files to structural pipelines, leakage checks, validation loops, and explainable models.

![Figure 1: Industrial O&M Trustworthy AI Pipeline Architecture.](outputs/figure_1_pipeline.png)

![Figure 2: Physical Latent Component Degradation and Vibration RMS Sensor Output over Time.](outputs/figure_2_degradation.png)

---

## 7. Empirical Results

### 7.1. Label Scarcity Learning Curves (Experiment 1)
To address label scarcity (RQ1), we evaluated the models by training them on subsets of training data representing fractions of 1%, 5%, 10%, 25%, 50%, and 100%. The test-set results for PR-AUC and Brier Score are detailed in **Table 5**.

| Fraction | Model | Test PR-AUC | Test Brier Score | Test ECE |
|:---|:---|---:|---:|---:|
| **1%** | `lr_struct` | 0.2381 | 0.0578 | 0.0280 |
| | `rf_struct` | 0.3852 | 0.0513 | 0.0273 |
| | `xgb_struct` | 0.2916 | 0.0607 | 0.0398 |
| | `lr_tfidf` | 0.0866 | 0.0623 | 0.0385 |
| | `lr_minilm` | 0.0775 | 0.0623 | 0.0384 |
| | `fusion_tfidf` | 0.2992 | 0.0608 | 0.0381 |
| | `fusion_minilm` | 0.2846 | 0.0621 | 0.0404 |
| **5%** | `lr_struct` | 0.2974 | 0.0557 | 0.0202 |
| | `rf_struct` | 0.3905 | 0.0556 | 0.0341 |
| | `xgb_struct` | 0.3204 | 0.0581 | 0.0344 |
| | `lr_tfidf` | 0.0798 | 0.0623 | 0.0385 |
| | `lr_minilm` | 0.0838 | 0.0622 | 0.0384 |
| | `fusion_tfidf` | 0.3237 | 0.0574 | 0.0335 |
| | `fusion_minilm` | 0.3344 | 0.0590 | 0.0359 |
| **10%** | `lr_struct` | 0.2976 | 0.0589 | 0.0309 |
| | `rf_struct` | 0.3969 | 0.0553 | 0.0352 |
| | `xgb_struct` | 0.3403 | 0.0636 | 0.0426 |
| | `lr_tfidf` | 0.0834 | 0.0623 | 0.0385 |
| | `lr_minilm` | 0.0886 | 0.0621 | 0.0383 |
| | `fusion_tfidf` | 0.3454 | 0.0617 | 0.0396 |
| | `fusion_minilm` | 0.3245 | 0.0633 | 0.0417 |
| **25%** | `lr_struct` | 0.3061 | 0.0642 | 0.0440 |
| | `rf_struct` | 0.4021 | 0.0564 | 0.0354 |
| | `xgb_struct` | 0.3228 | 0.0660 | 0.0458 |
| | `lr_tfidf` | 0.0872 | 0.0622 | 0.0384 |
| | `lr_minilm` | 0.0890 | 0.0621 | 0.0383 |
| | `fusion_tfidf` | 0.3120 | 0.0713 | 0.0506 |
| | `fusion_minilm` | 0.3192 | 0.0694 | 0.0490 |
| **50%** | `lr_struct` | 0.3036 | 0.0648 | 0.0460 |
| | `rf_struct` | 0.3995 | 0.0574 | 0.0378 |
| | `xgb_struct` | 0.3402 | 0.0623 | 0.0440 |
| | `lr_tfidf` | 0.0864 | 0.0622 | 0.0384 |
| | `lr_minilm` | 0.0883 | 0.0621 | 0.0382 |
| | `fusion_tfidf` | 0.3345 | 0.0643 | 0.0464 |
| | `fusion_minilm` | 0.3192 | 0.0650 | 0.0471 |
| **100%** | `lr_struct` | 0.3027 | 0.0667 | 0.0512 |
| | `rf_struct` | 0.3981 | 0.0594 | 0.0426 |
| | `xgb_struct` | 0.3292 | 0.0613 | 0.0424 |
| | `lr_tfidf` | 0.0894 | 0.0621 | 0.0384 |
| | `lr_minilm` | 0.0889 | 0.0621 | 0.0383 |
| | `fusion_tfidf` | 0.3244 | 0.0629 | 0.0462 |
| | `fusion_minilm` | 0.3314 | 0.0620 | 0.0470 |

**Table 5:** Main Learning Curve Results under Label Scarcity (Experiment 1).

The results show that Random Forest (`rf_struct`) has the strongest low-label behavior in this synthetic setting, obtaining a PR-AUC of 0.3852 with 1% of the training labels and 0.3981 with the full labeled training set. XGBoost and fusion models also retain useful ranking performance at 1% labels, with PR-AUC values between 0.2846 and 0.2992.

We observe minor non-monotonic fluctuations in the learning curves for XGBoost and Fusion models at intermediate fractions (for example, the test PR-AUC of `xgb_struct` drops slightly from 0.3403 at 10% labels to 0.3228 at 25% labels). This behavior is a common characteristic of learning curves under severe class imbalance; stratified random subsampling at intermediate fractions can introduce variance in model initialization and optimization thresholds, which temporarily impacts precision-recall characteristics before the model converges on the full dataset.

Conversely, text-only baselines (`lr_tfidf` and `lr_minilm`) remain below 0.09 PR-AUC in most settings, showing that the synthetic technician reports alone are insufficient indicators without sensor and maintenance-history data.

![Figure 3: PR-AUC Learning Curves Across Training Label Fractions.](outputs/figure_3_learning_curves.png)

### 7.2. Baseline Predictive Performance (100% Labeled Training Set)
The baseline test-set results (using a default decision threshold of 0.5 and 100% labels) are presented in **Table 6**.

| Model | ROC-AUC | PR-AUC | F1-Score | Balanced Accuracy | Precision | Recall (Sensitivity) | FNR | FPR | Brier Score | ECE |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lr_struct` | 0.9110 | 0.3027 | 0.3711 | 0.6861 | 0.3231 | 0.4358 | 0.5642 | 0.0403 | 0.0667 | 0.0512 |
| `rf_struct` | 0.9275 | 0.3981 | 0.3752 | 0.6797 | 0.3426 | 0.4148 | 0.5852 | 0.0414 | 0.0594 | 0.0426 |
| `xgb_struct` | 0.9175 | 0.3292 | 0.3233 | 0.6377 | 0.3246 | 0.3219 | 0.6781 | 0.0472 | 0.0613 | 0.0424 |
| `lr_tfidf` | 0.5422 | 0.0894 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0651 | 0.0621 | 0.0384 |
| `lr_minilm` | 0.5539 | 0.0889 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.0651 | 0.0621 | 0.0383 |
| `fusion_tfidf` | 0.9182 | 0.3244 | 0.3361 | 0.6488 | 0.3253 | 0.3478 | 0.6522 | 0.0456 | 0.0629 | 0.0462 |
| `fusion_minilm`| 0.9180 | 0.3314 | 0.3345 | 0.6464 | 0.3277 | 0.3415 | 0.6585 | 0.0460 | 0.0620 | 0.0470 |

**Table 6:** Main Predictive Performance Results on Test Set (100% Labels).

The structured-only Random Forest (`rf_struct`) achieves the highest chronological-test PR-AUC (0.3981), outperforming XGBoost (`xgb_struct`, 0.3292) and Logistic Regression (`lr_struct`, 0.3027). Text-only models (`lr_tfidf` and `lr_minilm`) perform poorly on their own (PR-AUC below 0.09) and predict no positives at the default 0.5 threshold. Multimodal fusion models (`fusion_tfidf` and `fusion_minilm`) remain competitive with XGBoost but do not exceed the structured-only Random Forest. In this synthetic configuration, technician reports add limited incremental signal beyond structured sensor and maintenance-history variables.

### 7.3. Cross-Site Generalization (Experiment 2)
To evaluate domain shift, we compared a random train/validation/test split against a held-out-site protocol in which models are trained on five sites and tested on the sixth unseen site. **Table 7** summarizes the random-split baseline and the mean performance across six held-out-site folds.

| Model | Random-Split PR-AUC | Cross-Site Mean PR-AUC | PR-AUC Drop | Random-Split FNR | Cross-Site Mean FNR | FNR Inflation |
|:---|---:|---:|---:|---:|---:|---:|
| `lr_struct` | 0.3625 | 0.3482 | 0.0143 | 0.8707 | 0.7121 | -0.1586 |
| `rf_struct` | 0.7082 | 0.4389 | 0.2693 | 0.4707 | 0.8152 | 0.3445 |
| `xgb_struct` | 0.7358 | 0.3697 | 0.3661 | 0.4978 | 0.8595 | 0.3617 |
| `lr_tfidf` | 0.0523 | 0.0547 | -0.0023 | 1.0000 | 1.0000 | 0.0000 |
| `lr_minilm` | 0.0528 | 0.0572 | -0.0044 | 1.0000 | 1.0000 | 0.0000 |
| `fusion_tfidf` | 0.7154 | 0.3800 | 0.3354 | 0.5130 | 0.8552 | 0.3421 |
| `fusion_minilm` | 0.7149 | 0.3816 | 0.3333 | 0.5098 | 0.8557 | 0.3459 |

**Table 7:** Random-Split versus Held-Out-Site Generalization Results.

The random split substantially overstates the deployment performance of tree-based structured and fusion models. Random Forest drops by 0.269 PR-AUC points under site-held-out evaluation, while XGBoost drops by 0.366 points. The FNR inflation is also operationally important: Random Forest, XGBoost, and both fusion models show absolute FNR increases of approximately 0.34 to 0.36 under the held-out-site protocol. Logistic Regression behaves differently: its random-split FNR is already high, and cross-site testing lowers FNR while still producing weaker PR-AUC than the Random Forest baseline. Text-only models remain near the rare-event baseline and do not provide useful fault recall in either split.

![Figure 4: PR-AUC under Random Split vs. Held-Out-Site Generalization.](outputs/figure_4_site_shift.png)

### 7.4. Robustness Analysis (Experiment 3)
We perturbed the test features for all seven models under sensor noise, structured-feature missingness, technician-report missingness, and report ambiguity. The complete robustness table is saved as `table_7_robustness_summary.csv`; **Table 8** summarizes the main patterns for representative models.

| Perturbation | Model | Baseline PR-AUC | Stress Level | Stressed PR-AUC | Baseline FNR | Stressed FNR |
|:---|:---|---:|:---|---:|---:|---:|
| Sensor noise | `rf_struct` | 0.3981 | 3.0x | 0.1358 | 0.5852 | 0.4344 |
| Sensor noise | `fusion_minilm` | 0.3314 | 3.0x | 0.1145 | 0.6585 | 0.5978 |
| Missing structured values | `rf_struct` | 0.3981 | 40% | 0.2821 | 0.5852 | 0.7696 |
| Missing structured values | `fusion_minilm` | 0.3314 | 40% | 0.2389 | 0.6585 | 0.8380 |
| Missing reports | `lr_tfidf` | 0.0894 | 75% | 0.0746 | 1.0000 | 1.0000 |
| Missing reports | `fusion_minilm` | 0.3314 | 75% | 0.3288 | 0.6585 | 0.6453 |
| Report ambiguity | `lr_minilm` | 0.0889 | 75% | 0.0723 | 1.0000 | 1.0000 |
| Report ambiguity | `fusion_minilm` | 0.3314 | 75% | 0.3293 | 0.6585 | 0.6515 |

**Table 8:** Selected Robustness Results under Test-Time Perturbations.

Sensor noise causes the largest ranking degradation for sensor-dependent models. For example, Random Forest PR-AUC falls from 0.3981 to 0.1358 under a 3.0x noise multiplier, and `fusion_minilm` falls from 0.3314 to 0.1145. Structured missingness has a different failure mode: PR-AUC declines less sharply than under noise, but FNR increases substantially for Random Forest and `fusion_minilm`, indicating more missed maintenance-required cases at the default threshold. Report perturbations primarily affect text-only baselines, which are already weak; their effect on fusion models is small because the trained fusion classifiers rely mainly on structured features.

![Figure 5: Robustness Curves under Sensor Noise, Missingness, and Unstructured Log Perturbations.](outputs/figure_5_robustness.png)

![Figure 6: Predictive Performance Comparison between TF-IDF and Frozen MiniLM Text Representations.](outputs/figure_6_text_comparison.png)

### 7.5. Constrained Decision Routing Results (Experiment 4)
We optimized routing thresholds on the validation set using a safety ceiling of $\alpha = 0.05$ (maximum 5% missed critical faults). The validation-optimized thresholds are listed in **Table 10**.

| Model | $t_{low}^*$ | $t_{high}^*$ | Validation Workload Reduction | Validation Missed Fault Rate |
|:---|---:|---:|---:|---:|
| `lr_struct` | 0.01 | 0.02 | 73.45% | 4.79% |
| `rf_struct` | 0.01 | 0.02 | 88.12% | 2.57% |
| `xgb_struct` | 0.00 | 0.01 | 0.00% | 0.00% |
| `lr_tfidf` | 0.00 | 0.01 | 0.00% | 0.00% |
| `lr_minilm` | 0.00 | 0.01 | 0.00% | 0.00% |
| `fusion_tfidf` | 0.00 | 0.01 | 0.00% | 0.00% |
| `fusion_minilm`| 0.00 | 0.01 | 0.00% | 0.00% |

**Table 10:** Selected Routing Thresholds and Validation Objective Values ($\alpha = 0.05$).

When we applied these frozen thresholds to the independent test set, we obtained the results shown in **Table 11**.

| Model | Auto-Clear Rate | Human-Review Rate | Urgent-Inspection Rate | Workload Reduction | Missed Critical Fault Rate | False Urgent-Inspection Rate | Urgent-Inspection Precision |
|:---|---:|---:|---:|---:|---:|---:|---:|
| `lr_struct` | 55.70% | 4.75% | 39.56% | 55.70% | 0.07% | 35.37% | 16.42% |
| `rf_struct` | 71.92% | 4.77% | 23.31% | 71.92% | 0.35% | 18.14% | 27.24% |
| `xgb_struct` | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6.51% |
| `lr_tfidf` | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6.51% |
| `lr_minilm` | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6.51% |
| `fusion_tfidf` | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6.51% |
| `fusion_minilm`| 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 100.00% | 6.51% |

**Table 11:** Test-Set Human-in-the-Loop Decision-Routing Results.

The Random Forest model (`rf_struct`) achieves the best balance: it reduces technician workload by **71.92%** (by auto-clearing healthy assets) while maintaining a missed critical fault rate of **0.35%**, well below the 5% safety ceiling. Logistic Regression (`lr_struct`) also performs well, achieving a **55.70%** workload reduction with a missed fault rate of **0.07%**. 

In contrast, XGBoost (`xgb_struct`) and the fusion models fail to reduce workload. The threshold optimizer sets $t_{low}^* = 0.0$ and $t_{high}^* = 0.01$ for these models, routing all cases to urgent inspection. This conservative routing occurs because these models generate high-variance, uncalibrated probabilities in low-risk regions. Even after Platt scaling, several true positive cases receive very low probabilities (below 0.01). To satisfy the strict safety constraint, the optimizer is forced to route all assets to inspection, resulting in 0% workload reduction.

![Figure 7: Probability Reliability Diagram (Calibration Curve) for the Fusion MiniLM Baseline.](outputs/figure_7_calibration.png)

![Figure 8: Routing Decisions Allocation (Auto-Clear, Human Review, Urgent Inspection) by Model.](outputs/figure_8_routing_distribution.png)

### 7.5. Routing Sensitivity Analysis
We evaluated the workload-safety trade-off for every model across $\alpha \in \{0.01, 0.03, 0.05, 0.10\}$, with each threshold pair selected on the validation set and then frozen before test evaluation. The complete sensitivity table is saved as `routing_sensitivity.csv`. At $\alpha=0.01$, no model safely auto-clears any test cases. At $\alpha=0.03$ and $\alpha=0.05$, Random Forest selects $t_{low}=0.01$ and $t_{high}=0.02$, giving 71.9% test workload reduction with a 0.35% test missed-fault rate. At $\alpha=0.10$, Random Forest increases workload reduction to 76.7% with a 2.44% test missed-fault rate, while Logistic Regression reaches 65.4% workload reduction with a 0.49% test missed-fault rate. XGBoost, text-only models, and fusion models remain conservatively routed to urgent inspection across all tested $\alpha$ values, indicating low-probability calibration tails that violate the auto-clear safety constraint.

![Figure 9: Workload Reduction vs. Safety Constraints (Sensitivity Analysis for alpha).](outputs/figure_9_sensitivity.png)

### 7.6. Explainability and Diagnostic Auditing

#### 7.6.1. Global Feature Importances
To verify that models are learning physically meaningful features rather than noise, we analyzed the global feature importances of the XGBoost baseline. The top features are listed in **Table 12**.

| Feature | Importance | Feature | Importance |
|:---|---:|:---|---:|
| `time_since_last_maintenance` | 5.9260 | `vibration_rms` | 0.2046 |
| `age` | 0.8952 | `site_id_Plant_1` | 0.1940 |
| `acoustic_level` | 0.3544 | `vibration_kurtosis` | 0.1610 |
| `operating_hours` | 0.3453 | `humidity` | 0.1542 |
| `component_type_Seal` | 0.2238 | `component_type_Motor Unit` | 0.1386 |
| `ambient_temp` | 0.2227 | `motor_current` | 0.1200 |
| `previous_fault_count` | 0.2068 | `temp_deviation` | 0.1135 |

**Table 12:** Global Feature Importances for the XGBoost Structured Baseline.

The most important features are `time_since_last_maintenance` and `age`, which directly scale the latent degradation rate in our physical simulation. These are followed by physical sensors like `acoustic_level` and `vibration_rms`, which capture active symptoms of wear.

![Figure 10: Global Feature SHAP Summary Plot for the Structured XGBoost Classifier.](outputs/figure_10_shap.png)

#### 7.6.2. Term Weight Importances (TF-IDF Baseline)
We extracted the coefficients of the TF-IDF Logistic Regression text baseline to understand what language features the model uses. The top coefficients are:
*   **Top Positive Coefficients (Indicating Faults)**: `high` (0.89), `thermal` (0.71), `vibration` (0.68), `temperature` (0.66), `detected` (0.65), `load` (0.60), `inspection` (0.56), `bearing` (0.53), `noise` (0.52), and `increase` (0.52).
*   **Top Negative Coefficients (Indicating Normal Health)**: `parameters` (-0.56), `operating` (-0.18), `normal` (-0.12), `potential` (-0.08), `build` (-0.04), `flow` (-0.03), and `blockage` (-0.03).

#### 7.6.3. Local Case Diagnostic Analysis
The pipeline records local diagnostic examples for one true positive, false positive, false negative, and true negative using the `fusion_minilm` model and a diagnostic threshold of 0.5. The saved cases are:

| Case | Test Index | Calibrated Probability |
|:---|---:|---:|
| True Positive | 15227 | 0.7986 |
| False Positive | 17430 | 0.7984 |
| False Negative | 21636 | 0.0116 |
| True Negative | 4499 | 0.0116 |

These examples identify records for manual inspection and figure generation. The current artifact records probabilities and case labels, but it does not yet save per-case SHAP contribution tables; therefore, this draft does not claim feature-level local explanations beyond the global SHAP and TF-IDF coefficient analyses.

---

## 8. Discussion

### 8.1. Classical TF-IDF vs. Sentence Transformer Embeddings
Our results show that both text-only baselines (`lr_tfidf` and `lr_minilm`) perform poorly on their own, achieving PR-AUC below 0.09 on the chronological test split. This occurs because technician reports in our simulation are brief, sparse, and qualitative, and they do not capture the continuous physical changes in the assets. When fused with structured sensors, neither TF-IDF nor MiniLM embeddings improve over the structured-only Random Forest. This suggests that, in this benchmark configuration, sensor and maintenance-history variables are the primary predictors, while text reports mainly provide auxiliary context.

### 8.2. Cross-Site Calibration and FNR Inflation
Domain shifts across manufacturing sites represent a major challenge for AI deployment. In this benchmark, random splits overstate performance for tree and fusion models: Random Forest, XGBoost, and both fusion models lose 0.27 to 0.37 PR-AUC points under held-out-site evaluation, and their FNR increases by approximately 0.34 to 0.36 absolute points. This occurs because site-specific differences in sensor baselines and operating conditions shift the probability distributions. A model using a default 0.5 threshold can miss many maintenance-required cases under site shift, which is operationally unacceptable for decision support. This result highlights the need for site-aware validation, probability calibration, and conservative threshold selection before deployment.

### 8.3. Explainability and Auditing
Post-hoc explainability methods like SHAP and term coefficient analysis are useful diagnostic tools for this benchmark:
*   They allow researchers to check whether the model emphasizes variables that are plausible under the simulated degradation process, such as maintenance history, asset age, vibration, and acoustic level.
*   They complement, but do not replace, the explicit leakage audit. Leakage prevention must be enforced by the feature pipeline before model training.
*   They support model diagnostics but should not be interpreted as causal evidence or as validation by real maintenance supervisors.

---

## 9. Limitations and Mitigation Strategies

Any attempt to use this benchmark for real industrial decision support would require external validation. **Table 13** summarizes the key limitations of the synthetic benchmark and corresponding mitigation strategies.

| Identified Limitation | Description | Mitigation Strategy |
|:---|:---|:---|
| **Synthetic Data Source** | The dataset is simulated and may not capture all physical complexities of real plants. | Use the benchmark to verify algorithms and pipeline code before deploying on real data. |
| **Simplified Degradation Dynamics**| Component wear is modeled as a Markovian process with additive noise, which simplifies complex wear physics. | Inject stochastic shock events and site-specific scaling factors to increase simulation complexity. |
| **Template-Generated Reports** | Text reports are generated from templates rather than natural language written by technicians. | Inject template ambiguity, missing reports, and misleading text flags to test NLP robustness. |
| **Simulated Decision Routing** | The routing model assumes human reviews and inspections are completed immediately and with perfect accuracy. | Perform sensitivity analysis on the safety ceiling $\alpha$ to evaluate the impact of human review delays. |
| **Out-of-Domain Generalization** | Performance on real plant installations may vary due to environmental factors. | Release the generator code and configuration so future users can customize and validate scenarios against their own data. |

**Table 13:** Benchmark Limitations and Mitigation Strategies.

---

## 10. Conclusion and Future Work

In this paper, we introduced a synthetic benchmark to evaluate AI-based O&M decision support under label scarcity, noisy observations, imperfect technician reports, and site shift. Using a simulated multi-site environment with 480 assets and 175,200 records, we showed that structured models outperform text-only baselines and that random splits can substantially overstate held-out-site performance for tree and fusion models.

We demonstrated that validation-set optimized threshold routing can reduce review workload while enforcing a predefined missed-fault constraint on the validation set before test evaluation. A calibrated Random Forest model reduced technician workload by **71.92%** while maintaining a missed critical fault rate of **0.35%** on the test set under the $\alpha=0.05$ validation constraint. In contrast, models with unstable low-probability calibration tails, including XGBoost and the fusion models, could not safely auto-clear assets under the tested constraints and were routed conservatively.

The results support the benchmark's purpose: evaluating industrial O&M models beyond aggregate predictive accuracy by testing calibration, robustness, split design, explanation plausibility, and operationally constrained routing. Future work should validate the protocol on real industrial data, strengthen the physics of the simulator, add richer non-templated technician language, evaluate domain-adaptive language models, and test human-in-the-loop routing with maintenance experts.

---

## Backmatter Declarations

### Author Contributions
Conceptualization, P.P.; methodology, P.P.; software, P.P.; validation, P.P.; formal analysis, P.P.; investigation, P.P.; data curation, P.P.; writing, original draft preparation, P.P.; writing, review and editing, P.P.; visualization, P.P.; project administration, P.P. The author has read and agreed to the published version of the manuscript.

### Data Availability Statement
The synthetic dataset generation code, benchmark configuration files, generated dataset, split manifests, model-training scripts, threshold-selection scripts, evaluation outputs, result tables, and figures will be made available in a public GitHub repository before submission. The repository URL, release tag, and commit hash must be inserted here after the repository is created. The dataset is fully synthetic and does not contain real industrial, personal, proprietary, or institution-owned data.

### Conflicts of Interest
The author declares no conflict of interest.

### Generative AI Disclosure
During the preparation of this manuscript/study, the author used ChatGPT and Gemini for research planning, code drafting, manuscript structuring, and language refinement. The author reviewed and edited all outputs, verified the methods, results, and references, and takes full responsibility for the content of this publication.

---

## References

1. Hakami, A. Strategies for overcoming data scarcity, imbalance, and feature selection challenges in machine learning models for predictive maintenance. *Scientific Reports* **2024**, *14*, 9645. https://doi.org/10.1038/s41598-024-59958-9.
2. Shyalika, C.; Wickramarachchi, R.; El Kalach, F.; Harik, R.; Sheth, A. Evaluating the Role of Data Enrichment Approaches towards Rare Event Analysis in Manufacturing. *Sensors* **2024**, *24*, 5009. https://doi.org/10.3390/s24155009.
3. Chen, C.; et al. The advance of digital twin for predictive maintenance: The role and function of machine learning. *Journal of Manufacturing Systems* **2023**, *71*, 581--594. https://doi.org/10.1016/j.jmsy.2023.10.010.
4. Yan, S.; Zhong, X.; Shao, H. Digital twin-assisted imbalanced fault diagnosis framework using subdomain adaptive mechanism and margin-aware regularization. *Reliability Engineering & System Safety* **2023**, *239*, 109522. https://doi.org/10.1016/j.ress.2023.109522.
5. Chen, L.; Li, Q.; Shen, C.; Zhu, J.; Wang, D.; Xia, M. Adversarial Domain-Invariant Generalization: A Generic Domain-Regressive Framework for Bearing Fault Diagnosis Under Unseen Conditions. *IEEE Transactions on Industrial Informatics* **2022**, *18*, 1790--1800. https://doi.org/10.1109/TII.2021.3078712.
6. Asutkar, S.; Tallur, S. Deep transfer learning strategy for efficient domain generalisation in machine fault diagnosis. *Scientific Reports* **2023**, *13*, 6607. https://doi.org/10.1038/s41598-023-33887-5.
7. Forest, F.; Fink, O. Calibrated Adaptive Teacher for Domain-Adaptive Intelligent Fault Diagnosis. *Sensors* **2024**, *24*, 7539. https://doi.org/10.3390/s24237539.
8. Hendriks, J.; Dumond, P.; Knox, D.A. Towards better benchmarking using the CWRU bearing fault dataset. *Mechanical Systems and Signal Processing* **2022**, *169*, 108732. https://doi.org/10.1016/j.ymssp.2021.108732.
9. Usuga-Cadavid, J.P.; Lamouri, S.; Grabot, B.; Fortin, A. Using deep learning to value free-form text data for predictive maintenance. *International Journal of Production Research* **2022**, *60*, 4548--4575. https://doi.org/10.1080/00207543.2021.1951868.
10. Sundaram, S.; Zeid, A. Technical language processing for Prognostics and Health Management: applying text similarity and topic modeling to maintenance work orders. *Journal of Intelligent Manufacturing* **2025**, *36*, 1637--1657. https://doi.org/10.1007/s10845-024-02323-4.
11. Zhou, J.; Guo, Y.; Yang, Z.; et al. T2MFDF: An LLM-Enhanced Multimodal Fault Diagnosis Framework Integrating Time-Series and Textual Data. *IEEE Transactions on Instrumentation and Measurement* **2025**, *74*, 3547911. https://doi.org/10.1109/TIM.2025.3583374.
12. Silva Filho, T.; Song, H.; Perello-Nieto, M.; Santos-Rodriguez, R.; Kull, M.; Flach, P. Classifier calibration: a survey on how to assess and improve predicted class probabilities. *Machine Learning* **2023**, *112*, 3211--3260. https://doi.org/10.1007/s10994-023-06336-7.
13. Hendrickx, K.; Perini, L.; Van der Plas, D.; Meert, W.; Davis, J. Machine learning with a reject option: a survey. *Machine Learning* **2024**, *113*, 3073--3110. https://doi.org/10.1007/s10994-024-06534-x.
14. Hasan, M.M.; Abdar, M.; Khosravi, A.; et al. Survey on Leveraging Uncertainty Estimation Toward Trustworthy Deep Neural Networks: The Case of Reject Option and Post-Training Processing. *ACM Computing Surveys* **2025**. https://doi.org/10.1145/3727633.
15. Sayin, B.; et al. Bringing Machine Learning Classifiers Into Critical Cyber-Physical Systems: A Matter of Design. *IEEE Access* **2025**. https://doi.org/10.1109/ACCESS.2025.3568501.
16. Cummins, L.; Sommers, A.; Bakhtiari Ramezani, S.; Mittal, S.; Jabour, J.; Seale, M.; Rahimi, S. Explainable Predictive Maintenance: A Survey of Current Methods, Challenges and Opportunities. *IEEE Access* **2024**. https://doi.org/10.1109/ACCESS.2024.3391130.
17. Brusa, E.; Cibrario, L.; Delprete, C.; Di Maggio, L.G. Explainable AI for Machine Fault Diagnosis: Understanding Features' Contribution in Machine Learning Models for Industrial Condition Monitoring. *Applied Sciences* **2023**, *13*, 2038. https://doi.org/10.3390/app13042038.
18. Barnard, A.S. BenchMake: turn any scientific data set into a reproducible benchmark. *Machine Learning: Science and Technology* **2025**, *6*, 030502. https://doi.org/10.1088/2632-2153/adf810.
19. McDermott, M.B.A.; Wang, S.; Marinsek, N.; Ranganath, R.; Foschini, L.; Ghassemi, M. Reproducibility in Machine Learning and Healthcare: How far do we have to go? *Science Translational Medicine* **2021**. https://doi.org/10.1126/scitranslmed.abb1655.
20. Jardine, A.K.; Lin, D.; Banjevic, D. A Review on Machinery Diagnostics and Prognostics Implementing Condition-Based Maintenance. *Mechanical Systems and Signal Processing* **2006**, *20*, 1483--1510.
21. Grall, A.; Dieulle, L.; Berenguer, C.; Roussignol, M. A Continuous-Time Predictive Maintenance Model for a Continuously Deteriorating System. *Reliability Engineering & System Safety* **2002**, *76*, 181--188.
22. scikit-learn developers. `TfidfVectorizer`. scikit-learn documentation. Available online: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html (accessed on 21 May 2026).
23. sentence-transformers. `sentence-transformers/all-MiniLM-L6-v2`. Hugging Face model card. Available online: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 (accessed on 21 May 2026).
24. Breiman, L. Random Forests. *Machine Learning* **2001**, *45*, 5--32.
25. Chen, T.; Guestrin, C. XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, CA, USA, 13--17 August 2016; pp. 785--794.
26. Lundberg, S.M.; Lee, S.-I. A Unified Approach to Interpreting Model Predictions. In *Advances in Neural Information Processing Systems 30*; Curran Associates, Inc.: Red Hook, NY, USA, 2017; pp. 4765--4774.
27. Platt, J. Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods. In *Advances in Large Margin Classifiers*; MIT Press: Cambridge, MA, USA, 1999; pp. 61--74.
