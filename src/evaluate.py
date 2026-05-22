import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, balanced_accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# Import modules from our package
from src.train import train_and_calibrate, DatasetPreprocessor, get_minilm_embeddings
from src.routing import optimize_thresholds, compute_routing_metrics, run_sensitivity_analysis
from src.explain import get_shap_explanations, get_tfidf_explanations, analyze_local_cases

RANDOM_STATE = 42

# Premium styling for matplotlib
def set_premium_style():
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#EAEAEA'
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['font.size'] = 10
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.edgecolor'] = '#EAEAEA'
    plt.rcParams['xtick.color'] = '#555555'
    plt.rcParams['ytick.color'] = '#555555'
    plt.rcParams['axes.labelcolor'] = '#333333'
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['font.family'] = 'sans-serif'
    
# Helper to compute PR-AUC
def pr_auc_score(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)

# Expected Calibration Error (ECE) helper
def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    
    ece = 0.0
    total_samples = len(y_true)
    for i in range(n_bins):
        bin_mask = bin_assignments == i
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_mask])
            bin_conf = np.mean(y_prob[bin_mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
    return ece

def generate_static_tables(output_dir="outputs"):
    """
    Saves static markdown and CSV tables as required by the brief.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: Schema
    t1 = pd.read_csv(os.path.join(output_dir, "data_dictionary.csv"))
    with open(os.path.join(output_dir, "table_1_schema.md"), "w") as f:
        f.write("# Table 1: Synthetic Dataset Schema and Feature Groups\n\n")
        f.write(t1.to_markdown(index=False))
        
    # Table 3: Experimental scenarios
    t3 = pd.DataFrame({
        "Experimental Scenario": [
            "Experiment 1: Label Scarcity Learning Curve",
            "Experiment 2: Cross-Site Domain Shift",
            "Experiment 3: Robustness under Sensor & Report Noise",
            "Experiment 4: Constrained Human-in-the-Loop Routing"
        ],
        "Controlled Variable(s)": [
            "Training label fraction: 1%, 5%, 10%, 25%, 50%, 100%",
            "Held-out plant site for testing: Plant_1 to Plant_6",
            "Sensor noise multiplier (1.0x to 3.0x), feature missingness (0% to 40%), report missingness (0% to 75%), ambiguity level",
            "Routing safety constraint limit alpha: 0.01, 0.03, 0.05, 0.10"
        ],
        "Evaluation Metrics": [
            "PR-AUC, ROC-AUC, Brier score, ECE, F1-score",
            "Drop in PR-AUC, increase in FNR, calibration drift",
            "PR-AUC degradation rate, False Negative Rate (FNR) inflation",
            "Auto-clear rate, Workload reduction, Missed Critical Fault Rate"
        ]
    })
    t3.to_csv(os.path.join(output_dir, "table_3_scenarios.csv"), index=False)
    with open(os.path.join(output_dir, "table_3_scenarios.md"), "w") as f:
        f.write("# Table 3: Experimental Scenarios and Controlled Variables\n\n")
        f.write(t3.to_markdown(index=False))

    # Table 4: Model Stack
    t4 = pd.DataFrame({
        "Model Type": [
            "Structured-only Logistic Regression (lr_struct)",
            "Structured-only Random Forest (rf_struct)",
            "Structured-only XGBoost (xgb_struct)",
            "Classical Text-only TF-IDF + Logistic Regression (lr_tfidf)",
            "Modern Text-only MiniLM + Logistic Regression (lr_minilm)",
            "Fusion Model: Structured + TF-IDF (fusion_tfidf)",
            "Fusion Model: Structured + MiniLM (fusion_minilm)"
        ],
        "Input Modalities": [
            "Structured sensor & maintenance features",
            "Structured sensor & maintenance features",
            "Structured sensor & maintenance features",
            "Raw technician reports (TF-IDF sparse tokens)",
            "Raw technician reports (MiniLM dense embeddings)",
            "Concatenated Structured & TF-IDF sparse features",
            "Concatenated Structured & MiniLM dense features"
        ],
        "Calibration Method": ["Platt Scaling (Sigmoid Validation-fit)"] * 7
    })
    t4.to_csv(os.path.join(output_dir, "table_4_model_stack.csv"), index=False)
    with open(os.path.join(output_dir, "table_4_model_stack.md"), "w") as f:
        f.write("# Table 4: Model Stack and Input Modalities\n\n")
        f.write(t4.to_markdown(index=False))

    # Table 10: Limitations
    t10 = pd.DataFrame({
        "Identified Limitation": [
            "Purely Synthetic Data Source",
            "Simplified Latent Degradation Dynamics",
            "Template-Generated Technician Reports",
            "Simulated Human-in-the-Loop Thresholds",
            "Out-of-Domain Generalization to Real Sites"
        ],
        "Description": [
            "Dataset is simulated and does not use real plant observations.",
            "Degradation modeled as Markovian process with additive noise/shocks.",
            "Text reports constructed from templates rather than natural speech.",
            "Workload reduction assumes immediate human availability and accuracy.",
            "Performance on real systems may vary due to unmodeled environmental physics."
        ],
        "Mitigation Strategy": [
            "Benchmark serves as a method verification protocol prior to deployment.",
            "Include stochastic shocks and site scaling parameters to reflect complexity.",
            "Inject 15% template ambiguity, omission, and misleading report flags.",
            "Run sensitivity analysis on alpha constraint to explore safety envelopes.",
            "Publish open generator code for site-specific customization and retraining."
        ]
    })
    t10.to_csv(os.path.join(output_dir, "table_10_limitations.csv"), index=False)
    with open(os.path.join(output_dir, "table_10_limitations.md"), "w") as f:
        f.write("# Table 10: Limitations and Mitigation Strategies\n\n")
        f.write(t10.to_markdown(index=False))

def build_evaluation_tables_and_plots(dataset_path, output_dir="outputs"):
    set_premium_style()
    np.random.seed(RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    df = pd.read_parquet(dataset_path)
    
    # Save Table 2: Label distributions
    fault_counts = df["fault_type"].value_counts()
    fault_dist = pd.DataFrame({
        "Fault Type": fault_counts.index,
        "Count": fault_counts.values,
        "Percentage (%)": (fault_counts.values / len(df)) * 100
    })
    
    priority_counts = df["maintenance_priority"].value_counts()
    priority_dist = pd.DataFrame({
        "Priority Label": priority_counts.index,
        "Count": priority_counts.values,
        "Percentage (%)": (priority_counts.values / len(df)) * 100
    })
    
    table_2_md = "# Table 2: Fault Types and Maintenance Priority Distribution\n\n"
    table_2_md += "### Class Distribution by Fault Type\n"
    table_2_md += fault_dist.to_markdown(index=False) + "\n\n"
    table_2_md += "### Class Distribution by Maintenance Priority\n"
    table_2_md += priority_dist.to_markdown(index=False)
    
    with open(os.path.join(output_dir, "table_2_distribution.md"), "w") as f:
        f.write(table_2_md)
    fault_dist.to_csv(os.path.join(output_dir, "table_2_faults.csv"), index=False)
    priority_dist.to_csv(os.path.join(output_dir, "table_2_priority.csv"), index=False)
    
    # Load split data
    train_df = pd.read_parquet(os.path.join(output_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(output_dir, "val.parquet"))
    test_df = pd.read_parquet(os.path.join(output_dir, "test.parquet"))
    
    # Train full baseline models (100% data)
    print("Training standard baseline models on 100% data split...")
    preprocessor, models, X_train, X_val = train_and_calibrate(train_df, val_df)
    
    # Transform test set
    test_struct, test_tfidf, test_minilm = preprocessor.transform(test_df)
    y_test = test_df["maintenance_required_7d"].values
    
    X_test_dict = {
        "struct": test_struct.values,
        "tfidf": test_tfidf.values,
        "minilm": test_minilm.values,
        "fusion_tfidf": np.hstack([test_struct.values, test_tfidf.values]),
        "fusion_minilm": np.hstack([test_struct.values, test_minilm.values])
    }
    
    model_feature_map = {
        "lr_struct": "struct",
        "rf_struct": "struct",
        "xgb_struct": "struct",
        "lr_tfidf": "tfidf",
        "lr_minilm": "minilm",
        "fusion_tfidf": "fusion_tfidf",
        "fusion_minilm": "fusion_minilm"
    }
    
    # Evaluate baseline test set performance
    eval_results = []
    test_probs_dict = {}
    
    for name, model in models.items():
        feat_key = model_feature_map[name]
        X_te = X_test_dict[feat_key]
        
        # Predict calibrated probability
        probs = model.predict_proba(X_te)[:, 1]
        test_probs_dict[name] = probs
        
        # Binary predictions based on default 0.5 threshold
        preds = (probs >= 0.5).astype(int)
        
        # Metrics
        roc = roc_auc_score(y_test, probs)
        pr = pr_auc_score(y_test, probs)
        f1 = f1_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        bal_acc = balanced_accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        fnr = 1.0 - rec
        fpr = 1.0 - precision_score(1 - y_test, 1 - preds)
        brier = brier_score_loss(y_test, probs)
        ece = expected_calibration_error(y_test, probs)
        
        eval_results.append({
            "Model": name,
            "ROC-AUC": roc,
            "PR-AUC": pr,
            "F1-Score": f1,
            "Macro-F1": macro_f1,
            "Balanced Accuracy": bal_acc,
            "Precision": prec,
            "Recall (Sensitivity)": rec,
            "False Negative Rate": fnr,
            "False Positive Rate": fpr,
            "Brier Score": brier,
            "ECE": ece
        })
        
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(os.path.join(output_dir, "table_5_predictive_performance.csv"), index=False)
    with open(os.path.join(output_dir, "table_5_predictive_performance.md"), "w") as f:
        f.write("# Table 5: Main Predictive Performance Results on Test Set\n\n")
        f.write(df_eval.to_markdown(index=False))
        
    print("Baseline performance metrics computed.")

    # Random split baseline for Experiment 2 comparison.
    random_eval_df = None
    random_split_dir = os.path.join(output_dir, "random_splits")
    if os.path.exists(os.path.join(random_split_dir, "train.parquet")):
        print("Training random-split baseline models for cross-site comparison...")
        random_train_df = pd.read_parquet(os.path.join(random_split_dir, "train.parquet"))
        random_val_df = pd.read_parquet(os.path.join(random_split_dir, "val.parquet"))
        random_test_df = pd.read_parquet(os.path.join(random_split_dir, "test.parquet"))
        random_prep, random_models, _, random_X_val = train_and_calibrate(random_train_df, random_val_df)

        r_struct, r_tfidf, r_minilm = random_prep.transform(random_test_df)
        random_X_test = {
            "struct": r_struct.values,
            "tfidf": r_tfidf.values,
            "minilm": r_minilm.values,
            "fusion_tfidf": np.hstack([r_struct.values, r_tfidf.values]),
            "fusion_minilm": np.hstack([r_struct.values, r_minilm.values])
        }
        y_random_test = random_test_df["maintenance_required_7d"].values
        random_results = []
        for name, model in random_models.items():
            feat_key = model_feature_map[name]
            probs = model.predict_proba(random_X_test[feat_key])[:, 1]
            preds = (probs >= 0.5).astype(int)
            rec = recall_score(y_random_test, preds)
            random_results.append({
                "Model": name,
                "Random-Split ROC-AUC": roc_auc_score(y_random_test, probs),
                "Random-Split PR-AUC": pr_auc_score(y_random_test, probs),
                "Random-Split FNR": 1.0 - rec,
                "Random-Split ECE": expected_calibration_error(y_random_test, probs)
            })
        random_eval_df = pd.DataFrame(random_results)
        random_eval_df.to_csv(os.path.join(output_dir, "random_split_performance.csv"), index=False)
    
    # ----------------------------------------------------
    # Experiment 1: Low-Label Learning Curve
    # ----------------------------------------------------
    print("Running Experiment 1: Low-label learning curves...")
    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
    exp1_results = []
    
    # Select subset models to track
    track_models = ["lr_struct", "rf_struct", "xgb_struct", "lr_tfidf", "lr_minilm", "fusion_tfidf", "fusion_minilm"]
    
    for frac in fractions:
        print(f"  Training on {frac*100}% labels...")
        if frac < 1.0:
            split_name = f"train_{int(frac * 100):03d}pct.parquet"
            scarcity_path = os.path.join(output_dir, "label_scarcity_splits", split_name)
            if os.path.exists(scarcity_path):
                train_sub = pd.read_parquet(scarcity_path)
            else:
                # Stratified sample to ensure we have positive labels
                train_sub, _ = train_test_split(
                    train_df, train_size=frac, random_state=RANDOM_STATE,
                    stratify=train_df["maintenance_required_7d"]
                )
        else:
            scarcity_path = os.path.join(output_dir, "label_scarcity_splits", "train_100pct.parquet")
            train_sub = pd.read_parquet(scarcity_path) if os.path.exists(scarcity_path) else train_df
            
        try:
            sub_prep, sub_models, _, _ = train_and_calibrate(train_sub, val_df)
            
            # Preprocess test set with sub-model's preprocessor
            t_struct, t_tfidf, t_minilm = sub_prep.transform(test_df)
            X_t_sub = {
                "struct": t_struct.values,
                "tfidf": t_tfidf.values,
                "minilm": t_minilm.values,
                "fusion_tfidf": np.hstack([t_struct.values, t_tfidf.values]),
                "fusion_minilm": np.hstack([t_struct.values, t_minilm.values])
            }
            
            for m_name in track_models:
                if m_name in sub_models:
                    model = sub_models[m_name]
                    feat_key = model_feature_map[m_name]
                    probs = model.predict_proba(X_t_sub[feat_key])[:, 1]
                    
                    pr = pr_auc_score(y_test, probs)
                    roc = roc_auc_score(y_test, probs)
                    brier = brier_score_loss(y_test, probs)
                    ece = expected_calibration_error(y_test, probs)
                    
                    exp1_results.append({
                        "Fraction": frac,
                        "Model": m_name,
                        "PR-AUC": pr,
                        "ROC-AUC": roc,
                        "Brier Score": brier,
                        "ECE": ece
                    })
        except Exception as e:
            print(f"Error on fraction {frac}: {e}")
            
    df_exp1 = pd.DataFrame(exp1_results)
    df_exp1.to_csv(os.path.join(output_dir, "exp1_learning_curves.csv"), index=False)
    
    # ----------------------------------------------------
    # Experiment 2: Cross-Site Domain Shift
    # ----------------------------------------------------
    print("Running Experiment 2: Cross-site generalization...")
    exp2_results = []
    
    for i in range(1, 7):
        site_id = f"Plant_{i}"
        print(f"  Holding out {site_id}...")
        split_dir = os.path.join(output_dir, "site_splits", site_id)
        
        tr_site = pd.read_parquet(os.path.join(split_dir, "train.parquet"))
        va_site = pd.read_parquet(os.path.join(split_dir, "val.parquet"))
        te_site = pd.read_parquet(os.path.join(split_dir, "test.parquet"))
        
        try:
            site_prep, site_models, _, _ = train_and_calibrate(tr_site, va_site)
            
            # Preprocess the held-out site test split
            te_struct, te_tfidf, te_minilm = site_prep.transform(te_site)
            X_te_site = {
                "struct": te_struct.values,
                "tfidf": te_tfidf.values,
                "minilm": te_minilm.values,
                "fusion_tfidf": np.hstack([te_struct.values, te_tfidf.values]),
                "fusion_minilm": np.hstack([te_struct.values, te_minilm.values])
            }
            y_te_site = te_site["maintenance_required_7d"].values
            
            for m_name in track_models:
                if m_name in site_models:
                    model = site_models[m_name]
                    feat_key = model_feature_map[m_name]
                    probs = model.predict_proba(X_te_site[feat_key])[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    pr = pr_auc_score(y_te_site, probs)
                    rec = recall_score(y_te_site, preds)
                    fnr = 1.0 - rec
                    ece = expected_calibration_error(y_te_site, probs)
                    
                    # Also compute the standard baseline model's performance on this specific site
                    # to measure transfer gap
                    # Standard model is trained on Plants 1-5 (excluding 6) or site-split trained
                    # Let's compare "Within Domain" vs "Held-Out"
                    exp2_results.append({
                        "Held_Out_Site": site_id,
                        "Model": m_name,
                        "PR-AUC": pr,
                        "FNR": fnr,
                        "ECE": ece
                    })
        except Exception as e:
            print(f"Error on site split {site_id}: {e}")
            
    df_exp2 = pd.DataFrame(exp2_results)
    df_exp2.to_csv(os.path.join(output_dir, "exp2_site_generalization.csv"), index=False)
    
    # Generate Table 6: Cross-site comparison
    # Summarize mean drop across sites
    # Let's construct a clean Table 6 representation
    t6_summary = []
    for m_name in track_models:
        site_metrics = df_exp2[df_exp2["Model"] == m_name]
        mean_pr = site_metrics["PR-AUC"].mean()
        std_pr = site_metrics["PR-AUC"].std()
        mean_fnr = site_metrics["FNR"].mean()
        
        # Find random-split test performance for comparison, falling back to chronological baseline.
        if random_eval_df is not None:
            base_row = random_eval_df[random_eval_df["Model"] == m_name]
            base_pr = base_row["Random-Split PR-AUC"].values[0] if len(base_row) > 0 else 0.0
            base_fnr = base_row["Random-Split FNR"].values[0] if len(base_row) > 0 else 0.0
            base_label = "Random-Split"
        else:
            base_row = df_eval[df_eval["Model"] == m_name]
            base_pr = base_row["PR-AUC"].values[0] if len(base_row) > 0 else 0.0
            base_fnr = base_row["False Negative Rate"].values[0] if len(base_row) > 0 else 0.0
            base_label = "Chronological"
        
        t6_summary.append({
            "Model": m_name,
            f"{base_label} Test PR-AUC": base_pr,
            "Cross-Site Mean PR-AUC": mean_pr,
            "PR-AUC Drop (Shift Gap)": base_pr - mean_pr,
            f"{base_label} FNR": base_fnr,
            "Cross-Site Mean FNR": mean_fnr,
            "FNR Inflation": mean_fnr - base_fnr
        })
        
    df_t6 = pd.DataFrame(t6_summary)
    df_t6.to_csv(os.path.join(output_dir, "table_6_cross_site_generalization.csv"), index=False)
    with open(os.path.join(output_dir, "table_6_cross_site_generalization.md"), "w") as f:
        f.write("# Table 6: Cross-Site Generalization results and Domain-Shift Gaps\n\n")
        f.write(df_t6.to_markdown(index=False))

    # ----------------------------------------------------
    # Experiment 3: Robustness to Sensor & Report Noise
    # ----------------------------------------------------
    print("Running Experiment 3: Robustness curves...")
    # We will perturb test set under four regimes
    exp3_results = []

    def append_robustness_results(perturbation_type, level, struct_df, tfidf_df, minilm_df):
        X_perturbed = {
            "struct": struct_df.values,
            "tfidf": tfidf_df.values,
            "minilm": minilm_df.values,
            "fusion_tfidf": np.hstack([struct_df.values, tfidf_df.values]),
            "fusion_minilm": np.hstack([struct_df.values, minilm_df.values])
        }
        for m_name, model in models.items():
            feat_key = model_feature_map[m_name]
            probs = model.predict_proba(X_perturbed[feat_key])[:, 1]
            preds = (probs >= 0.5).astype(int)
            exp3_results.append({
                "Model": m_name,
                "Perturbation_Type": perturbation_type,
                "Level": level,
                "PR-AUC": pr_auc_score(y_test, probs),
                "Macro-F1": f1_score(y_test, preds, average="macro"),
                "Recall": recall_score(y_test, preds),
                "FNR": 1.0 - recall_score(y_test, preds)
            })
    
    # 3.1 Sensor Noise
    multipliers = [1.0, 1.5, 2.0, 3.0]
    for mult in multipliers:
        test_df_perturbed = test_df.copy()
        for col in preprocessor.num_cols:
            std = test_df[col].std()
            noise = rng.normal(0, (mult - 1.0) * std, size=len(test_df))
            test_df_perturbed[col] = test_df_perturbed[col] + noise
            
        t_struct_p, t_tfidf_p, t_minilm_p = preprocessor.transform(test_df_perturbed)
        append_robustness_results("Sensor Noise", f"{mult}x", t_struct_p, t_tfidf_p, t_minilm_p)
        
    # 3.2 Missing Structured Values
    missing_rates = [0.0, 0.10, 0.25, 0.40]
    for rate in missing_rates:
        test_df_perturbed = test_df.copy()
        if rate > 0.0:
            for col in preprocessor.num_cols:
                mask = rng.random(len(test_df)) < rate
                test_df_perturbed.loc[mask, col] = np.nan
                
        t_struct_p, t_tfidf_p, t_minilm_p = preprocessor.transform(test_df_perturbed)
        append_robustness_results("Missing Sensors", f"{int(rate*100)}%", t_struct_p, t_tfidf_p, t_minilm_p)
        
    # 3.3 Report Missingness
    report_missing_rates = [0.0, 0.25, 0.50, 0.75]
    for rate in report_missing_rates:
        test_df_perturbed = test_df.copy()
        if rate > 0.0:
            mask = rng.random(len(test_df)) < rate
            test_df_perturbed.loc[mask, "technician_report"] = np.nan
            
        t_struct_p, t_tfidf_p, t_minilm_p = preprocessor.transform(test_df_perturbed)
        append_robustness_results("Missing Reports", f"{int(rate*100)}%", t_struct_p, t_tfidf_p, t_minilm_p)
        
    # 3.4 Report Ambiguity (Noise)
    ambiguity_rates = [0.0, 0.25, 0.50, 0.75]
    for rate in ambiguity_rates:
        test_df_perturbed = test_df.copy()
        if rate > 0.0:
            # Shuffle reports for rate fraction of cases to represent mismatch/misleading logs
            mask = rng.random(len(test_df)) < rate
            reports = test_df_perturbed["technician_report"].values.copy()
            reports[mask] = rng.permutation(reports[mask])
            test_df_perturbed["technician_report"] = reports
            
        t_struct_p, t_tfidf_p, t_minilm_p = preprocessor.transform(test_df_perturbed)
        append_robustness_results("Report Ambiguity", f"{int(rate*100)}%", t_struct_p, t_tfidf_p, t_minilm_p)
        
    df_exp3 = pd.DataFrame(exp3_results)
    df_exp3.to_csv(os.path.join(output_dir, "exp3_robustness_results.csv"), index=False)
    
    # Save Table 7: Calibration and Robustness summaries
    df_t7 = df_exp3.copy()
    df_t7.to_csv(os.path.join(output_dir, "table_7_robustness_summary.csv"), index=False)
    with open(os.path.join(output_dir, "table_7_robustness_summary.md"), "w") as f:
        f.write("# Table 7: Robustness and Performance Degradation Under Test Perturbations\n\n")
        f.write(df_t7.to_markdown(index=False))

    # ----------------------------------------------------
    # Experiment 4: Calibrated Routing & Sensitivity
    # ----------------------------------------------------
    print("Running Experiment 4: Calibrated decision-routing...")
    # Run threshold grid search on validation set for each model, for alpha = 0.05
    val_probs_dict = {}
    for name, model in models.items():
        feat_key = model_feature_map[name]
        val_probs_dict[name] = model.predict_proba(X_val[feat_key])[:, 1]
        
    y_val = val_df["maintenance_required_7d"].values
    
    t8_results = []
    t9_results = []
    
    for name, model in models.items():
        feat_key = model_feature_map[name]
        v_probs = val_probs_dict[name]
        t_probs = test_probs_dict[name]
        
        # Optimize on validation split
        t_low, t_high, val_wr, val_mcfr = optimize_thresholds(v_probs, y_val, alpha=0.05)
        
        t8_results.append({
            "Model": name,
            "t_low*": t_low,
            "t_high*": t_high,
            "Validation Workload Reduction": val_wr,
            "Validation Missed Fault Rate": val_mcfr
        })
        
        # Apply frozen thresholds on test split
        metrics = compute_routing_metrics(t_probs, y_test, t_low, t_high)
        
        # Count false urgent inspections: negative cases routed to urgent inspection
        urgent_mask = t_probs >= t_high
        negatives = y_test == 0
        false_urgents = np.sum(negatives & urgent_mask)
        total_negatives = np.sum(negatives)
        false_urgent_rate = false_urgents / total_negatives if total_negatives > 0 else 0.0
        
        # Precision of urgent decisions: true positives routed to urgent / all routed to urgent
        total_urgent = np.sum(urgent_mask)
        true_urgents = np.sum((y_test == 1) & urgent_mask)
        urgent_precision = true_urgents / total_urgent if total_urgent > 0 else 0.0
        
        t9_results.append({
            "Model": name,
            "Auto-Clear Rate": metrics["auto_clear_rate"],
            "Human-Review Rate": metrics["human_review_rate"],
            "Urgent-Inspection Rate": metrics["urgent_inspection_rate"],
            "Workload Reduction": metrics["workload_reduction"],
            "Missed Critical Fault Rate": metrics["missed_critical_fault_rate"],
            "False Urgent-Inspection Rate": false_urgent_rate,
            "Urgent-Inspection Precision": urgent_precision
        })
        
    df_t8 = pd.DataFrame(t8_results)
    df_t8.to_csv(os.path.join(output_dir, "table_8_routing_thresholds.csv"), index=False)
    with open(os.path.join(output_dir, "table_8_routing_thresholds.md"), "w") as f:
        f.write("# Table 8: Selected Routing Thresholds and Validation Objective Values\n\n")
        f.write(df_t8.to_markdown(index=False))
        
    df_t9 = pd.DataFrame(t9_results)
    df_t9.to_csv(os.path.join(output_dir, "table_9_routing_results.csv"), index=False)
    with open(os.path.join(output_dir, "table_9_routing_results.md"), "w") as f:
        f.write("# Table 9: Test-Set Human-in-the-Loop Decision-Routing Results\n\n")
        f.write(df_t9.to_markdown(index=False))
        
    # Sensitivity analysis for alpha in 0.01, 0.03, 0.05, 0.10 for every model.
    # Thresholds are always selected on validation probabilities and then frozen for test evaluation.
    sensitivity_frames = []
    for name in models.keys():
        df_model_sens = run_sensitivity_analysis(
            val_probs_dict[name],
            y_val,
            test_probs_dict[name],
            y_test
        )
        df_model_sens.insert(0, "Model", name)
        sensitivity_frames.append(df_model_sens)

    df_sens = pd.concat(sensitivity_frames, ignore_index=True)
    df_sens.to_csv(os.path.join(output_dir, "routing_sensitivity.csv"), index=False)
    print("Decision-routing threshold tuning and sensitivity analysis complete.")
    
    # ----------------------------------------------------
    # Generate Figures (1 to 10)
    # ----------------------------------------------------
    print("Generating figures...")
    
    # Figure 1: Pipeline Block Diagram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    # Draw simple blocks representing the flowchart
    boxes = [
        ("Latent Degradation\n(Hidden State $D_t$)", (0.1, 0.4)),
        ("Sensors & Reports\n(Observed Modalities)", (0.4, 0.4)),
        ("Chronological Splits\n(Train / Val / Test)", (0.7, 0.4)),
        ("Model Stack &\nPlatt Calibration", (0.7, 0.1)),
        ("Threshold Selection\n(Validation Grid Search)", (0.4, 0.1)),
        ("Decision Routing\n(Workload Reduction Layer)", (0.1, 0.1))
    ]
    for label, pos in boxes:
        ax.text(pos[0], pos[1], label, bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F7FA", edgecolor="#CCCCCC", lw=1),
                ha="center", va="center", fontsize=9)
    # Draw connecting lines / arrows
    arrowprops = dict(arrowstyle="->", color="#555555", lw=1.2)
    ax.annotate("", xy=(0.28, 0.4), xytext=(0.22, 0.4), arrowprops=arrowprops)
    ax.annotate("", xy=(0.58, 0.4), xytext=(0.52, 0.4), arrowprops=arrowprops)
    ax.annotate("", xy=(0.7, 0.18), xytext=(0.7, 0.32), arrowprops=arrowprops)
    ax.annotate("", xy=(0.52, 0.1), xytext=(0.58, 0.1), arrowprops=arrowprops)
    ax.annotate("", xy=(0.22, 0.1), xytext=(0.28, 0.1), arrowprops=arrowprops)
    ax.set_title("Figure 1: Overall Synthetic Benchmark and Evaluation Pipeline")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_1_pipeline.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Latent degradation & sensor observations
    latent_df = pd.read_parquet(os.path.join(output_dir, "latent_database.parquet"))
    # Pick a single asset that has a maintenance event and fault
    # AST_001 might have it. Let's inspect AST_002 or find one with a reset.
    selected_asset = "AST_001"
    for asset_cand in latent_df["asset_id"].unique():
        if latent_df[(latent_df["asset_id"] == asset_cand) & (latent_df["latent_maintenance_triggered"] == True)].shape[0] > 0:
            selected_asset = asset_cand
            break
            
    asset_latent = latent_df[latent_df["asset_id"] == selected_asset].sort_values(by="date")
    asset_features = df[df["asset_id"] == selected_asset].sort_values(by="date")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(asset_latent["date"], asset_latent["latent_degradation"], color="#005B94", lw=1.5, label="Latent Degradation ($D_t$)")
    ax1.axhline(0.75, color="#D9534F", linestyle="--", alpha=0.8, label="Fault Threshold (0.75)")
    ax1.set_ylabel("Degradation Level")
    ax1.set_title(f"Figure 2: Latent Degradation and Sensor Generation for Asset {selected_asset}")
    ax1.grid(True)
    ax1.legend(loc="upper left")
    
    ax2.plot(asset_features["date"], asset_features["vibration_rms"], color="#E67E22", lw=1.0, label="Observed Vibration RMS (Noisy)")
    ax2.set_ylabel("Vibration RMS")
    ax2.set_xlabel("Date")
    ax2.grid(True)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_2_degradation.png"), dpi=300)
    plt.close()
    
    # Figure 3: Label-scarcity performance curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m_name in track_models:
        sub_df = df_exp1[df_exp1["Model"] == m_name]
        ax.plot(sub_df["Fraction"] * 100, sub_df["PR-AUC"], marker="o", label=m_name, lw=1.5)
    ax.set_xlabel("Percentage of Labeled Training Data (%)")
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Figure 3: Label-Scarcity Performance Curves")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_3_learning_curves.png"), dpi=300)
    plt.close()
    
    # Figure 4: Random split versus held-out-site performance
    # Average held-out site PR-AUC vs baseline PR-AUC
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_indices = np.arange(len(track_models))
    bar_width = 0.35
    
    if random_eval_df is not None:
        baseline_pr = [random_eval_df[random_eval_df["Model"] == m]["Random-Split PR-AUC"].values[0] for m in track_models]
        baseline_label = "Random-Split Test"
    else:
        baseline_pr = [df_eval[df_eval["Model"] == m]["PR-AUC"].values[0] for m in track_models]
        baseline_label = "Chronological Test"
    held_out_pr = [df_exp2[df_exp2["Model"] == m]["PR-AUC"].mean() for m in track_models]
    
    ax.bar(x_indices - bar_width/2, baseline_pr, bar_width, label=baseline_label, color="#34495E")
    ax.bar(x_indices + bar_width/2, held_out_pr, bar_width, label="Cross-Site Average Test", color="#E74C3C")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(track_models)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Figure 4: Random Split vs. Held-Out-Site Performance")
    ax.grid(True, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_4_site_shift.png"), dpi=300)
    plt.close()
    
    # Figure 5: Robustness curves
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
    
    fig_model = "fusion_minilm"
    sub = df_exp3[(df_exp3["Perturbation_Type"] == "Sensor Noise") & (df_exp3["Model"] == fig_model)]
    ax1.plot(sub["Level"], sub["PR-AUC"], marker="o", color="#2C3E50", label="PR-AUC")
    ax1.set_ylabel("PR-AUC")
    ax1.set_title("Sensor Noise Multiplier")
    ax1.grid(True)
    
    sub = df_exp3[(df_exp3["Perturbation_Type"] == "Missing Sensors") & (df_exp3["Model"] == fig_model)]
    ax2.plot(sub["Level"], sub["PR-AUC"], marker="s", color="#16A085", label="PR-AUC")
    ax2.set_ylabel("PR-AUC")
    ax2.set_title("Missing Structured Sensors (%)")
    ax2.grid(True)
    
    sub = df_exp3[(df_exp3["Perturbation_Type"] == "Missing Reports") & (df_exp3["Model"] == fig_model)]
    ax3.plot(sub["Level"], sub["PR-AUC"], marker="^", color="#2980B9", label="PR-AUC")
    ax3.set_ylabel("PR-AUC")
    ax3.set_xlabel("Missingness Rate")
    ax3.grid(True)
    
    sub = df_exp3[(df_exp3["Perturbation_Type"] == "Report Ambiguity") & (df_exp3["Model"] == fig_model)]
    ax4.plot(sub["Level"], sub["PR-AUC"], marker="d", color="#8E44AD", label="PR-AUC")
    ax4.set_ylabel("PR-AUC")
    ax4.set_xlabel("Ambiguity Rate")
    ax4.grid(True)
    
    plt.suptitle("Figure 5: Robustness Curves under Sensor Noise and Missingness Modalities (fusion_minilm)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_5_robustness.png"), dpi=300)
    plt.close()
    
    # Figure 6: TF-IDF versus MiniLM comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m_name in ["lr_tfidf", "lr_minilm"]:
        sub_df = df_exp1[df_exp1["Model"] == m_name]
        ax.plot(sub_df["Fraction"] * 100, sub_df["PR-AUC"], marker="o", label=m_name, lw=1.5)
    ax.set_xlabel("Percentage of Labeled Training Data (%)")
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Figure 6: TF-IDF vs. MiniLM Text-Only Model Comparison under Label Scarcity")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_6_text_comparison.png"), dpi=300)
    plt.close()
    
    # Figure 7: Calibration curves
    fig, ax = plt.subplots(figsize=(6, 5))
    # Draw perfect calibration line
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # Show calibrated reliability diagram for fusion_minilm
    cal_model = models["fusion_minilm"]
    cal_feat_key = model_feature_map["fusion_minilm"]
    cal_probs = test_probs_dict["fusion_minilm"]
    
    prob_true, prob_pred = calibration_curve(y_test, cal_probs, n_bins=10)
    ax.plot(prob_pred, prob_true, "s-", color="#2C3E50", label="fusion_minilm (Calibrated)")
    
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Figure 7: Reliability Diagram (Calibration Curve)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_7_calibration.png"), dpi=300)
    plt.close()
    
    # Figure 8: Decision routing distributions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    model_names = list(models.keys())
    auto_clear = [df_t9[df_t9["Model"] == m]["Auto-Clear Rate"].values[0] * 100 for m in model_names]
    human_rev = [df_t9[df_t9["Model"] == m]["Human-Review Rate"].values[0] * 100 for m in model_names]
    urgent_insp = [df_t9[df_t9["Model"] == m]["Urgent-Inspection Rate"].values[0] * 100 for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax.bar(x - width, auto_clear, width, label="Auto-Clear (Routine)", color="#2ECC71")
    ax.bar(x, human_rev, width, label="Human Review", color="#F1C40F")
    ax.bar(x + width, urgent_insp, width, label="Urgent Inspection", color="#E74C3C")
    
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel("Routing Percentage (%)")
    ax.set_title("Figure 8: Decision-Routing Category Distribution by Model")
    ax.grid(True, axis='y')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_8_routing_distribution.png"), dpi=300)
    plt.close()
    
    # Figure 9: Sensitivity Analysis
    fig, ax = plt.subplots(figsize=(8, 4.8))
    plotted_models = ["lr_struct", "rf_struct", "xgb_struct", "fusion_minilm"]
    colors = {
        "lr_struct": "#2C3E50",
        "rf_struct": "#27AE60",
        "xgb_struct": "#C0392B",
        "fusion_minilm": "#8E44AD",
    }

    for model_name in plotted_models:
        model_sens = df_sens[df_sens["Model"] == model_name].sort_values("alpha")
        ax.plot(
            model_sens["alpha"],
            model_sens["test_workload_reduction"] * 100,
            marker="o",
            lw=2,
            color=colors[model_name],
            label=model_name,
        )

    ax.set_xlabel("Missed Fault Rate Ceiling (Alpha)")
    ax.set_ylabel("Test Workload Reduction (%)")
    ax.set_title("Figure 9: Workload Reduction Sensitivity to Safety Bound")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_9_sensitivity.png"), dpi=300)
    plt.close()
    
    # Figure 10: Explanation stability / SHAP global importance
    # Run SHAP on baseline xgb_struct model
    xgb_base = models["xgb_struct"]
    struct_cols = list(test_struct.columns)
    
    print("Generating SHAP explanations...")
    global_imp, _ = get_shap_explanations(xgb_base, X_train["struct"], X_test_dict["struct"], struct_cols, output_dir=output_dir)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    top_10 = global_imp.head(10).sort_values(by="Importance", ascending=True)
    ax.barh(top_10["Feature"], top_10["Importance"], color="#3498DB")
    ax.set_xlabel("Mean Absolute SHAP Value (Feature Importance)")
    ax.set_title("Figure 10: SHAP Global Feature Importance (xgb_struct)")
    ax.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_10_shap.png"), dpi=300)
    plt.close()
    
    # Get local case analysis for Figure 10 / Results text
    # Pick fusion_minilm threshold and label
    t_probs_fm = test_probs_dict["fusion_minilm"]
    diagnostic_threshold = 0.5
    
    cases = analyze_local_cases(
        y_test,
        t_probs_fm,
        diagnostic_threshold,
        X_test_dict["fusion_minilm"],
        list(test_struct.columns) + [f"minilm_{j}" for j in range(384)]
    )
    
    # Save local case index information for report writing
    local_cases_df = pd.DataFrame([
        {"Case": k, "Test Index": v["index"], "Calibrated Probability": v["prob"], "Diagnostic Threshold": diagnostic_threshold}
        for k, v in cases.items()
    ])
    local_cases_df.to_csv(os.path.join(output_dir, "local_case_indices.csv"), index=False)
    print("All figures and tables generated successfully in outputs directory.")

if __name__ == "__main__":
    generate_static_tables()
    build_evaluation_tables_and_plots("outputs/dataset.parquet")
