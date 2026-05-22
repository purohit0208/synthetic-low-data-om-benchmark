import os
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

def get_shap_explanations(model, X_train, X_test, feature_names, output_dir="outputs"):
    """
    Computes global and local explanations using SHAP.
    Falls back to model feature importances if SHAP fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if model has a base estimator or is CalibratedClassifierCV
    # CalibratedClassifierCV wraps the base estimator in self.estimator or self.calibrated_classifiers_[0].estimator
    base_estimator = model
    if hasattr(model, "estimator"):
        base_estimator = model.estimator
    elif hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        base_estimator = model.calibrated_classifiers_[0].estimator
        
    global_importance = None
    shap_values_test = None
    
    try:
        import shap
        print("Initializing SHAP TreeExplainer...")
        # To make it fast, we can sample the test data
        n_samples = min(200, X_test.shape[0])
        X_test_sample = X_test[:n_samples]
        
        # XGBoost or RF tree explainer
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_test_sample)
        
        # Handle binary classification outputs shape (might be list [neg_shap, pos_shap] or single array)
        if isinstance(shap_values, list):
            # For list, use the positive class shap values (index 1)
            shap_values_test = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_test = shap_values
            
        # Calculate mean absolute SHAP value for global importance
        mean_shap = np.abs(shap_values_test).mean(axis=0)
        global_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": mean_shap
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
    except Exception as e:
        print(f"Warning: SHAP calculation failed or was skipped ({e}). Falling back to feature importances...")
        
    # Fallback to standard feature importances if SHAP failed
    if global_importance is None:
        if hasattr(base_estimator, "feature_importances_"):
            importances = base_estimator.feature_importances_
        elif hasattr(base_estimator, "coef_"):
            importances = np.abs(base_estimator.coef_[0])
        else:
            importances = np.ones(len(feature_names)) / len(feature_names)
            
        global_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        
        # Create mock shap values for local fallback
        shap_values_test = np.zeros((X_test.shape[0], X_test.shape[1]))
        
    # Save global importance table
    global_importance.to_csv(os.path.join(output_dir, "global_feature_importance.csv"), index=False)
    return global_importance, shap_values_test

def get_tfidf_explanations(tfidf_model, vectorizer, output_dir="outputs"):
    """
    Extracts the highest and lowest coefficient weights for word tokens in TF-IDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle calibration wrapper
    base_estimator = tfidf_model
    if hasattr(tfidf_model, "estimator"):
        base_estimator = tfidf_model.estimator
    elif hasattr(tfidf_model, "calibrated_classifiers_") and len(tfidf_model.calibrated_classifiers_) > 0:
        base_estimator = tfidf_model.calibrated_classifiers_[0].estimator
        
    if hasattr(base_estimator, "coef_"):
        coefs = base_estimator.coef_[0]
        words = vectorizer.get_feature_names_out()
        
        word_weights = pd.DataFrame({
            "Word": words,
            "Weight": coefs
        }).sort_values(by="Weight", ascending=False).reset_index(drop=True)
        
        word_weights.to_csv(os.path.join(output_dir, "tfidf_word_importance.csv"), index=False)
        return word_weights
    return None

def analyze_local_cases(y_true, y_pred_probs, threshold, X_test, feature_names):
    """
    Identifies representative indexes for TP, FP, FN, TN cases based on a threshold.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]
    
    cases = {}
    
    # Find sample for each case
    # Try to pick the most representative (e.g. high probability for TP, low for TN)
    if len(tp_indices) > 0:
        idx = tp_indices[np.argmax(y_pred_probs[tp_indices])]
        cases["TP"] = {"index": int(idx), "prob": float(y_pred_probs[idx]), "features": X_test[idx]}
    if len(fp_indices) > 0:
        idx = fp_indices[np.argmax(y_pred_probs[fp_indices])]
        cases["FP"] = {"index": int(idx), "prob": float(y_pred_probs[idx]), "features": X_test[idx]}
    if len(fn_indices) > 0:
        idx = fn_indices[np.argmin(y_pred_probs[fn_indices])]
        cases["FN"] = {"index": int(idx), "prob": float(y_pred_probs[idx]), "features": X_test[idx]}
    if len(tn_indices) > 0:
        idx = tn_indices[np.argmin(y_pred_probs[tn_indices])]
        cases["TN"] = {"index": int(idx), "prob": float(y_pred_probs[idx]), "features": X_test[idx]}
        
    return cases

def get_local_shap_case_artifacts(
    model,
    X_test,
    y_true,
    y_pred_probs,
    feature_names,
    threshold=0.5,
    output_dir="outputs",
    prefix="xgb_struct",
):
    """
    Generate local SHAP waterfall-style diagnostics for one TP, FP, FN, and TN case.

    The explanations are computed for the interpretable structured model input,
    not for dense MiniLM dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_estimator = model
    if hasattr(model, "estimator"):
        base_estimator = model.estimator
    elif hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        base_estimator = model.calibrated_classifiers_[0].estimator

    cases = analyze_local_cases(y_true, y_pred_probs, threshold, X_test, feature_names)
    if not cases:
        return pd.DataFrame()

    try:
        import matplotlib.pyplot as plt
        import shap

        explainer = shap.TreeExplainer(base_estimator)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_arr = np.asarray(expected_value).ravel()
            expected_value = expected_arr[1] if len(expected_arr) > 1 else expected_arr[0]
        case_rows = []
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        case_order = ["TP", "FP", "FN", "TN"]

        for ax, case_name in zip(axes, case_order):
            if case_name not in cases:
                ax.axis("off")
                continue

            idx = cases[case_name]["index"]
            x = X_test[idx:idx + 1]
            shap_values = explainer.shap_values(x)
            if isinstance(shap_values, list):
                values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                values = np.asarray(shap_values)[0]

            explanation = shap.Explanation(
                values=values,
                base_values=expected_value,
                data=x[0],
                feature_names=feature_names,
            )
            plt.figure(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=10, show=False)
            water_path = os.path.join(
                output_dir,
                f"figure_11_{case_name.lower()}_local_shap_waterfall.png",
            )
            plt.savefig(water_path, dpi=300, bbox_inches="tight")
            plt.close()
            plt.figure(fig.number)
            plt.sca(ax)

            top_idx = np.argsort(np.abs(values))[-8:]
            top_idx = top_idx[np.argsort(values[top_idx])]
            colors = ["#D9534F" if values[j] > 0 else "#2E86AB" for j in top_idx]
            labels = [feature_names[j] for j in top_idx]
            ax.barh(labels, values[top_idx], color=colors)
            ax.axvline(0, color="#333333", linewidth=0.8)
            ax.set_title(
                f"{case_name}: y={int(y_true[idx])}, p={y_pred_probs[idx]:.3f}, test index={idx}",
                fontsize=10,
            )
            ax.set_xlabel("Local SHAP contribution")

            for rank, j in enumerate(top_idx[::-1], start=1):
                case_rows.append({
                    "Case": case_name,
                    "Test Index": int(idx),
                    "True Label": int(y_true[idx]),
                    "Predicted Probability": float(y_pred_probs[idx]),
                    "Diagnostic Threshold": float(threshold),
                    "Rank": rank,
                    "Feature": feature_names[j],
                    "Feature Value": float(X_test[idx, j]),
                    "SHAP Contribution": float(values[j]),
                    "Abs SHAP Contribution": float(abs(values[j])),
                })

        fig.suptitle("Local SHAP diagnostics for structured XGBoost cases", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = os.path.join(output_dir, "figure_11_local_shap_cases.png")
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        local_df = pd.DataFrame(case_rows)
        local_df.to_csv(os.path.join(output_dir, "local_shap_contributions.csv"), index=False)
        return local_df
    except Exception as e:
        print(f"Warning: Local SHAP artifact generation failed ({e}).")
        return pd.DataFrame()
