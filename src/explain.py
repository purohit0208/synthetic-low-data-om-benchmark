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
