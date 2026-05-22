import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Seed for reproducibility
RANDOM_STATE = 42

class SingleClassFallbackClassifier:
    def __init__(self, single_class_value):
        self.single_class_value = single_class_value
        self.classes_ = np.array([0, 1])
        
    def fit(self, X, y):
        pass
        
    def predict(self, X):
        return np.full(X.shape[0], self.single_class_value)
        
    def predict_proba(self, X):
        probs = np.zeros((X.shape[0], 2))
        if self.single_class_value == 0:
            probs[:, 0] = 1.0
        else:
            probs[:, 1] = 1.0
        return probs

_minilm_model = None

def get_minilm_embeddings(texts, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Computes MiniLM embeddings for a list of texts.
    Fails loudly by default if the sentence-transformer cannot be loaded.
    Set ALLOW_MOCK_MINILM=1 only for explicit offline smoke tests.
    """
    global _minilm_model
    # Ensure all elements are strings
    texts_clean = [str(t) if pd.notna(t) else "" for t in texts]
    
    try:
        from sentence_transformers import SentenceTransformer
        if _minilm_model is None:
            print(f"Attempting to load {model_name}...")
            _minilm_model = SentenceTransformer(model_name)
        embeddings = _minilm_model.encode(texts_clean, show_progress_bar=False)
        return np.array(embeddings)
    except Exception as e:
        if os.environ.get("ALLOW_MOCK_MINILM", "0") != "1":
            raise RuntimeError(
                f"Failed to load or run {model_name}. Install/cache the Hugging Face model, "
                "or set ALLOW_MOCK_MINILM=1 only for a clearly labeled offline smoke test."
            ) from e
        print(f"Warning: Failed to run sentence-transformer ({e}). Using explicit offline mock embeddings...")
        # Fallback: deterministic pseudo-embedding based on keywords in the text
        embeddings = []
        keywords = ["vibration", "bearing", "leak", "temp", "noise", "pressure", "flow", "blockage", "sensor", "drift", "overheat"]
        for text in texts_clean:
            h = hash(text) % (2**32)
            rng = np.random.default_rng(h)
            # 384 dimensions for all-MiniLM-L6-v2
            vec = rng.normal(0, 0.05, 384)
            # Add semantic signal for keywords
            for idx, kw in enumerate(keywords):
                if kw in text.lower():
                    vec[idx * 10 : (idx + 1) * 10] += 0.8
            embeddings.append(vec)
        return np.array(embeddings)

class DatasetPreprocessor:
    """
    Fits preprocessing pipeline on training data and transforms validation/test sets.
    """
    def __init__(self, use_text_embeddings=False):
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
        self.use_text_embeddings = use_text_embeddings
        
        self.num_cols = [
            "age", "operating_hours", "load_factor", "duty_cycle", 
            "ambient_temp", "humidity", "vibration_rms", "vibration_kurtosis", 
            "acoustic_level", "motor_current", "temp_deviation", 
            "pressure_deviation", "flow_rate_deviation", 
            "time_since_last_maintenance", "previous_fault_count"
        ]
        self.cat_cols = ["site_id", "asset_type", "component_type", "shift_type"]
        
    def fit(self, df):
        # Fit numerical preprocessing on the same imputed representation used
        # at transform time, so missing-value experiments remain consistent.
        num_imputed = self.num_imputer.fit_transform(df[self.num_cols])
        self.num_scaler.fit(num_imputed)

        # Fit categorical preprocessing after imputation for the same reason.
        cat_imputed = self.cat_imputer.fit_transform(df[self.cat_cols])
        self.cat_encoder.fit(cat_imputed)
        
        # Fit TF-IDF
        reports = df["technician_report"].fillna("")
        self.tfidf_vectorizer.fit(reports)
        
    def transform(self, df):
        # Process numerical
        num_imputed = self.num_imputer.transform(df[self.num_cols])
        num_scaled = self.num_scaler.transform(num_imputed)
        num_df = pd.DataFrame(num_scaled, columns=self.num_cols)
        
        # Process categorical
        cat_imputed = self.cat_imputer.transform(df[self.cat_cols])
        cat_encoded = self.cat_encoder.transform(cat_imputed)
        cat_feature_names = self.cat_encoder.get_feature_names_out(self.cat_cols)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names)
        
        # Combine structured features
        struct_features = pd.concat([num_df, cat_df], axis=1)
        
        # Process TF-IDF
        reports = df["technician_report"].fillna("")
        tfidf_features = self.tfidf_vectorizer.transform(reports).toarray()
        tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf_cols)
        
        # Process MiniLM
        if self.use_text_embeddings:
            minilm_features = get_minilm_embeddings(reports)
            minilm_cols = [f"minilm_{i}" for i in range(minilm_features.shape[1])]
            minilm_df = pd.DataFrame(minilm_features, columns=minilm_cols)
        else:
            minilm_df = pd.DataFrame()
            
        return struct_features, tfidf_df, minilm_df

def get_model_stack():
    """
    Defines the dictionary of classifiers in the stack.
    """
    models = {
        "lr_struct": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "rf_struct": RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1),
        "xgb_struct": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        
        "lr_tfidf": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "lr_minilm": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        
        "fusion_tfidf": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1),
        "fusion_minilm": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)
    }
    return models

def train_and_calibrate(train_df, val_df, target_col="maintenance_required_7d"):
    """
    Trains all models in the stack and calibrates them using validation data.
    Returns:
        preprocessor (DatasetPreprocessor)
        calibrated_models (dict): dictionary of calibrated models
        X_train_dict (dict): preprocessed features for inspection
        X_val_dict (dict)
    """
    # 1. Preprocess data
    preprocessor = DatasetPreprocessor(use_text_embeddings=True)
    preprocessor.fit(train_df)
    
    # Transform train and val
    train_struct, train_tfidf, train_minilm = preprocessor.transform(train_df)
    val_struct, val_tfidf, val_minilm = preprocessor.transform(val_df)
    
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    
    # 2. Build feature matrices for each model
    X_train_dict = {
        "struct": train_struct.values,
        "tfidf": train_tfidf.values,
        "minilm": train_minilm.values,
        "fusion_tfidf": np.hstack([train_struct.values, train_tfidf.values]),
        "fusion_minilm": np.hstack([train_struct.values, train_minilm.values])
    }
    
    X_val_dict = {
        "struct": val_struct.values,
        "tfidf": val_tfidf.values,
        "minilm": val_minilm.values,
        "fusion_tfidf": np.hstack([val_struct.values, val_tfidf.values]),
        "fusion_minilm": np.hstack([val_struct.values, val_minilm.values])
    }
    
    # Map model name to feature key
    model_feature_map = {
        "lr_struct": "struct",
        "rf_struct": "struct",
        "xgb_struct": "struct",
        "lr_tfidf": "tfidf",
        "lr_minilm": "minilm",
        "fusion_tfidf": "fusion_tfidf",
        "fusion_minilm": "fusion_minilm"
    }
    
    raw_models = get_model_stack()
    calibrated_models = {}
    
    # 3. Train and calibrate
    for name, clf in raw_models.items():
        feature_key = model_feature_map[name]
        X_t = X_train_dict[feature_key]
        X_v = X_val_dict[feature_key]
        
        print(f"Training and calibrating {name}...")
        
        # Check if we have enough positive samples to train
        if len(np.unique(y_train)) < 2:
            print(f"Warning: only one class in y_train for {name}. Fitting single class fallback classifier.")
            single_val = y_train[0]
            clf = SingleClassFallbackClassifier(single_val)
            calibrated_models[name] = clf
            continue
            
        # Fit base model
        clf.fit(X_t, y_train)
        
        # Calibrate using sigmoid (Platt scaling) on validation set
        if len(np.unique(y_val)) < 2:
            # Fallback if validation set lacks positive cases
            calibrated_models[name] = clf
        else:
            calibrated_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
            calibrated_clf.fit(X_v, y_val)
            calibrated_models[name] = calibrated_clf
            
    return preprocessor, calibrated_models, X_train_dict, X_val_dict
