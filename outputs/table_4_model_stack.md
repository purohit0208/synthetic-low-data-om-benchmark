# Table 4: Model Stack and Input Modalities

| Model Type                                                  | Input Modalities                                 | Calibration Method                     |
|:------------------------------------------------------------|:-------------------------------------------------|:---------------------------------------|
| Structured-only Logistic Regression (lr_struct)             | Structured sensor & maintenance features         | Platt Scaling (Sigmoid Validation-fit) |
| Structured-only Random Forest (rf_struct)                   | Structured sensor & maintenance features         | Platt Scaling (Sigmoid Validation-fit) |
| Structured-only XGBoost (xgb_struct)                        | Structured sensor & maintenance features         | Platt Scaling (Sigmoid Validation-fit) |
| Classical Text-only TF-IDF + Logistic Regression (lr_tfidf) | Raw technician reports (TF-IDF sparse tokens)    | Platt Scaling (Sigmoid Validation-fit) |
| Modern Text-only MiniLM + Logistic Regression (lr_minilm)   | Raw technician reports (MiniLM dense embeddings) | Platt Scaling (Sigmoid Validation-fit) |
| Fusion Model: Structured + TF-IDF (fusion_tfidf)            | Concatenated Structured & TF-IDF sparse features | Platt Scaling (Sigmoid Validation-fit) |
| Fusion Model: Structured + MiniLM (fusion_minilm)           | Concatenated Structured & MiniLM dense features  | Platt Scaling (Sigmoid Validation-fit) |