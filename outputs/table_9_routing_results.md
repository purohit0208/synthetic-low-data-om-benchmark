# Table 9: Test-Set Human-in-the-Loop Decision-Routing Results

| Model         |   Auto-Clear Rate |   Human-Review Rate |   Urgent-Inspection Rate |   Workload Reduction |   Missed Critical Fault Rate |   False Urgent-Inspection Rate |   Urgent-Inspection Precision |
|:--------------|------------------:|--------------------:|-------------------------:|---------------------:|-----------------------------:|-------------------------------:|------------------------------:|
| lr_struct     |          0.556955 |           0.0474545 |                 0.395591 |             0.556955 |                  0.000698324 |                       0.353656 |                     0.164196  |
| rf_struct     |          0.719182 |           0.0476818 |                 0.233136 |             0.719182 |                  0.00349162  |                       0.181447 |                     0.272373  |
| xgb_struct    |          0        |           0         |                 1        |             0        |                  0           |                       1        |                     0.0650909 |
| lr_tfidf      |          0        |           0         |                 1        |             0        |                  0           |                       1        |                     0.0650909 |
| lr_minilm     |          0        |           0         |                 1        |             0        |                  0           |                       1        |                     0.0650909 |
| fusion_tfidf  |          0        |           0         |                 1        |             0        |                  0           |                       1        |                     0.0650909 |
| fusion_minilm |          0        |           0         |                 1        |             0        |                  0           |                       1        |                     0.0650909 |