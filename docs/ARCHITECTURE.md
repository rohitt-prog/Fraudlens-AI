# System Architecture

FraudLens AI uses a distributed monorepo microservice architecture.

## Overview

```
+---------------------------------------+
|              Client App               |
|      (frontend/ Next.js Dashboard)     |
+-------------------+-------------------+
                    | (JSON REST API)
                    v
+-------------------+-------------------+
|            FastAPI Backend            |
|            (backend/ Service)         |
+-------------------+-------------------+
                    | (Internal Service Call)
                    v
+-------------------+-------------------+
|         ML Inference Engine           |
|            (ml/src/inference)         |
+-------------------+-------------------+
                    | (Fitted Encoders / Weights)
                    v
+-------------------+-------------------+
|           Fitted Artifacts            |
|            (ml/artifacts/)            |
+-------------------+-------------------+
```

## Description
1. **Frontend**: Next.js App Router providing interactive transactional visualizations, alert monitoring, and explainability cards.
2. **Backend**: FastAPI serving transaction scoring requests, querying logs, and coordinating predictions.
3. **ML Service**: Python library handling XGBoost and PyTorch training pipelines, feature engineering, and inference operations using pre-trained models saved in `ml/artifacts/`.
