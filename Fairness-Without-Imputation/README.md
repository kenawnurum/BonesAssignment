# Fairness-Without-Imputation

This repository contains an implementation and experiments for the paper "Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values" (Jeong, Wang, & Calmon, AAAI-22).

Structure:

```
Fairness-Without-Imputation/
├── datasets/
├── notebooks/
│   └── fairness_project.ipynb
├── results/
│   ├── graphs
│   ├── tables
├── report/
│   └── report.pdf (add your report here)
├── README.md
└── requirements.txt
```

Run the notebook in `notebooks/fairness_project.ipynb` (Google Colab or Jupyter).

Quick CLI demo
---------------

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the demo (synthetic data):

```bash
python main.py --dataset synthetic --out results/model.joblib
```

The entrypoint `main.py` trains the paper-inspired baseline (`implementation/fair_missing_classifier.py`),
evaluates fairness metrics, and saves a joblib artifact with the model and metrics.
