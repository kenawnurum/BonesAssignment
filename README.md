# Fairness-Without-Imputation
# Fairness Without Imputation: Fair Prediction with Missing Values

This repository contains a practical implementation and experimental analysis inspired by the paper:

> **"Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values"**  
> Jeong, Wang, & Calmon (AAAI 2022)

The project investigates how missing values affect fairness-aware machine learning systems and compares traditional machine learning models using fairness metrics on benchmark datasets.

---

# Project Objective

Traditional machine learning systems often rely on imputation methods to handle missing values before training predictive models. However, recent fairness research shows that imputation itself can introduce demographic bias when missing-value distributions differ across sensitive groups.

This project studies the relationship between:
- Missing values
- Predictive accuracy
- Algorithmic fairness

using open-source datasets and fairness evaluation frameworks.

---

# Key Features

- Fairness-aware machine learning experiments
- Missing value handling analysis
- Comparison of multiple ML algorithms
- Fairness metric evaluation
- Performance vs fairness trade-off visualization
- Open-source reproducible implementation

---

# Implemented Models

The following machine learning models were implemented and evaluated:

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Additionally, an approximate MIA-inspired fairness implementation was explored using:
- Missing Incorporated as Attribute (MIA) encoding
- Fairness-aware regularization concepts

---

# Datasets Used

## 1. Adult Income Dataset
- Source: UCI Machine Learning Repository
- Task: Predict whether income exceeds \$50K
- Sensitive attribute: Sex

## 2. COMPAS Dataset
- Source: ProPublica COMPAS dataset
- Task: Fairness analysis in criminal justice predictions
- Sensitive attribute: Race

Artificial missing values were introduced in some experiments to analyze fairness under incomplete data conditions.

---

# Fairness Metrics

The project evaluates fairness using:

- Demographic Parity Difference (DPD)
- Equalized Odds Difference (EOD)
- Disparate Impact Ratio
- Calibration Difference

---

# Project Structure

```text
Fairness-Without-Imputation/
├── datasets/
├── notebooks/
│   └── fairness_project.ipynb
├── implementation/
│   └── fairness_withoutimputation.py
├── results/
│   ├── graphs/
│   ├── tables/
├── report/
│   └── report.pdf
├── README.md
└── requirements.txt
```

---

# Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

Optional libraries:

```bash
pip install fairlearn
pip install xgboost
pip install gurobipy
```

---

# Running the Project

## Google Colab / Jupyter Notebook

Open:

```text
notebooks/fairness_project.ipynb
```

and run all cells sequentially.

---

# Main Experimental Pipeline

The implementation follows these steps:

1. Load Adult Income or COMPAS dataset
2. Introduce or preserve missing values
3. Encode categorical variables
4. Split training and testing data
5. Apply missing-value handling
6. Train ML models
7. Evaluate accuracy and fairness metrics
8. Visualize fairness-performance trade-offs

---

# Experimental Results

The experiments showed:

- XGBoost achieved the highest predictive accuracy
- Decision Tree provided better interpretability
- Random Forest improved robustness over single trees
- Fairness disparities remained present even with imputation
- Equalized Odds Difference varied significantly across models

The results support the paper’s central claim that:
> fairness issues caused by missing data cannot always be solved through imputation alone.

---

# Important Note About the Implementation

This repository provides a **practical approximation** of the ideas proposed in the original paper.

The original paper introduces:
- Fair MIP Forest
- Mixed Integer Programming (MIP)
- Optimization-based fairness constraints

Due to computational and solver limitations, the current implementation uses:
- Open-source Python frameworks
- Standard machine learning algorithms
- Fairness evaluation metrics
- Approximate MIA-inspired missing-value handling

Therefore, this project should be viewed as:
- an experimental fairness analysis framework,
- not a complete reproduction of the original optimization-based Fair MIP Forest algorithm.

---

# Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Fairlearn
- XGBoost
- Matplotlib
- Seaborn
- Gurobi (experimental section)

---

# Reference

```bibtex
@inproceedings{jeong2022fairness,
  title={Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values},
  author={Jeong, Haewon and Wang, Hao and Calmon, Flavio P.},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={9},
  pages={9558--9566},
  year={2022}
}
```

---

# Author

Kenaw Nuru  
IIT Madras

