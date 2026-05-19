# Fairness Without Imputation: Fair Prediction with Missing Values

This repository contains a practical implementation and experimental analysis inspired by the paper:

> **"Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values"**  
> Jeong, Wang, & Calmon (AAAI 2022)

The project investigates how missing values affect fairness-aware machine learning systems and compares traditional machine learning models using fairness metrics on benchmark datasets.

---

# Project Objective

Traditional machine learning systems often rely on imputation methods to handle missing values before training predictive models. However, recent fairness research shows that imputation itself can introduce demographic bias when missing-value distributions differ across sensitive groups.

This project studies the relationship between missing values, predictive accuracy, and algorithmic fairness using open-source datasets and fairness evaluation frameworks.

---

# Key Features

This implementation includes fairness-aware machine learning experiments, missing-value handling analysis, comparison of multiple machine learning algorithms, fairness metric evaluation, and visualization of performance versus fairness trade-offs. The implementation also explores approximate Missing Incorporated as Attribute (MIA) concepts inspired by the original paper.

---

# Implemented Models

The project evaluates the following machine learning algorithms:

- Decision Tree Classifier  
- Random Forest Classifier  
- XGBoost Classifier  

An approximate MIA-inspired fairness-aware implementation is also explored using missing-value encoding and fairness regularization concepts.

---

# Datasets Used

The experiments were conducted using the Adult Income Dataset from the UCI Machine Learning Repository and the COMPAS Dataset from ProPublica.

The Adult Income dataset is used to predict whether an individual's income exceeds \$50K using demographic and socioeconomic attributes. The sensitive attribute used for fairness evaluation is sex.

The COMPAS dataset is used for fairness analysis in criminal justice prediction systems. The sensitive attribute used in the experiments is race.

Artificial missing values were introduced in several experiments to analyze fairness behavior under incomplete data conditions.

---

# Fairness Metrics

The project evaluates fairness using Demographic Parity Difference (DPD), Equalized Odds Difference (EOD), Disparate Impact Ratio, and Calibration Difference.

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

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Optional libraries for fairness experiments and optimization:

```bash
pip install fairlearn
pip install xgboost
pip install gurobipy
```

---

# Running the Project

Open the notebook:

```text
notebooks/fairness_project.ipynb
```

The notebook can be executed in Google Colab or Jupyter Notebook environments.

---

# Experimental Pipeline

The implementation follows the following workflow:

1. Load Adult Income or COMPAS dataset  
2. Introduce or preserve missing values  
3. Encode categorical variables  
4. Split training and testing datasets  
5. Apply missing-value handling methods  
6. Train machine learning models  
7. Evaluate fairness and accuracy metrics  
8. Visualize fairness-performance trade-offs  

---

# Experimental Results

The experiments demonstrated that XGBoost achieved the highest predictive accuracy among all evaluated models. Decision Tree models provided higher interpretability, while Random Forest improved robustness compared to single-tree models.

The fairness evaluation showed that fairness disparities still remain even after applying standard imputation methods. Equalized Odds Difference values varied significantly across models, indicating that predictive performance alone is insufficient for fairness evaluation.

The results support the central claim of the selected research paper that fairness problems caused by missing values cannot always be solved using traditional imputation alone.

---

# Important Note About the Implementation

This repository provides a practical approximation of the ideas proposed in the original paper.

The original paper introduces the Fair MIP Forest framework using Mixed Integer Programming (MIP) optimization and fairness-aware decision tree construction. Due to computational and solver limitations, the current implementation uses open-source Python frameworks, traditional machine learning algorithms, fairness evaluation metrics, and approximate MIA-inspired missing-value handling techniques.

Therefore, this implementation should be viewed as an experimental fairness-analysis framework rather than a complete reproduction of the original Fair MIP Forest optimization algorithm.

---

# Libraries Used

Python, Pandas, NumPy, Scikit-learn, Fairlearn, XGBoost, Matplotlib, Seaborn, and Gurobi were used during implementation and experimentation.

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
