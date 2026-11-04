# AIT-535-Machine-Learning
# Boston Housing Regression Analysis: Linear Regression vs. Random Forest

## 📌 Project Overview
This repository contains a comprehensive Jupyter Notebook project evaluating and comparing supervised machine learning regression models on the classic **Boston Housing dataset** (Harrison & Rubinfeld, 1978). The primary objective is to predict the continuous dependent variable `MEDV` (Median value of owner-occupied homes in $1,000s) using both parametric (Linear Regression) and non-parametric ensemble (Random Forest) architectures within automated scikit-learn pipelines.

* **Author:** Candice Bell  
* **Course:** AIT-535  
* **Repository:** https://github.com/CBell64/AIT-535-Machine-Learning/
---

## 🛠️ Methodology & Technical Pipeline
The workflow is structured across 15 reproducible steps, ensuring robust data preprocessing, model validation, and error evaluation:
1. **Data Ingestion & Cleaning:** Loaded the space-delimited Boston Housing dataset with standard column headers (`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`, `MEDV`).
2. **Train-Test Split:** Partitioned the dataset into an 80% training set and a 20% testing set to ensure unbiased out-of-sample evaluation.
3. **Pipeline Encapsulation:** Integrated feature scaling (`StandardScaler`) directly into scikit-learn pipeline objects to prevent data leakage during cross-validation (**Raschka et al., 2022, ch. 2, 3**).
4. **Hyperparameter Tuning:** Utilized `GridSearchCV` to optimize model configurations and maximize predictive accuracy.
5. **Model Evaluation:** Adapted traditional classification metrics to regression-specific standards, evaluating models using **Coefficient of Determination ($R^2$)**, **Root Mean Squared Error (RMSE)**, and **Mean Absolute Error (MAE)** (**Raschka et al., 2022, ch. 3**).

---

## 📊 Key Results & Performance Comparison

| Evaluation Metric | Linear Regression | Random Forest Regressor |
| :--- | :--- | :--- |
| **$R^2$ Score** | ~0.67 | **>0.86** |
| **RMSE** | Higher error variance | **Significantly lower error magnitude** |
| **MAE** | Moderate absolute deviation | **Optimized minimum deviation** |

* **Discussion:** The **Random Forest Regressor** significantly outperformed Linear Regression across all metrics. Residual analysis and error distribution plots indicate that real estate valuations depend heavily on complex, multi-variable interactions (e.g., non-linear dynamics between `LSTAT` and home prices). While Linear Regression is limited by strict additivity constraints, Random Forest successfully captures these non-linear boundaries (**Raschka et al., 2022, ch. 2**).

---

## 📂 Repository Structure
```text
├── AIT-535-RS-T1-BostonHousing.csv  # Raw dataset file
├── housing_analysis_notebook.ipynb  # Main 15-step Jupyter Notebook
└── README.md                        # Project documentation
