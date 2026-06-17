# Youth Debt Crisis Early Warning and Policy Simulation Framework
### Counterfactual Explanation-Based XAI Approach Using Korean Welfare Panel Data

---

> **Note:** This repository is anonymized for double-blind peer review. Author information will be added upon acceptance.

---

## Overview

This repository contains the full reproducible pipeline for a study on youth debt crisis prediction and policy simulation using explainable AI (XAI) methods. The framework integrates gradient boosting ensemble models, SHAP-based feature attribution, and counterfactual explanation methods (NiCE and DiCE) to generate actionable, individual-level policy prescriptions for youth debt crisis intervention.

**Key contributions:**
- Early warning scoring model for youth debt crisis using public welfare panel data (no sensitive financial records required)
- SHAP-based causal structure analysis of debt crisis risk factors
- Cross-method counterfactual analysis (NiCE + DiCE) for algorithmic recourse
- Policy heterogeneity simulation by employment status subgroup

---

## Repository Structure

```
youth-debt-crisis/
├── data/                          # Raw and processed data (not included, see below)
│   ├── youth_debt_preprocessed.csv
│   ├── X_train_sm.csv
│   ├── X_test.csv
│   ├── y_train_sm.csv
│   └── y_test.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Data loading, DSR computation, target definition
│   ├── 02_eda.ipynb               # Exploratory data analysis
│   ├── 03_model_lgbm_xgb.ipynb   # Model training, evaluation, ensemble
│   ├── 04_shap_analysis.ipynb     # SHAP feature importance and dependence plots
│   ├── 05_nice_counterfactual.ipynb   # NiCE counterfactual explanations
│   ├── 05b_dice_counterfactual.ipynb  # DiCE counterfactual explanations
│   ├── 05c_case_comparison.ipynb      # NiCE vs DiCE case-level comparison
│   └── 06_policy_simulation.ipynb     # Policy scenario simulation
├── outputs/                       # Generated figures, tables, model files
├── requirements.txt
└── README.md
```

---

## Data

This study uses the **Korea Welfare Panel Study (KOWEPS) 19th Wave (2024)**, a publicly available dataset produced by the Korea Institute for Health and Social Affairs (KIHASA).

**How to obtain the data:**
1. Visit [https://www.koweps.re.kr](https://www.koweps.re.kr)
2. Register for a free account
3. Download `koweps_hpc19_2024_beta2.dta` (merged household-individual-child dataset)
4. Place the file in the `data/` directory

> **Important:** Raw KOWEPS data files must not be redistributed per the KOWEPS Terms of Use. Only preprocessed aggregate outputs are included in this repository.

---

## Methods

### Target Variable
Debt Service Ratio (DSR) ≥ 0.40 is used as a proxy for debt crisis, consistent with the Korean Financial Services Commission's regulatory threshold.

```
DSR = (annual interest + annual principal repayment) / total annual income
debt_crisis = 1 if DSR ≥ 0.40 else 0
```

### Models
- **LightGBM** + **XGBoost** ensemble (soft voting)
- SMOTE applied to training set for class imbalance
- 5-fold stratified cross-validation

### XAI Methods
| Method | Purpose |
|---|---|
| SHAP (TreeExplainer) | Global and local feature attribution |
| NiCE (sparsity-optimized) | Minimum-change individual recourse |
| DiCE (random method) | Diverse counterfactual pathways |

---

## Results Summary

| Metric | Value |
|---|---|
| Sample (youth, income earners) | 1,916 |
| Crisis rate | 15.6% |
| CV ROC-AUC (LightGBM) | 0.882 ± 0.027 |
| CV ROC-AUC (XGBoost) | 0.883 ± 0.035 |
| NiCE avg. features changed | 1.7 |
| DiCE avg. features changed | 1.6 |
| Top intervention variable (NiCE) | Financial institution loan |
| Top intervention variable (DiCE) | Temporary wage income |

---

## Environment Setup

```bash
# Create conda environment
conda create -n youth-debt python=3.10 -y
conda activate youth-debt

# Install dependencies
pip install "pandas==1.5.3"
pip install numpy pyreadstat scikit-learn lightgbm xgboost imbalanced-learn \
            shap nicex matplotlib seaborn plotly jupyter ipykernel \
            ipywidgets tqdm joblib scipy statsmodels

# Register Jupyter kernel
python -m ipykernel install --user --name youth-debt --display-name "Python (youth-debt)"

# For DiCE (separate environment recommended)
conda create -n diceml python=3.10 -y
conda activate diceml
pip install dice-ml lightgbm pandas numpy scikit-learn jupyter ipykernel
python -m ipykernel install --user --name diceml --display-name "Python (diceml)"
```

---

## Notebook Execution Order

Run notebooks in the following order. Notebooks 05, 05b, 05c require the respective environments noted below.

| Notebook | Environment | Description |
|---|---|---|
| 01_preprocessing | youth-debt | Data loading and feature engineering |
| 02_eda | youth-debt | Descriptive statistics and visualization |
| 03_model_lgbm_xgb | youth-debt | Model training and evaluation |
| 04_shap_analysis | youth-debt | SHAP explainability analysis |
| 05_nice_counterfactual | youth-debt | NiCE counterfactual generation |
| 05b_dice_counterfactual | diceml | DiCE counterfactual generation |
| 05c_case_comparison | youth-debt | Cross-method case comparison |
| 06_policy_simulation | youth-debt | Policy scenario simulation |

---

## Citation

> Anonymous Authors. (under review). *[Title anonymized for review]*. Submitted to [Journal anonymized for review].

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

The KOWEPS dataset is subject to its own terms of use. Please refer to [https://www.koweps.re.kr](https://www.koweps.re.kr) for data licensing information.
