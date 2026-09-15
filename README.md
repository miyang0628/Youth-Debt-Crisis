# Measuring Latent Youth Financial Vulnerability

### A Mixed Explainable-Machine-Learning Framework for Indicator Construction, Validity Assessment, and Policy-Relevant Interpretation Using Korean Welfare Panel Data

---

> **Note:** This repository is anonymized for double-blind peer review. Author and affiliation information will be added upon acceptance.

---

## Overview

This repository contains the full reproducible pipeline for a study that treats **youth financial vulnerability as a latent social construct** and asks how it can be operationalized into an indicator that is at once **measurable, interpretable, and verifiable**. The framing is one of social-science measurement rather than of algorithmic prediction: no new learner is proposed. Instead, established tools are bound into a single measurement logic of *construction → interpretation → validation*.

Using the 19th Wave (2024) of the Korea Welfare Panel Study (KOWEPS, *n* = 1,916 youth aged 19–39), the pipeline (i) constructs a debt-service-ratio (DSR)-based crisis indicator, (ii) estimates its manifestation with a LightGBM–XGBoost ensemble treated as a **nonlinear measurement instrument**, (iii) renders the instrument interpretable through SHAP as a **quantitative–qualitative bridge**, and (iv) checks the stability of the indicator's implied structure through **cross-algorithm counterfactual agreement** (NiCE and DiCE) used as a **convergent-validity** criterion.

**Key methodological contributions:**

- Operationalization of a latent social construct into an interpretable indicator using only publicly available welfare panel data (no administrative or credit-bureau records)
- SHAP-based knowledge extraction as the qualitative half of a mixed-method indicator, surfacing nonmonotonic constituent structure
- Cross-algorithm counterfactual agreement (NiCE + DiCE) proposed as a lightweight convergent-validity check for learned social indicators
- Structural-coherence reading of subgroup heterogeneity by employment status

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
│   ├── 01_preprocessing.ipynb     # Data loading, DSR computation, indicator definition
│   ├── 02_eda.ipynb               # Descriptive structure of the construct
│   ├── 03_model_lgbm_xgb.ipynb    # Component 1: the measurement instrument
│   ├── 04_shap_analysis.ipynb     # Component 2: quantitative–qualitative bridge
│   ├── 05_nice_counterfactual.ipynb   # Component 3: NiCE validity reading
│   ├── 05b_dice_counterfactual.ipynb  # Component 3: DiCE validity reading
│   ├── 05c_case_comparison.ipynb      # Convergent-validity: NiCE vs DiCE agreement
│   └── 06_policy_simulation.ipynb     # Structural-coherence reading
├── outputs/                       # Generated figures, tables, model files
├── requirements.txt
└── README.md
```

---

## Data

This study uses the **Korea Welfare Panel Study (KOWEPS) 19th Wave (2024)**, a publicly available dataset produced by the Korea Institute for Health and Social Affairs (KIHASA).

**How to obtain the data:**

1. Visit <https://www.koweps.re.kr>
2. Register for a free account
3. Download `koweps_hpc19_2024_beta2.dta` (merged household–individual–child dataset)
4. Place the file in the `data/` directory

> **Important:** Raw KOWEPS data files must not be redistributed per the KOWEPS Terms of Use. Only preprocessed aggregate outputs are included in this repository.

The analytical sample is restricted to youth aged 19–39 with positive reported income (*n* = 1,916). DSR values are winsorized at 500% to limit the influence of extreme outliers.

---

## The Measurement Framework

The pipeline mirrors the measurement logic of construction, interpretation, and validation.

### Indicator Construction — the latent target

A binary manifest indicator of debt-service distress is defined at the regulatory ceiling of the Korean Financial Services Commission's Total-DSR framework:

```
DSR = (annual interest + annual principal repayment) / total annual income
debt_crisis = 1 if DSR ≥ 0.40 else 0
```

The three quantities used to construct the DSR (annual interest, annual repayment, total income) are excluded from the instrument's inputs to prevent target leakage.

### Component 1 — the measurement instrument

- **LightGBM** + **XGBoost** ensemble (soft voting), treated as a nonlinear estimator of the construct's manifestation rather than as a black-box classifier
- SMOTE applied to the **training set only**; the test set retains the original class distribution so evaluation reflects real conditions
- 5-fold stratified cross-validation, with SMOTE applied **within each training fold** to prevent leakage
- Discrimination (ROC-AUC) is read as a measurement-fidelity criterion — how well the instrument tracks the latent construct

### Component 2 — the quantitative–qualitative bridge

| Method                    | Role                                                         |
| ------------------------- | ----------------------------------------------------------- |
| SHAP (TreeExplainer)      | Renders the instrument's internal constituent structure legible; global + local attribution; nonmonotonicity via dependence structure |

SHAP is used not as post-hoc justification of a prediction but as the mechanism that turns a purely quantitative estimator into a mixed-method object whose qualitative structure can be read and criticized.

### Component 3 — the convergent-validity check

| Method                    | Role                                                        |
| ------------------------- | ----------------------------------------------------------- |
| NiCE (sparsity-optimized) | Nearest-unlike-neighbour readings; observed-range, parsimonious |
| DiCE (random method)      | Diversity-driven readings; wider constituent space          |

Agreement between the two algorithmically distinct procedures on **which constituents are decisive** is the convergent-validity criterion. Because the methods share no optimization principle, their convergence is difficult to explain as a shared artifact.

### Structural-coherence reading

Constituent values are perturbed and the instrument is re-read under seven stylized scenarios (A-20/40/60, B-20/40/60, C), stratified by employment status. These are **sensitivity readings of the instrument's internal structure, not causal impact estimates.**

---

## Results Summary

| Item                                          | Value                        |
| --------------------------------------------- | ---------------------------- |
| Sample (youth, income earners)                | 1,916                        |
| Distress rate (DSR ≥ 0.40)                    | 15.6%                        |
| CV ROC-AUC (LightGBM)                         | 0.882 ± 0.027                |
| CV ROC-AUC (XGBoost)                          | 0.883 ± 0.035                |
| Dominant constituent #1 (mean \|SHAP\|)       | Financial institution loan (1.682) |
| Dominant constituent #2 (mean \|SHAP\|)       | Temporary wage income (1.466) |
| Income–distress relationship                  | Nonmonotonic (stability, not level, protective) |
| Mean constituent changes — NiCE               | 1.7                          |
| Mean constituent changes — DiCE               | 1.6                          |
| Cross-method convergence                      | Both isolate financial-institution loan and temporary wage income |

All numerical results are produced by the notebooks below and are unchanged across framings of the analysis.

---

## Environment Setup

```
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

| Notebook                  | Environment | Description                                        |
| ------------------------- | ----------- | ------------------------------------------------- |
| 01\_preprocessing         | youth-debt  | Data loading, DSR construction, indicator definition |
| 02\_eda                   | youth-debt  | Descriptive structure of the construct            |
| 03\_model\_lgbm\_xgb      | youth-debt  | Component 1: measurement instrument (fidelity)    |
| 04\_shap\_analysis        | youth-debt  | Component 2: quantitative–qualitative bridge      |
| 05\_nice\_counterfactual  | youth-debt  | Component 3: NiCE validity reading                |
| 05b\_dice\_counterfactual | diceml      | Component 3: DiCE validity reading                |
| 05c\_case\_comparison     | youth-debt  | Convergent-validity: cross-method agreement       |
| 06\_policy\_simulation    | youth-debt  | Structural-coherence reading by subgroup          |

---

## Citation

> Anonymous Authors. (under review). *[Title anonymized for review]*. Submitted to *[Journal anonymized for review]*.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

The KOWEPS dataset is subject to its own terms of use. Please refer to <https://www.koweps.re.kr> for data licensing information.
