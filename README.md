# AD-Risk-Prediction

R Statistical Analysis Code for **AI-Enabled Modeling for Alzheimer's Disease Risk Prediction and Validation**.

## Overview

This repository contains the complete R code for predicting Alzheimer's Disease (AD) onset risk using multimodal clinical data and machine learning. The analysis pipeline includes:

1. **Data preparation** — variable coding, training/validation split
2. **Baseline comparison** — demographic and clinical characteristics between cohorts
3. **Univariate analysis** — t-tests and chi-square/Fisher exact tests
4. **LASSO feature selection** — 10-fold cross-validated logistic LASSO
5. **Multivariable logistic regression** — adjusted odds ratios and forest plot
6. **Machine learning models** — Random Forest and XGBoost with 5-fold CV
7. **Model evaluation** — AUC, sensitivity, specificity, PPV, NPV, Brier score
8. **Calibration analysis** — calibration curves and Hosmer-Lemeshow test
9. **Feature importance** — variable importance and partial dependence plots
10. **Decision curve analysis** — clinical net benefit assessment
11. **Sensitivity analysis** — model performance without cognitive scores (MMSE/MoCA)

## Selected Predictors (via LASSO)

| Variable | Description |
|---|---|
| `diabetes` | Diabetes mellitus (Yes/No) |
| `apoe4` | APOE ε4 carrier status |
| `csf_ptau181_abeta42` | CSF p-tau181/Aβ42 ratio |
| `folate` | Serum folate level |
| `mmse` | Mini-Mental State Examination score |
| `moca` | Montreal Cognitive Assessment score |

## Requirements

- R ≥ 4.0
- Packages: tidyverse, tableone, gtsummary, broom, glmnet, pROC, caret, randomForest, xgboost, rms, ResourceSelection, rmda, ggplot2, patchwork, vip, pdp, DescTools

## Usage

1. Place `ad_risk_dataset.csv` in the working directory
2. Run `AD_Risk_Prediction_Analysis.R` in R/RStudio
3. Output tables and figures will be generated in the working directory

## Citation

If you use this code, please cite the associated manuscript.

## License

This project is available for academic and research use.
