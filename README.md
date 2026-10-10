# Health Insurance Premium Prediction

Regression models that predict annual health-insurance premiums from 50,000 policyholder records. Error analysis showed the model failed mainly on young customers. Splitting the data at age 25 pushed **R² to 0.997** for customers over 25.

## Results

| Model | R² (test) | Predictions > 10% off |
|---|---|---|
| Linear regression (all data) | 0.929 | – |
| XGBoost (all data) | 0.981 | 29.8% |
| **XGBoost, age > 25** | **0.997** | **0.5%** |
| XGBoost, age ≤ 25 | 0.60 | 72% |

The young segment stays hard to predict with the available features. That is a data-collection finding for the business, not a tuning problem.

## Approach

1. **Cleaning:** take the absolute value of negative dependant counts (typos), drop ages above 100, and cap income using quantiles.
2. **Features:** a medical-history risk score and encoded categorical variables. Multicollinearity was checked with VIF (all below 5).
3. **Modelling:** Linear, Ridge and XGBoost regression, with XGBoost tuned by `RandomizedSearchCV`.
4. **Error analysis:** the share of predictions more than 10% off. Large errors clustered in young customers, so the data was split at age 25 and one model was trained per segment.

**Input features:** age, gender, region, BMI category, smoking status, employment, income, medical history, insurance plan, number of dependants.

## Repository structure

```
code/
├── 01_model_full_dataset.ipynb    end-to-end model on all 50k rows
├── 02_split_by_age.ipynb          writes the two age-segment CSVs
├── 03_model_age_upto_25.ipynb     model for age ≤ 25
└── 04_model_age_above_25.ipynb    model for age > 25
data/
├── premiums.xlsx                  source data
└── data_above_25.csv, data_lessthanor_25.csv
```

## Getting started

```bash
pip install -r requirements.txt
cd code
jupyter notebook
```

Run the notebooks in order from inside `code/`. They read and write under `../data/`.

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, statsmodels, seaborn
