# FinTech Innovations: Loan Approval Prediction

A machine learning classification project that predicts whether a loan
application will be approved or denied.

The project covers the full machine learning workflow: data exploration,
preprocessing, model comparison, hyperparameter tuning, evaluation, and
business recommendations. The full analysis is in
[`financial_loan_risk.ipynb`](financial_loan_risk.ipynb).

## Project Overview

FinTech Innovations uses applicant information such as income, credit
history, debt, assets, employment status, and an internal `RiskScore`
when reviewing loan applications.

The goal of this project was to build a classification model that can
reproduce historical approval decisions while accounting for the
different business costs of incorrect predictions.

A false positive is more costly than a false negative:

| Error          | Result                                    | Estimated Cost |
| -------------- | ----------------------------------------- | -------------- |
| False Positive | Approve an applicant who should be denied | $2,000         |
| False Negative | Deny an applicant who should be approved  | $500           |

Because of the class imbalance and different error costs, accuracy alone
was not enough. Model performance was compared using ROC-AUC, recall,
precision, and a custom business-cost metric.

## Dataset

The dataset contains 20,000 loan applications and 35 raw features.

| Item                             | Value            |
| -------------------------------- | ---------------- |
| Rows                             | 20,000           |
| Columns                          | 35               |
| Target                           | `LoanApproved` |
| Denied                           | 76.1%            |
| Approved                         | 23.9%            |
| Missing`EducationLevel`        | 901              |
| Missing`MaritalStatus`         | 1,331            |
| Missing`SavingsAccountBalance` | 572              |

The target is imbalanced. A model that predicted every application as
denied would already have about 76% accuracy, so accuracy by itself
would give a misleading view of model performance.

![Loan approval class distribution](images/target_distribution.png)

## Exploratory Data Analysis

The EDA focused on feature distributions, missing values, class balance,
correlations, and differences between approved and denied applications.

![Numeric feature distributions by approval outcome](images/numeric_distributions.png)

`RiskScore` showed the strongest separation between approved and denied
applications. `CreditScore`, `DebtToIncomeRatio`, previous defaults, and
other financial features also showed clear differences between the two
groups.

Categorical features also contained useful signal. `BankruptcyHistory`
and `EmploymentStatus` showed noticeable differences in approval rates.

![Approval rate by employment status, home ownership, bankruptcy history, and loan purpose](images/categorical_approval_rates.png)

![Correlation matrix of numeric features](images/correlation_heatmap.png)

One important issue is the role of `RiskScore`. Since it is an internal
score already available during the approval process, it may contain much
of the same logic used to create the historical `LoanApproved` labels.
This becomes important when interpreting the final model performance.

## Preprocessing

Preprocessing was handled with scikit-learn `Pipeline` and
`ColumnTransformer` so that transformations are fit only on the training
data.

### Numeric Features

- Median imputation for missing values
- `StandardScaler` for feature scaling

### Nominal Features

The nominal categorical features include:

- `EmploymentStatus`
- `MaritalStatus`
- `HomeOwnershipStatus`
- `BankruptcyHistory`
- `LoanPurpose`

These features use:

- Most-frequent imputation
- One-hot encoding
- Unknown-category handling for inference

### Ordinal Features

`EducationLevel` uses ordinal encoding while preserving the natural
order:

```text
High School < Associate < Bachelor < Master < Doctorate
```

Keeping preprocessing inside the model pipeline helps prevent data
leakage during training and cross-validation.

## Modeling

Two classification models were compared:

- Logistic Regression
- Random Forest

Both models used `class_weight='balanced'` to account for the class
imbalance.

The validation process used:

- Stratified train/test split
- 5-fold cross-validation
- ROC-AUC
- Recall
- Precision
- Custom business-cost scoring

## Model Comparison

| Model                         | ROC-AUC         | Recall          | Precision | Avg. Business Cost |
| ----------------------------- | --------------- | --------------- | --------- | ------------------ |
| **Logistic Regression** | **1.000** | **1.000** | 0.997     | **$4,100**   |
| Random Forest                 | 0.999           | 0.984           | 0.972     | $49,400            |

Logistic Regression performed better across the selected metrics and was
carried forward for tuning.

## Hyperparameter Tuning

Tuning was completed in two stages:

1. `RandomizedSearchCV` with 25 iterations to search a broad parameter
   range.
2. `GridSearchCV` around the best region found by the random search.

The final Logistic Regression configuration used:

```text
C ≈ 32.5
penalty = 'l1'
solver = 'liblinear'
class_weight = 'balanced'
```

`C` is the inverse of regularization strength in scikit-learn, so the
selected value represents relatively weak L1 regularization compared
with smaller values of `C`.

## Final Model Performance

The tuned model was evaluated once on the held-out test set.

![Confusion matrix and ROC curve on the test set](images/confusion_matrix_roc.png)

| Metric                  | Test Result |
| ----------------------- | ----------- |
| ROC-AUC                 | 1.000       |
| Recall                  | 1.000       |
| Precision               | 1.000       |
| Estimated Business Cost | $0          |

The model also produced recall and precision of 1.000 across the tested
`EmploymentStatus` and `HomeOwnershipStatus` groups.

These results are extremely high, which is important to recognize
instead of treating the scores as proof that the model would perform the
same way on real lending data.

The results suggest that the historical approval labels may follow a
highly deterministic rule based on features already present in the
dataset, especially `RiskScore`. Because of this, the model is better
understood as reproducing the existing approval logic than predicting
real-world loan repayment or default risk.

## Feature Importance

Logistic Regression coefficients were used to inspect which transformed
features had the largest impact on the model.

![Top 15 features by absolute logistic regression coefficient](images/feature_importance.png)

| Rank | Feature                            | Absolute Coefficient |
| ---- | ---------------------------------- | -------------------- |
| 1    | `RiskScore`                      | 27.20                |
| 2    | `BankruptcyHistory_Yes`          | 20.49                |
| 3    | `EmploymentStatus_Unemployed`    | 10.51                |
| 4    | `EmploymentStatus_Self-Employed` | 10.03                |
| 5    | `DebtToIncomeRatio`              | 8.50                 |
| 6    | `InterestRate`                   | 6.60                 |
| 7    | `CreditScore`                    | 6.16                 |
| 8    | `MonthlyIncome`                  | 6.14                 |

`RiskScore` is the strongest feature, followed by bankruptcy history and
employment status. Debt-to-income ratio, interest rate, credit score,
and monthly income also have a large effect on the model.

The tuned model was also compared against an untuned Logistic Regression
baseline to see how tuning changed the model's reliance on individual
features.

![Baseline vs. tuned feature importance comparison](images/baseline_vs_tuned_importance.png)

## Business Recommendations

1. **Use the model as decision support first.** High-confidence
   decisions could be automated while borderline applications continue
   to go through manual review.
2. **Use business cost when selecting the classification threshold.** A
   false approval and a false denial do not have the same financial
   impact, so the threshold should reflect that difference instead of
   automatically using 0.5.
3. **Continue checking performance across applicant groups.** The
   current subgroup results are strong, but production data may behave
   differently from this dataset.
4. **Test the model without `RiskScore`.** If the goal changes from
   reproducing the existing approval process to finding independent
   risk signals, removing `RiskScore` would show how much predictive
   value comes from the remaining applicant information.

## Limitations

- `LoanApproved` represents historical approval decisions, not whether
  a borrower actually repaid or defaulted on the loan.
- The model can reproduce bias or rules already present in the
  historical approval process.
- The $2,000 false-positive and $500 false-negative costs are
  project assumptions rather than portfolio-specific financial
  estimates.
- The tuning process used 25 randomized search iterations followed by
  a focused grid search rather than an exhaustive search.
- `RiskScore` may contain information closely related to the logic
  used to create the target.
- The near-perfect test results should not be treated as evidence of
  real-world lending performance without validation on actual loan
  outcomes.

## Repo Structure

| File                                                      | Description                                               |
| --------------------------------------------------------- | --------------------------------------------------------- |
| [`financial_loan_risk.ipynb`](financial_loan_risk.ipynb) | Full EDA, preprocessing, modeling, tuning, and evaluation |
| [`financial_loan_data.csv`](financial_loan_data.csv)     | Source dataset with 20,000 loan applications              |
| `images/`                                               | Visualizations generated from the analysis                |

## Running the Project

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace financial_loan_risk.ipynb
```
