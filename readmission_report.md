# Hospital Readmission Prediction: Project Report

> **Data note:** the patient records are *simulated* with clinically plausible relationships. Results show how the
> method works, not real-world clinical performance. Swap in real data (e.g. the public UCI "Diabetes 130-US hospitals"
> dataset) to get real findings.

## 1. Problem
Predict which discharged patients will return within 30 days, so that limited follow-up resources (calls, home visits)
go to the highest-risk patients.

## 2. Data and cleaning
- 8,000 patients, 12 features; **16.1% were readmitted** (imbalanced, so accuracy is misleading).
- Missing values (% of rows): num_medications 2.1%, hemoglobin 9.0%, sodium 7.1%.
- 10 impossible values (age > 110, hemoglobin outside 4-20 g/dL) set to missing.
- Imputation (median + missing-indicator) is part of the modeling pipeline, so test data never influences it.

## 3. Modeling approach
- 75/25 stratified train/test split; test set untouched until the end.
- Models: Logistic Regression and Gradient Boosting. Selected by 5-fold cross-validated PR-AUC on training data only.

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC |
|:--|--:|--:|--:|--:|
| Logistic Regression | 0.734 | 0.379 | 0.735 | 0.387 |
| Gradient Boosting | 0.722 | 0.359 | 0.722 | 0.383 |

Selected model: **Logistic Regression**. No-skill PR-AUC would be about 0.161.

## 4. What the model learned
Top drivers by permutation importance: prior_admissions_12m (0.097), age (0.036), comorbidity_count (0.017), heart_failure (0.017), hemoglobin (0.016).

Risk by decile: the lowest-risk decile had a readmission rate of 4.0% and the highest 44.0%.
The top decile contains 27% of all readmissions.

## 5. Turning predictions into decisions
Assumption (illustrative): missing a readmission costs **5x** a wasted follow-up. The threshold (0.16) was chosen to
minimise that cost on training out-of-fold predictions, then applied to the test set.

At this threshold the model flags 741 of 2,000 test patients, catching **67%** of readmissions with
**29%** precision.

| Strategy | Cost per patient |
|:--|--:|
| Model (Logistic Regression) | 0.530 |
| Rule: >=2 prior admissions | 0.619 |
| Follow up with no one | 0.805 |
| Follow up with everyone | 0.839 |

Cost is 37% lower than following up with everyone and 14% lower than the simple prior-admissions rule.
These percentages depend entirely on the assumed 5:1 cost ratio, so the real cost ratio should be set with clinicians.

## 6. Fairness and robustness check (test-set subgroups)
|                        |    n |   readmit rate |   AUC |
|:-----------------------|-----:|---------------:|------:|
| Age < 55               |  529 |          0.076 | 0.628 |
| Age 55-74              |  909 |          0.149 | 0.702 |
| Age 75+                |  562 |          0.262 | 0.71  |
| Female                 |  997 |          0.157 | 0.744 |
| Male                   | 1003 |          0.165 | 0.727 |
| Dispo: Home            | 1220 |          0.157 | 0.748 |
| Dispo: Home health     |  422 |          0.145 | 0.726 |
| Dispo: Skilled nursing |  358 |          0.196 | 0.687 |

Subgroups with few patients have noisy AUCs; differences of a few points are not evidence of bias.

## 7. Conclusions
1. Routine fields give moderate discrimination (test ROC-AUC 0.73): enough to rank patients for follow-up, not enough to predict any individual's outcome confidently.
2. The strongest signals in this simulated cohort are prior_admissions_12m, age, comorbidity_count.
3. Choosing the threshold from a cost assumption, not 0.5, is what makes the model useful for an imbalanced outcome.
4. Check calibration (see dashboard) before using predicted probabilities as actual risk percentages.

## 8. Limitations
- Simulated data: relationships were built in, so recovering them is expected and says nothing about real clinical truth.
- A real deployment would need external validation, monitoring for drift, and review of subgroup performance across more attributes.
- Predictions are associations, not causes: a high-risk flag shows who to help, not what caused readmission.
