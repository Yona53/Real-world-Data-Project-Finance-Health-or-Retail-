"""
Applied Data Project (Health): predicting 30-day hospital readmission
NOTE: the patient data here is SIMULATED (clinically plausible, but not real).
To use real data, replace load_data() with pd.read_csv(...) and set TARGET/feature lists.
Run: python readmission_project.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)

OUT = "/mnt/user-data/outputs"
SEED = 42
rng = np.random.default_rng(SEED)
sns.set_theme(style="whitegrid", font_scale=0.95)
pal = sns.color_palette("deep")
TARGET = "readmitted_30d"

# ------------------------------------------------------------------
# 1. DATA (simulated, with realistic mess)
# ------------------------------------------------------------------
def load_data(n=8000):
    age = np.clip(rng.normal(65, 16, n), 18, 95).round()
    sex = rng.choice(["F", "M"], n)
    prior = rng.poisson(np.clip(0.3 + (age - 40) * 0.02, 0.1, 2.5))        # admissions in prior 12 months
    comorb = rng.poisson(np.clip(0.5 + (age - 30) * 0.04, 0.2, 4))          # chronic conditions
    hf = rng.random(n) < (0.05 + 0.002 * (age - 18) + 0.03 * comorb).clip(0, 0.6)
    ckd = rng.random(n) < (0.04 + 0.0025 * (age - 18)).clip(0, 0.5)
    los = np.clip(rng.gamma(2.0, 2.2, n) + 1.5 * hf + 0.5 * comorb, 1, 30).round()
    meds = np.clip(rng.normal(8 + 1.5 * comorb + 0.1 * los, 3), 1, 40).round()
    emergency = rng.random(n) < 0.55
    dispo = rng.choice(["Home", "Home health", "Skilled nursing"], n,
                       p=[.62, .2, .18])
    hgb = np.clip(rng.normal(13.2 - 0.6 * ckd - 0.15 * comorb, 1.6), 6, 18).round(1)
    sodium = np.clip(rng.normal(139 - 1.5 * hf, 3.5), 120, 150).round()

    # true risk: nonlinear + one interaction + unobserved noise (so no model is perfect)
    z = (-3.35 + 0.025 * (age - 65) + 0.42 * prior + 0.14 * comorb + 0.35 * hf + 0.30 * ckd
         + 0.04 * np.minimum(los, 10) + 0.012 * (meds - 8) + 0.25 * emergency
         + 0.30 * (dispo == "Skilled nursing") - 0.20 * (dispo == "Home health")
         - 0.12 * (hgb - 13) + 0.25 * hf * np.minimum(prior, 3)
         + 0.04 * np.maximum(0, 135 - sodium) * 3
         + rng.normal(0, 0.9, n))
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(dict(age=age, sex=sex, prior_admissions_12m=prior, comorbidity_count=comorb,
                           heart_failure=hf.astype(int), ckd=ckd.astype(int), length_of_stay=los,
                           num_medications=meds, emergency_admission=emergency.astype(int),
                           discharge_disposition=dispo, hemoglobin=hgb, sodium=sodium))
    df[TARGET] = y
    # realistic mess
    df.loc[rng.random(n) < 0.09, "hemoglobin"] = np.nan
    df.loc[rng.random(n) < 0.07, "sodium"] = np.nan
    df.loc[rng.random(n) < 0.02, "num_medications"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "hemoglobin"] = 0       # impossible lab values
    df.loc[rng.choice(n, 4, replace=False), "age"] = 150            # impossible ages
    return df

raw = load_data()

# ------------------------------------------------------------------
# 2. CLEANING
# ------------------------------------------------------------------
df = raw.copy()
bad_age = ~df["age"].between(18, 110)
bad_hgb = df["hemoglobin"].notna() & ~df["hemoglobin"].between(4, 20)
df.loc[bad_age, "age"] = np.nan
df.loc[bad_hgb, "hemoglobin"] = np.nan
n_invalid = int(bad_age.sum() + bad_hgb.sum())
missing = df.drop(columns=TARGET).isna().mean().mul(100).round(1)
missing = missing[missing > 0]
prevalence = df[TARGET].mean()

# ------------------------------------------------------------------
# 3. SPLIT + MODELS (imputation lives inside the pipeline: no leakage)
# ------------------------------------------------------------------
num = ["age", "prior_admissions_12m", "comorbidity_count", "length_of_stay",
       "num_medications", "hemoglobin", "sodium"]
binary = ["heart_failure", "ckd", "emergency_admission"]
cat = ["sex", "discharge_disposition"]

def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), num),
        ("bin", "passthrough", binary),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])

X, y = df.drop(columns=TARGET), df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

models = {
    "Logistic Regression": Pipeline([("p", prep()), ("m", LogisticRegression(max_iter=2000))]),
    "Gradient Boosting": Pipeline([("p", prep()), ("m", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0, random_state=SEED))]),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof, cv_scores = {}, {}
for name, m in models.items():
    oof[name] = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    cv_scores[name] = (roc_auc_score(y_tr, oof[name]), average_precision_score(y_tr, oof[name]))
    m.fit(X_tr, y_tr)

# choose the final model by cross-validated PR-AUC (training data only)
best = max(cv_scores, key=lambda k: cv_scores[k][1])
final = models[best]
p_te = {n: m.predict_proba(X_te)[:, 1] for n, m in models.items()}

# simple clinical-style baseline: flag patients with >= 2 prior admissions
rule_flag = (X_te["prior_admissions_12m"] >= 2).astype(int)

# ------------------------------------------------------------------
# 4. DECISION THRESHOLD FROM ASSUMED COSTS (chosen on training OOF only)
#    Assumption: missing a readmission (FN) costs 5x a wasted follow-up (FP).
# ------------------------------------------------------------------
C_FN, C_FP = 5, 1
ths = np.linspace(0.02, 0.6, 117)
def cost(yt, yp, t):
    f = yp >= t
    return (C_FN * ((~f) & (yt == 1)).sum() + C_FP * (f & (yt == 0)).sum())
oof_costs = [cost(y_tr.values, oof[best], t) for t in ths]
thr = float(ths[int(np.argmin(oof_costs))])
flag = (p_te[best] >= thr)
cm = confusion_matrix(y_te, flag)
tn, fp, fn, tp = cm.ravel()
strategies = {
    "Follow up with no one": C_FN * y_te.sum(),
    "Follow up with everyone": C_FP * (y_te == 0).sum(),
    "Rule: >=2 prior admissions": C_FN * ((rule_flag == 0) & (y_te == 1)).sum() + C_FP * ((rule_flag == 1) & (y_te == 0)).sum(),
    f"Model ({best})": C_FN * fn + C_FP * fp,
}
strat = pd.Series(strategies) / len(y_te)

# permutation importance on the test set (original features)
pi = permutation_importance(final, X_te, y_te, scoring="average_precision",
                            n_repeats=15, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)

# subgroup performance
def sub_auc(mask):
    return roc_auc_score(y_te[mask], p_te[best][mask]) if y_te[mask].nunique() == 2 else np.nan
groups = {
    "Age < 55": X_te["age"].fillna(65) < 55,
    "Age 55-74": X_te["age"].fillna(65).between(55, 74),
    "Age 75+": X_te["age"].fillna(65) >= 75,
    "Female": X_te["sex"] == "F", "Male": X_te["sex"] == "M",
    **{f"Dispo: {d}": X_te["discharge_disposition"] == d for d in sorted(X["discharge_disposition"].unique())},
}
sub = pd.DataFrame({"n": {k: int(v.sum()) for k, v in groups.items()},
                    "readmit rate": {k: y_te[v].mean() for k, v in groups.items()},
                    "AUC": {k: sub_auc(v) for k, v in groups.items()}})

# lift by risk decile
dec = pd.DataFrame({"y": y_te.values, "p": p_te[best]})
dec["decile"] = pd.qcut(dec["p"].rank(method="first"), 10, labels=range(1, 11))
dec_rate = dec.groupby("decile", observed=True)["y"].mean()
top_dec_share = dec[dec.decile == 10]["y"].sum() / dec["y"].sum()

# ------------------------------------------------------------------
# 5. FIGURES
# ------------------------------------------------------------------
fig, ax = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Predicting 30-day hospital readmission (simulated patient data)", fontsize=20, fontweight="bold")

# (a) risk by prior admissions
a = ax[0, 0]
g = df.groupby(df["prior_admissions_12m"].clip(upper=4))[TARGET].agg(["mean", "size"])
a.bar(g.index.astype(str).str.replace("4", "4+"), g["mean"], color=pal[0])
a.axhline(prevalence, color="r", ls="--", label=f"Overall {prevalence:.0%}")
a.set_title("Readmission rate by prior admissions", fontweight="bold"); a.set_ylabel("Rate"); a.legend()
# (b) risk by age band
a = ax[0, 1]
ab = pd.cut(df["age"], [17, 45, 55, 65, 75, 85, 110])
g = df.groupby(ab, observed=True)[TARGET].mean()
a.bar(g.index.astype(str), g.values, color=pal[1]); a.tick_params(axis="x", rotation=30)
a.axhline(prevalence, color="r", ls="--"); a.set_title("Readmission rate by age", fontweight="bold")
# (c) comorbid conditions
a = ax[0, 2]
rows = []
for c in ["heart_failure", "ckd", "emergency_admission"]:
    rows.append({"flag": c, "No": df.loc[df[c] == 0, TARGET].mean(), "Yes": df.loc[df[c] == 1, TARGET].mean()})
r = pd.DataFrame(rows).set_index("flag")
r.plot.bar(ax=a, rot=0, color=[pal[7], pal[3]]); a.set_title("Rate by clinical flag", fontweight="bold")
# (d) ROC
a = ax[0, 3]
for n, p in p_te.items():
    fpr, tpr, _ = roc_curve(y_te, p); a.plot(fpr, tpr, lw=2, label=f"{n} (AUC {roc_auc_score(y_te, p):.3f})")
a.plot([0, 1], [0, 1], "k--", alpha=.5); a.legend(loc="lower right")
a.set_title("ROC (test set)", fontweight="bold"); a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
# (e) PR
a = ax[1, 0]
for n, p in p_te.items():
    pr, rc, _ = precision_recall_curve(y_te, p)
    a.plot(rc, pr, lw=2, label=f"{n} (AP {average_precision_score(y_te, p):.3f})")
a.axhline(prevalence, color="k", ls="--", alpha=.5, label="No skill")
a.set_xlabel("Recall"); a.set_ylabel("Precision"); a.legend(); a.set_title("Precision-recall (test set)", fontweight="bold")
# (f) calibration
a = ax[1, 1]
for n, p in p_te.items():
    fx, mx = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
    a.plot(mx, fx, marker="o", label=f"{n} (Brier {brier_score_loss(y_te, p):.3f})")
a.plot([0, .6], [0, .6], "k--", alpha=.5); a.legend()
a.set_xlabel("Predicted risk"); a.set_ylabel("Observed rate"); a.set_title("Calibration", fontweight="bold")
# (g) permutation importance
a = ax[1, 2]
s = imp.head(8).iloc[::-1]; a.barh(s.index, s.values, color=pal[2])
a.set_title("What drives predictions\n(permutation importance, AP drop)", fontweight="bold")
# (h) cost per patient
a = ax[1, 3]
s = strat.sort_values()
a.barh(s.index, s.values, color=[pal[2] if "Model" in i else pal[7] for i in s.index])
a.set_title(f"Cost per patient (FN = {C_FN}x FP)", fontweight="bold"); a.set_xlabel("Cost units per patient (lower is better)")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(f"{OUT}/readmission_dashboard.png", dpi=110); plt.close()

# ------------------------------------------------------------------
# 6. REPORT (all numbers computed)
# ------------------------------------------------------------------
auc = {n: roc_auc_score(y_te, p) for n, p in p_te.items()}
ap = {n: average_precision_score(y_te, p) for n, p in p_te.items()}
sens, ppv = tp / (tp + fn), tp / (tp + fp) if (tp + fp) else float("nan")
save_vs_all = 1 - strat[f"Model ({best})"] / strat["Follow up with everyone"]
save_vs_rule = 1 - strat[f"Model ({best})"] / strat["Rule: >=2 prior admissions"]

rep = f"""# Hospital Readmission Prediction: Project Report

> **Data note:** the patient records are *simulated* with clinically plausible relationships. Results show how the
> method works, not real-world clinical performance. Swap in real data (e.g. the public UCI "Diabetes 130-US hospitals"
> dataset) to get real findings.

## 1. Problem
Predict which discharged patients will return within 30 days, so that limited follow-up resources (calls, home visits)
go to the highest-risk patients.

## 2. Data and cleaning
- {len(raw):,} patients, {X.shape[1]} features; **{prevalence:.1%} were readmitted** (imbalanced, so accuracy is misleading).
- Missing values (% of rows): {", ".join(f"{k} {v}%" for k, v in missing.items())}.
- {n_invalid} impossible values (age > 110, hemoglobin outside 4-20 g/dL) set to missing.
- Imputation (median + missing-indicator) is part of the modeling pipeline, so test data never influences it.

## 3. Modeling approach
- 75/25 stratified train/test split; test set untouched until the end.
- Models: Logistic Regression and Gradient Boosting. Selected by 5-fold cross-validated PR-AUC on training data only.

| Model | CV ROC-AUC | CV PR-AUC | Test ROC-AUC | Test PR-AUC |
|:--|--:|--:|--:|--:|
""" + "\n".join(f"| {n} | {cv_scores[n][0]:.3f} | {cv_scores[n][1]:.3f} | {auc[n]:.3f} | {ap[n]:.3f} |" for n in models) + f"""

Selected model: **{best}**. No-skill PR-AUC would be about {prevalence:.3f}.

## 4. What the model learned
Top drivers by permutation importance: {", ".join(f"{i} ({v:.3f})" for i, v in imp.head(5).items())}.

Risk by decile: the lowest-risk decile had a readmission rate of {dec_rate.iloc[0]:.1%} and the highest {dec_rate.iloc[-1]:.1%}.
The top decile contains {top_dec_share:.0%} of all readmissions.

## 5. Turning predictions into decisions
Assumption (illustrative): missing a readmission costs **{C_FN}x** a wasted follow-up. The threshold ({thr:.2f}) was chosen to
minimise that cost on training out-of-fold predictions, then applied to the test set.

At this threshold the model flags {int(flag.sum()):,} of {len(y_te):,} test patients, catching **{sens:.0%}** of readmissions with
**{ppv:.0%}** precision.

| Strategy | Cost per patient |
|:--|--:|
""" + "\n".join(f"| {k} | {v:.3f} |" for k, v in strat.sort_values().items()) + f"""

Cost is {save_vs_all:.0%} lower than following up with everyone and {save_vs_rule:.0%} lower than the simple prior-admissions rule.
These percentages depend entirely on the assumed 5:1 cost ratio, so the real cost ratio should be set with clinicians.

## 6. Fairness and robustness check (test-set subgroups)
{sub.round(3).to_markdown()}

Subgroups with few patients have noisy AUCs; differences of a few points are not evidence of bias.

## 7. Conclusions
1. Routine fields give moderate discrimination (test ROC-AUC {auc[best]:.2f}): enough to rank patients for follow-up, not enough to predict any individual's outcome confidently.
2. The strongest signals in this simulated cohort are {', '.join(imp.head(3).index)}.
3. Choosing the threshold from a cost assumption, not 0.5, is what makes the model useful for an imbalanced outcome.
4. Check calibration (see dashboard) before using predicted probabilities as actual risk percentages.

## 8. Limitations
- Simulated data: relationships were built in, so recovering them is expected and says nothing about real clinical truth.
- A real deployment would need external validation, monitoring for drift, and review of subgroup performance across more attributes.
- Predictions are associations, not causes: a high-risk flag shows who to help, not what caused readmission.
"""
open(f"{OUT}/readmission_report.md", "w").write(rep)
print(rep)
