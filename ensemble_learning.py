"""
Practical Assignment 5: Ensemble Learning and Performance Evaluation
Stacking, AdaBoost and Voting ensembles compared with base classifiers.

Name: Revati Bhosale | Div: C | Batch: C2 | Roll No: 173 | PRN: 202401040048
"""
import warnings
warnings.filterwarnings("ignore")      # hide library deprecation warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

SEED = 42

# ------------------------------------------------------------------ 1. Data
data = load_breast_cancer()
X, y = data.data, data.target
print("=" * 60)
print("DATASET: Breast Cancer Wisconsin (Diagnostic)")
print("=" * 60)
print("Samples  :", X.shape[0])
print("Features :", X.shape[1])
print("Classes  :", {str(k): int(v) for k, v in zip(data.target_names, np.bincount(y))})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
print("Train set:", X_train.shape, " Test set:", X_test.shape)

# ------------------------------------------------------------ 2. Base models
def base_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
        "KNN (k=5)":           make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=SEED),
        "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED)),
        "Naive Bayes":         GaussianNB(),
    }

# --------------------------------------------------------------- 3. Ensembles
def ensemble_models():
    estimators = [
        ("lr",  base_models()["Logistic Regression"]),
        ("knn", base_models()["KNN (k=5)"]),
        ("dt",  base_models()["Decision Tree"]),
        ("svm", base_models()["SVM (RBF)"]),
        ("nb",  base_models()["Naive Bayes"]),
    ]
    # (a) Voting: majority (hard) voting of the five base classifiers
    voting = VotingClassifier(estimators=estimators, voting="hard")
    # (b) AdaBoost: decision stumps / shallow trees as weak learners
    ada_stump = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=SEED),
                                   n_estimators=100, learning_rate=1.0, random_state=SEED)
    ada_tree = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
                                  n_estimators=100, learning_rate=0.5, random_state=SEED)
    # (c) Stacking: logistic regression meta-classifier on out-of-fold predictions
    stacking = StackingClassifier(estimators=estimators,
                                  final_estimator=LogisticRegression(max_iter=1000, random_state=SEED),
                                  cv=5)
    return {"Voting (hard)": voting,
            "AdaBoost (stumps)": ada_stump,
            "AdaBoost (depth-3 trees)": ada_tree,
            "Stacking (LR meta)": stacking}

models = {**base_models(), **ensemble_models()}

# ----------------------------------------------------- 4. Train and evaluate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
print("\n" + "=" * 60)
print("TRAINING AND EVALUATION (test set = 20% hold-out)")
print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        auc = np.nan                                  # hard voting has no probabilities
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    rows.append({"Model": name,
                 "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred),
                 "ROC-AUC": auc,
                 "CV Mean": cv_scores.mean(),
                 "CV Std": cv_scores.std()})
    print(f"{name:26s} acc={rows[-1]['Accuracy']:.4f}  f1={rows[-1]['F1']:.4f}  cv={cv_scores.mean():.4f}")

results = pd.DataFrame(rows).set_index("Model")
pd.set_option("display.width", 200)
print("\n" + "=" * 60)
print("PERFORMANCE COMPARISON")
print("=" * 60)
print(results.round(4).to_string())
results.round(4).to_csv("results.csv")

best = results["F1"].idxmax()
print("\nBest model by F1-score:", best)

# --------------------------------------------- 5. Detailed reports + figures
print("\n" + "=" * 60)
print("CONFUSION MATRICES (rows = actual, columns = predicted)")
print("=" * 60)
for name in models:
    cm = confusion_matrix(y_test, fitted[name].predict(X_test))
    print(f"{name:26s} TN={cm[0,0]:2d} FP={cm[0,1]:2d} FN={cm[1,0]:2d} TP={cm[1,1]:2d}")

for name in ["Voting (hard)", "AdaBoost (stumps)", "Stacking (LR meta)"]:
    print("\nClassification report:", name)
    print(classification_report(y_test, fitted[name].predict(X_test), target_names=data.target_names, digits=4))

# Stacking meta-classifier weights
stack = fitted["Stacking (LR meta)"]
print("Stacking meta-classifier coefficients (one per base model):")
for n, w in zip([e[0] for e in stack.estimators], stack.final_estimator_.coef_[0]):
    print(f"   {n:4s}: {w:+.4f}")

# Figure 1: bar chart of metrics
plt.figure(figsize=(11, 5))
results[["Accuracy", "Precision", "Recall", "F1"]].plot(kind="bar", ax=plt.gca(), width=0.8)
plt.ylim(0.88, 1.03); plt.ylabel("Score"); plt.title("Base models vs ensemble models (test set)")
plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3); plt.legend(loc="upper center", ncol=4)
plt.tight_layout(); plt.savefig("fig_metrics.png", dpi=110); plt.close()

# Figure 2: confusion matrices of the three ensembles
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, name in zip(axes, ["Voting (hard)", "AdaBoost (stumps)", "Stacking (LR meta)"]):
    ConfusionMatrixDisplay.from_estimator(fitted[name], X_test, y_test, display_labels=data.target_names,
                                          ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=110); plt.close()

# Figure 3: ROC curves
plt.figure(figsize=(7, 6))
for name in ["Logistic Regression", "Decision Tree", "SVM (RBF)", "AdaBoost (stumps)", "Stacking (LR meta)"]:
    RocCurveDisplay.from_estimator(fitted[name], X_test, y_test, ax=plt.gca(), name=name)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4); plt.title("ROC curves"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_roc.png", dpi=110); plt.close()

# --------------------------------------------------- 6. Predict one new sample
sample = X_test[0:1]
print("\nSINGLE SAMPLE PREDICTION (first test record, actual class = %s)" % data.target_names[y_test[0]])
for name in ["Voting (hard)", "AdaBoost (stumps)", "Stacking (LR meta)"]:
    print(f"   {name:26s} -> {data.target_names[fitted[name].predict(sample)[0]]}")
