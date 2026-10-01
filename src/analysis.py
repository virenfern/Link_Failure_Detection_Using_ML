"""
analysis.py
Extra analysis on top of train_model.py (reuses its data loading, split and models):

  1. coefficients.csv          theta0, theta1, theta2 for LogReg, GDA, Linear SVM (original units)
  2. model_comparison.csv      accuracy / precision / recall / F1 for all models (+ RBF SVM, QDA)
  3. fig_svm.png               linear SVM decision boundary
  4. fig_qda.png               QDA (curved) decision boundary
  5. confusion_matrices.png    confusion matrices for all models
  6. newton_convergence.png    logistic-regression cost per Newton iteration
  7. cross_validation.csv      5-fold stratified CV, mean +/- std

Run from the repo root (after generate_data.py):
    python src/analysis.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:                                    # works whichever name you gave your training file
    from train_model import load_features, LogisticRegressionNewton, GDA, ROOT, SEED
except ImportError:
    from train_models import load_features, LogisticRegressionNewton, GDA, ROOT, SEED

RES = ROOT / "results"


# ------------------------------------------------------------ fitting
def fit_all(Xtr, ytr):
    """Train every model. Returns (dict name -> predict function, dict of extras)."""
    lr = LogisticRegressionNewton().fit(Xtr, ytr)
    gda = GDA().fit(Xtr, ytr)
    sc = StandardScaler().fit(Xtr)
    Ztr = sc.transform(Xtr)
    svm = LinearSVC(max_iter=20000).fit(Ztr, ytr)
    rbf = SVC(kernel="rbf").fit(Ztr, ytr)
    qda = QuadraticDiscriminantAnalysis().fit(Xtr, ytr)

    # convert the SVM (trained on standardised data) back to original X1/X2 units
    w = svm.coef_[0] / sc.scale_
    b = svm.intercept_[0] - np.sum(svm.coef_[0] * sc.mean_ / sc.scale_)

    predict = {
        "Logistic Regression": lr.predict,
        "GDA": gda.predict,
        "Linear SVM": lambda X: svm.predict(sc.transform(X)),
        "RBF SVM": lambda X: rbf.predict(sc.transform(X)),
        "QDA": qda.predict,
    }
    extras = {"lr": lr, "theta": {"Logistic Regression": lr.theta, "GDA": gda.theta,
                                  "Linear SVM": np.r_[b, w]}, "qda": qda}
    return predict, extras


def scores(y, p):
    return {"Accuracy": (p == y).mean(),
            "Precision": precision_score(y, p, zero_division=0),
            "Recall": recall_score(y, p, zero_division=0),
            "F1": f1_score(y, p, zero_division=0)}


# ------------------------------------------------------------ plots
def plot_line(X, y, theta, title, path):
    plt.figure(figsize=(5.5, 4.8))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="g", marker="x", s=10, label="healthy")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="k", s=10, label="not healthy")
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    plt.plot(xs, -(theta[0] + theta[1] * xs) / theta[2], "b-", label="decision boundary")
    plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
    plt.xlabel("X1"); plt.ylabel("X2"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_region(X, y, predict_fn, title, path):
    """Curved boundary: colour the plane by the model's prediction."""
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
                         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300))
    zz = predict_fn(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    plt.figure(figsize=(5.5, 4.8))
    plt.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5], colors=["#d9f2d9", "#e0e0e0"])
    plt.contour(gx, gy, zz, levels=[0.5], colors="b")
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="g", marker="x", s=10, label="healthy")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="k", s=10, label="not healthy")
    plt.xlabel("X1"); plt.ylabel("X2"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ------------------------------------------------------------ main
def main():
    RES.mkdir(exist_ok=True)
    X, y, x1, x2, bad_sets = load_features(ROOT / "data" / "link_data.csv")

    # same 75/25 split as train_model.py
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    predict, extras = fit_all(Xtr, ytr)

    # 1. coefficient table (paper's Table 1)
    coef = pd.DataFrame([{"Classifier": n, "theta0": t[0], "theta1": t[1], "theta2": t[2]}
                         for n, t in extras["theta"].items()]).round(4)
    coef.to_csv(RES / "coefficients.csv", index=False)
    print("Coefficients (boundary: theta0 + theta1*X1 + theta2*X2 = 0)\n", coef.to_string(index=False))

    # 2. model comparison on the test set
    rows = [{"Classifier": n, **scores(yte, f(Xte))} for n, f in predict.items()]
    comp = pd.DataFrame(rows).round(4)
    comp.to_csv(RES / "model_comparison.csv", index=False)
    print("\nTest-set comparison\n", comp.to_string(index=False))

    # 3-4. boundary plots for the models not plotted by train_model.py
    plot_line(Xte, yte, extras["theta"]["Linear SVM"], "Linear SVM", RES / "fig_svm.png")
    plot_region(Xte, yte, predict["QDA"], "QDA (separate covariance per class)", RES / "fig_qda.png")

    # 5. confusion matrices
    names = list(predict)
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.4))
    for ax, n in zip(axes, names):
        ConfusionMatrixDisplay(confusion_matrix(yte, predict[n](Xte)),
                               display_labels=["healthy", "not healthy"]).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(n, fontsize=9)
        ax.set_xticklabels(["healthy", "not healthy"], fontsize=7)
        ax.set_yticklabels(["healthy", "not healthy"], fontsize=7, rotation=90, va="center")
    plt.tight_layout(); plt.savefig(RES / "confusion_matrices.png", dpi=150); plt.close()

    # 6. Newton's method convergence
    losses = extras["lr"].losses
    plt.figure(figsize=(5, 3.8))
    plt.plot(range(1, len(losses) + 1), losses, "o-")
    plt.xlabel("Newton iteration"); plt.ylabel("cost J(theta)")
    plt.title("Logistic regression: Newton convergence")
    plt.tight_layout(); plt.savefig(RES / "newton_convergence.png", dpi=150); plt.close()
    print(f"\nLogReg cost J(theta) = {losses[-1]:.5f} after {len(losses)} Newton iterations")
    print("Cost per iteration:", np.round(losses, 5))

    # 7. 5-fold stratified cross-validation on the full dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = {n: [] for n in names}
    for a, b in skf.split(X, y):
        pred, _ = fit_all(X[a], y[a])
        for n in names:
            folds[n].append(scores(y[b], pred[n](X[b])))
    cv_rows = []
    for n in names:
        d = pd.DataFrame(folds[n])
        cv_rows.append({"Classifier": n, **{f"{m} (mean)": d[m].mean() for m in d},
                        "Accuracy (std)": d["Accuracy"].std(), "Recall (std)": d["Recall"].std()})
    cv = pd.DataFrame(cv_rows).round(4)
    cv.to_csv(RES / "cross_validation.csv", index=False)
    print("\n5-fold cross-validation\n", cv.to_string(index=False))
    print("\nSaved all outputs to results/")


if __name__ == "__main__":
    main()