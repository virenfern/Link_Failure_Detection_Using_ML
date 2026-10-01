"""
train_model.py
Reads data/link_data.csv (from generate_data.py), builds the paper's features,
trains Logistic Regression (Newton), GDA and Linear SVM, evaluates them, and
tests span-level localization using the bad_span column.

Run from the repo root:  python src/train_models.py
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
SEED = 42


# ------------------------------------------------------------ load + features
def metric_columns(df, include, exclude=()):
    """Columns whose name matches all `include` regexes and none of `exclude`, ordered by span number."""
    cols = [c for c in df.columns
            if all(re.search(p, c, re.I) for p in include)
            and not any(re.search(p, c, re.I) for p in exclude)]
    return sorted(cols, key=lambda c: int(re.findall(r"\d+", c)[-1]))


def load_features(csv_path):
    df = pd.read_csv(csv_path)
    tl = metric_columns(df, ["target", "loss"])
    sl = metric_columns(df, ["loss"], exclude=["target"])
    ag = metric_columns(df, ["amp", "gain"])
    tg = metric_columns(df, ["target", "gain"], exclude=["loss"])
    assert len(tl) == len(sl) == len(ag) == len(tg) > 0, \
        f"column detection failed: {len(tl)}, {len(sl)}, {len(ag)}, {len(tg)} -- check your column names"
    x1 = df[ag].values - df[tg].values          # Eq.1
    x2 = df[tl].values - df[sl].values
    X = np.c_[x1.sum(1), x2.sum(1)]             # Eq.2  -> X1, X2
    y = df["label"].values.astype(int)
    bad = df["bad_span"].values.astype(int) if "bad_span" in df else None
    # optional: all damaged spans per link, stored as "3;7" (falls back to the single bad_span)
    if "bad_spans" in df:
        bad_sets = [set(int(v) for v in str(s).split(";") if s == s and str(s) != "") for s in df["bad_spans"]]
    elif bad is not None:
        bad_sets = [{int(b)} if b >= 0 else set() for b in bad]
    else:
        bad_sets = None
    return X, y, x1, x2, bad_sets


# ------------------------------------------------------------ models
class LogisticRegressionNewton:
    @staticmethod
    def _g(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y, iters=50, tol=1e-8):
        Xb = np.c_[np.ones(len(X)), X]
        th = np.zeros(Xb.shape[1])
        self.losses = []
        for _ in range(iters):
            h = self._g(Xb @ th)
            grad = Xb.T @ (h - y) / len(y)
            H = (Xb.T * (h * (1 - h))) @ Xb / len(y) + 1e-8 * np.eye(Xb.shape[1])
            step = np.linalg.solve(H, grad)
            th -= step
            hc = np.clip(self._g(Xb @ th), 1e-12, 1 - 1e-12)
            self.losses.append(-np.mean(y * np.log(hc) + (1 - y) * np.log(1 - hc)))   # Eq.3
            if np.linalg.norm(step) < tol:
                break
        self.theta = th
        return self

    def predict(self, X):
        return (self._g(np.c_[np.ones(len(X)), X] @ self.theta) >= 0.5).astype(int)


class GDA:
    def fit(self, X, y):
        phi = y.mean()
        mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
        d0, d1 = X[y == 0] - mu0, X[y == 1] - mu1
        Si = np.linalg.inv((d0.T @ d0 + d1.T @ d1) / len(y))     # shared covariance
        w = Si @ (mu1 - mu0)
        b = -0.5 * (mu1 @ Si @ mu1 - mu0 @ Si @ mu0) + np.log(phi / (1 - phi))
        self.theta = np.r_[b, w]
        return self

    def predict(self, X):
        return (np.c_[np.ones(len(X)), X] @ self.theta >= 0).astype(int)


# ------------------------------------------------------------ plotting
def plot_boundary(X, y, theta, title, path):
    plt.figure(figsize=(5.5, 4.8))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c="g", marker="x", s=10, label="healthy")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c="k", s=10, label="not healthy")
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    plt.plot(xs, -(theta[0] + theta[1] * xs) / theta[2], "b-", label="p(y|x)=0.5")
    plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
    plt.xlabel("X1"); plt.ylabel("X2"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ------------------------------------------------------------ main
def main():
    (ROOT / "results").mkdir(exist_ok=True)
    X, y, x1, x2, bad_sets = load_features(ROOT / "data" / "link_data.csv")
    print(f"Links: {len(y)} | not healthy: {y.mean():.1%} | features X1, X2 built from {x1.shape[1]} spans")

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)   # 75/25 as in the paper
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    lr = LogisticRegressionNewton().fit(Xtr, ytr)
    gda = GDA().fit(Xtr, ytr)
    sc = StandardScaler().fit(Xtr)
    svm = LinearSVC(max_iter=10000).fit(sc.transform(Xtr), ytr)
    preds = {"Logistic Regression": lr.predict(Xte), "GDA": gda.predict(Xte),
             "Linear SVM": svm.predict(sc.transform(Xte))}

    rows = []
    for name, p in preds.items():
        rows.append({"Classifier": name, "Accuracy": (p == yte).mean(),
                     "Precision": precision_score(yte, p, zero_division=0),
                     "Recall": recall_score(yte, p, zero_division=0),
                     "F1": f1_score(yte, p, zero_division=0)})
        print(f"\n{name}\n confusion matrix [[TN FP],[FN TP]]:\n{confusion_matrix(yte, p)}")
    res = pd.DataFrame(rows).round(4)
    print("\n", res.to_string(index=False))
    res.to_csv(ROOT / "results" / "classifier_results.csv", index=False)

    print(f"\nLogReg final cost: {lr.losses[-1]:.5f} after {len(lr.losses)} Newton iterations")
    print("theta (LogReg):", np.round(lr.theta, 4))
    print("theta (GDA)   :", np.round(gda.theta, 4))

    plot_boundary(Xte, yte, lr.theta, "Logistic regression", ROOT / "results" / "fig_logreg.png")
    plot_boundary(Xte, yte, gda.theta, "GDA", ROOT / "results" / "fig_gda.png")

    if bad_sets is not None:
        faulty = te[(y[te] == 1) & np.array([len(bad_sets[i]) > 0 for i in te])]
        healthy_tr = tr[ytr == 0]
        # normalise using healthy training links
        m1, s1 = x1[healthy_tr].mean(0), x1[healthy_tr].std(0) + 1e-9
        m2, s2 = x2[healthy_tr].mean(0), x2[healthy_tr].std(0) + 1e-9
        score = np.abs((x1[faulty] - m1) / s1) + np.abs((x2[faulty] - m2) / s2)
        order = np.argsort(-score, axis=1)
        top1 = np.mean([order[i, 0] in bad_sets[f] for i, f in enumerate(faulty)])
        top3 = np.mean([len(bad_sets[f] & set(order[i, :3])) > 0 for i, f in enumerate(faulty)])
        print(f"\nLocalization on {len(faulty)} faulty test links: top-1 = {top1:.1%}, top-3 = {top3:.1%} (hit = a truly damaged span)")
        pd.DataFrame([{"top1": top1, "top3": top3}]).to_csv(ROOT / "results" / "localization.csv", index=False)
    print("\nSaved tables and figures to results/")


if __name__ == "__main__":
    main()