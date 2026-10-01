"""
generate_data.py
Creates a simulated optical-network dataset (the original paper's live data is not public).

Each row = one end-to-end link made of 19 spans.
Per span we store 4 raw metrics:
    span_loss, target_span_loss, amp_gain, target_gain
Plus:
    osnr      : simulated OSNR (dB)
    label     : 0 = healthy, 1 = not healthy (decided from simulated OSNR)
    bad_span  : index (0-18) of the first damaged span, -1 if none
    bad_spans : ALL damaged span indices, e.g. "3;7" ("" if none) -> used to test localization

Run from the repo root:
    python src/generate_data.py
Output:
    data/link_data.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- settings you can change ----------------
SEED = 42              # fixed seed -> same data every run (important for teamwork)
N_LINKS = 6000         # number of links (paper: 6000)
N_SPANS = 19           # spans per link (paper: 19)
FAIL_FRACTION = 0.30   # about 30% of links get damaged
OSNR_THRESHOLD = 24.0  # OSNR (dB) below this -> link is "not healthy"
# ----------------------------------------------------------

rng = np.random.default_rng(SEED)


def make_targets():
    """Target (baseline) values for every span of every link."""
    target_span_loss = rng.uniform(15, 25, size=(N_LINKS, N_SPANS))
    # amplifiers are set to roughly cancel the span loss
    target_gain = target_span_loss + rng.normal(0, 0.2, size=(N_LINKS, N_SPANS))
    return target_span_loss, target_gain


def make_healthy_measurements(target_span_loss, target_gain):
    """Healthy behaviour: measured values = target + small noise."""
    span_loss = target_span_loss + rng.normal(0, 0.3, size=target_span_loss.shape)
    amp_gain = target_gain + rng.normal(0, 0.3, size=target_gain.shape)
    return span_loss, amp_gain


def damage_links(span_loss, amp_gain):
    """Pick ~30% of links and damage 1-2 of their spans."""
    is_damaged = rng.random(N_LINKS) < FAIL_FRACTION
    bad_span = np.full(N_LINKS, -1)
    all_bad = [""] * N_LINKS                             # NEW: every damaged span, e.g. "3;7"

    for i in np.where(is_damaged)[0]:
        n_bad = rng.integers(1, 3)                       # 1 or 2 damaged spans
        spans = rng.choice(N_SPANS, size=n_bad, replace=False)
        bad_span[i] = spans[0]                           # remember the first one
        all_bad[i] = ";".join(str(int(j)) for j in spans)  # NEW: remember all of them
        for j in spans:
            span_loss[i, j] += rng.uniform(2, 12)        # extra fiber loss
            amp_gain[i, j] -= rng.uniform(1, 8)          # gain drifts from target
    return span_loss, amp_gain, bad_span, all_bad        # NEW: also return all_bad


def compute_labels(span_loss, target_span_loss, amp_gain, target_gain):
    """Simulated OSNR drops as excess loss and gain error grow."""
    excess_loss = (span_loss - target_span_loss).clip(min=0).sum(axis=1)
    gain_error = (target_gain - amp_gain).clip(min=0).sum(axis=1)
    osnr = 28 - 0.35 * excess_loss - 0.25 * gain_error + rng.normal(0, 0.4, N_LINKS)
    label = (osnr < OSNR_THRESHOLD).astype(int)          # 1 = not healthy
    return osnr, label


def build_dataframe(span_loss, target_span_loss, amp_gain, target_gain,
                    osnr, label, bad_span, all_bad):
    """Flatten everything into one wide table (one row per link)."""
    cols = {}
    for name, arr in [("span_loss", span_loss), ("target_span_loss", target_span_loss),
                      ("amp_gain", amp_gain), ("target_gain", target_gain)]:
        for s in range(N_SPANS):
            cols[f"{name}_{s}"] = arr[:, s]
    cols["osnr"] = osnr
    cols["label"] = label
    cols["bad_span"] = bad_span
    cols["bad_spans"] = all_bad                          # NEW
    return pd.DataFrame(cols)


def main():
    target_span_loss, target_gain = make_targets()
    span_loss, amp_gain = make_healthy_measurements(target_span_loss, target_gain)
    span_loss, amp_gain, bad_span, all_bad = damage_links(span_loss, amp_gain)   # NEW
    osnr, label = compute_labels(span_loss, target_span_loss, amp_gain, target_gain)

    df = build_dataframe(span_loss, target_span_loss, amp_gain, target_gain,
                         osnr, label, bad_span, all_bad)                          # NEW

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "link_data.csv"
    df.to_csv(out_file, index=False)

    print(f"Saved {out_file}")
    print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
    print(f"Healthy: {(df.label == 0).sum()}   Not healthy: {(df.label == 1).sum()} "
          f"({df.label.mean():.1%})")


if __name__ == "__main__":
    main()