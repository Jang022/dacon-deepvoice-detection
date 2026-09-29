#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare file-level fusion methods on MixFake evaluation results."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("output") / "mixfake_1200_evaluation_merged.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output") / "fusion_comparison.csv",
    )
    return p.parse_args()


def authenticity_to_label(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()

    real_values = {
        "real", "bonafide", "bona_fide", "bona-fide",
        "authentic", "genuine", "human", "r", "0"
    }
    fake_values = {
        "fake", "spoof", "synthetic", "generated",
        "deepfake", "f", "1"
    }

    if s in real_values:
        return 0
    if s in fake_values:
        return 1

    if any(k in s for k in ["bonafide", "bona fide", "authentic", "genuine", "real"]):
        return 0
    if any(k in s for k in ["spoof", "fake", "synthetic", "generated", "deepfake"]):
        return 1

    return np.nan


def calc_metrics(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))

    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    acc = float(accuracy_score(y_true, (y_score >= 0.5).astype(np.int64)))
    auc = float(roc_auc_score(y_true, y_score))

    return {
        "accuracy": acc,
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_threshold,
    }


def add_result(results, name, formula, y_true, score):
    score = np.clip(np.asarray(score, dtype=np.float64), 0.0, 1.0)
    m = calc_metrics(y_true, score)

    results.append({
        "name": name,
        "formula": formula,
        "accuracy": m["accuracy"],
        "auc": m["auc"],
        "eer": m["eer"],
        "eer_threshold": m["eer_threshold"],
    })


def main():
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)

    required = {
        "VOICE_FAKE_PROB",
        "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB",
        "MUSIC_PRESENT_PROB",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if "file_label" in df.columns:
        y_true = pd.to_numeric(df["file_label"], errors="coerce").to_numpy()
    elif "syn_authenticity" in df.columns:
        y_true = df["syn_authenticity"].apply(authenticity_to_label).to_numpy()
    else:
        raise ValueError(
            "Need either 'file_label' or 'syn_authenticity' in merged CSV."
        )

    valid = np.isfinite(y_true)
    df = df.loc[valid].reset_index(drop=True)
    y_true = y_true[valid].astype(np.int64)

    vf = df["VOICE_FAKE_PROB"].to_numpy(dtype=np.float64)
    mf = df["MUSIC_FAKE_PROB"].to_numpy(dtype=np.float64)
    vp = df["VOICE_PRESENT_PROB"].to_numpy(dtype=np.float64)
    mp = df["MUSIC_PRESENT_PROB"].to_numpy(dtype=np.float64)

    # Component-level file evidence used by the current baseline.
    v = np.clip(vp * vf, 0.0, 1.0)
    m = np.clip(mp * mf, 0.0, 1.0)

    results = []

    # ---------------------------------------------------------
    # Fixed fusion candidates
    # ---------------------------------------------------------
    add_result(
        results,
        "MAX_baseline",
        "max(v, m)",
        y_true,
        np.maximum(v, m),
    )

    add_result(
        results,
        "Noisy_OR",
        "1 - (1-v)*(1-m)",
        y_true,
        1.0 - (1.0 - v) * (1.0 - m),
    )

    add_result(
        results,
        "Mean",
        "(v+m)/2",
        y_true,
        0.5 * v + 0.5 * m,
    )

    add_result(
        results,
        "Sum_clipped",
        "clip(v+m, 0, 1)",
        y_true,
        np.clip(v + m, 0.0, 1.0),
    )

    # ---------------------------------------------------------
    # Weighted MAX
    # Voice branch stays at 1.0; down/up-weight music branch.
    # ---------------------------------------------------------
    for music_weight in np.arange(0.50, 1.21, 0.05):
        music_weight = round(float(music_weight), 2)

        add_result(
            results,
            f"Weighted_MAX_m{music_weight:.2f}",
            f"max(v, {music_weight:.2f}*m)",
            y_true,
            np.maximum(v, music_weight * m),
        )

    # ---------------------------------------------------------
    # Weighted SUM
    # alpha = voice weight, 1-alpha = music weight
    # ---------------------------------------------------------
    for alpha in np.arange(0.0, 1.001, 0.05):
        alpha = round(float(alpha), 2)
        beta = round(1.0 - alpha, 2)

        add_result(
            results,
            f"Weighted_SUM_v{alpha:.2f}_m{beta:.2f}",
            f"{alpha:.2f}*v + {beta:.2f}*m",
            y_true,
            alpha * v + beta * m,
        )

    # ---------------------------------------------------------
    # Weighted Noisy-OR
    # Scale music evidence before OR.
    # ---------------------------------------------------------
    for music_weight in np.arange(0.50, 1.21, 0.05):
        music_weight = round(float(music_weight), 2)

        mw = np.clip(music_weight * m, 0.0, 1.0)
        score = 1.0 - (1.0 - v) * (1.0 - mw)

        add_result(
            results,
            f"Weighted_NoisyOR_m{music_weight:.2f}",
            f"1-(1-v)*(1-clip({music_weight:.2f}*m,0,1))",
            y_true,
            score,
        )

    result_df = pd.DataFrame(results).sort_values(
        by=["eer", "auc"],
        ascending=[True, False],
    ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output, index=False)

    print(f"Samples: {len(y_true)}")
    print("\n===== Top 15 Fusion Results (sorted by EER) =====")
    print(
        result_df[
            ["name", "accuracy", "auc", "eer", "eer_threshold"]
        ].head(15).to_string(index=False)
    )

    baseline = result_df[result_df["name"] == "MAX_baseline"].iloc[0]
    best = result_df.iloc[0]

    print("\n===== Baseline MAX =====")
    print(f"EER : {baseline['eer']:.4f}")
    print(f"AUC : {baseline['auc']:.4f}")

    print("\n===== Best Candidate =====")
    print(f"Name      : {best['name']}")
    print(f"Formula   : {best['formula']}")
    print(f"Accuracy  : {best['accuracy']:.4f}")
    print(f"ROC-AUC   : {best['auc']:.4f}")
    print(f"EER       : {best['eer']:.4f}")
    print(f"EER Thres : {best['eer_threshold']:.4f}")
    print(f"EER gain  : {baseline['eer'] - best['eer']:+.4f}")

    print(f"\nSaved full comparison to: {args.output}")
    print(
        "\nNOTE: Fusion parameters were compared on the same 1200 samples, "
        "so use this primarily for selecting promising candidates. "
        "A separate holdout set is better for a final unbiased estimate."
    )


if __name__ == "__main__":
    main()
