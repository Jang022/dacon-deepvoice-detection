#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate 1200 MixFake samples")
    p.add_argument("--submission", type=Path, default=Path("output") / "submission.csv")
    p.add_argument("--metadata", type=Path, default=Path("data") / "test_1200_metadata.csv")
    return p.parse_args()

def stem_id(value):
    if pd.isna(value):
        return ""
    s = str(value).strip().replace("\\", "/")
    return Path(s.rsplit("/", 1)[-1]).stem

def authenticity_to_label(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    real_values = {"real","bonafide","bona_fide","bona-fide","authentic","genuine","human","r","0"}
    fake_values = {"fake","spoof","synthetic","generated","deepfake","f","1"}
    if s in real_values:
        return 0
    if s in fake_values:
        return 1
    if any(k in s for k in ["bonafide","bona fide","authentic","genuine","real"]):
        return 0
    if any(k in s for k in ["spoof","fake","synthetic","generated","deepfake"]):
        return 1
    return np.nan

def eer_and_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thresholds[i])

def evaluate_binary(name, y_true, y_score):
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = np.asarray(y_true)[mask].astype(np.int64)
    y_score = np.asarray(y_score)[mask].astype(np.float64)
    print(f"\n===== {name} =====")
    print(f"Samples   : {len(y_true)}")
    if len(y_true) == 0:
        print("No valid samples")
        return None
    print(f"Real/Fake : {(y_true == 0).sum()} / {(y_true == 1).sum()}")
    if np.unique(y_true).size < 2:
        print("ROC-AUC   : N/A (only one class)")
        print("EER       : N/A (only one class)")
        return {"samples":len(y_true),"auc":np.nan,"eer":np.nan,"eer_threshold":np.nan}
    pred = (y_score >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true, pred))
    auc = float(roc_auc_score(y_true, y_score))
    eer, th = eer_and_threshold(y_true, y_score)
    print(f"Accuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"EER       : {eer:.4f}")
    print(f"EER Thres : {th:.4f}")
    return {"samples":len(y_true),"accuracy":acc,"auc":auc,"eer":eer,"eer_threshold":th}

def main():
    args = parse_args()
    pred = pd.read_csv(args.submission)
    meta = pd.read_csv(args.metadata)

    required_pred = {
        "ID","FILE_FAKE_PROB","VOICE_FAKE_PROB","MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB","MUSIC_PRESENT_PROB"
    }
    missing = required_pred - set(pred.columns)
    if missing:
        raise ValueError(f"submission missing columns: {sorted(missing)}")

    required_meta = {
        "syn_file","syn_authenticity","fore_file","fore_authenticity",
        "back_file","back_type","back_authenticity"
    }
    missing = required_meta - set(meta.columns)
    if missing:
        raise ValueError(
            f"metadata missing columns: {sorted(missing)}\nAvailable: {list(meta.columns)}"
        )

    pred["ID"] = pred["ID"].astype(str).str.strip()
    meta["ID"] = meta["syn_file"].apply(stem_id)

    df = pred.merge(meta, on="ID", how="inner", validate="one_to_one")
    print("===== Merge Check =====")
    print(f"Submission rows : {len(pred)}")
    print(f"Metadata rows   : {len(meta)}")
    print(f"Matched rows    : {len(df)}")
    if len(df) != len(pred):
        pred_only = sorted(set(pred["ID"]) - set(meta["ID"]))
        print("Unmatched submission IDs:", pred_only[:10])
        raise RuntimeError("Not all submission rows matched metadata")

    df["file_label"] = df["syn_authenticity"].apply(authenticity_to_label)
    df["voice_label"] = df["fore_authenticity"].apply(authenticity_to_label)
    df["music_label"] = df["back_authenticity"].apply(authenticity_to_label)

    file_result = evaluate_binary(
        "FILE FAKE",
        df["file_label"].to_numpy(dtype=float),
        df["FILE_FAKE_PROB"].to_numpy(dtype=float),
    )
    voice_result = evaluate_binary(
        "VOICE FAKE",
        df["voice_label"].to_numpy(dtype=float),
        df["VOICE_FAKE_PROB"].to_numpy(dtype=float),
    )

    music_mask = (
        df["back_type"].fillna("").astype(str).str.strip().str.lower().eq("music")
    )
    music_result = evaluate_binary(
        "MUSIC FAKE (back_type == music)",
        df.loc[music_mask, "music_label"].to_numpy(dtype=float),
        df.loc[music_mask, "MUSIC_FAKE_PROB"].to_numpy(dtype=float),
    )

    music_present_true = music_mask.astype(int).to_numpy()
    music_presence_result = evaluate_binary(
        "MUSIC PRESENCE",
        music_present_true.astype(float),
        df["MUSIC_PRESENT_PROB"].to_numpy(dtype=float),
    )

    fore_present = (
        df["fore_file"].fillna("").astype(str).str.strip().ne("").astype(int).to_numpy()
    )
    voice_presence_result = evaluate_binary(
        "VOICE PRESENCE",
        fore_present.astype(float),
        df["VOICE_PRESENT_PROB"].to_numpy(dtype=float),
    )

    print("\n===== Summary =====")
    eers_ok = (
        file_result is not None and voice_result is not None and music_result is not None
        and np.isfinite(file_result["eer"]) and np.isfinite(voice_result["eer"])
        and np.isfinite(music_result["eer"])
    )
    if eers_ok:
        file_eer = file_result["eer"]
        voice_eer = voice_result["eer"]
        music_eer = music_result["eer"]
        ads = (
            0.5 * (1.0 - file_eer)
            + 0.2 * (1.0 - voice_eer)
            + 0.3 * (1.0 - music_eer)
        )
        print(f"File EER  : {file_eer:.4f}")
        print(f"Voice EER : {voice_eer:.4f}")
        print(f"Music EER : {music_eer:.4f}")
        print(f"ADS       : {ads:.4f}")
    else:
        ads = np.nan
        print("ADS       : N/A")

    vp_auc = voice_presence_result["auc"] if voice_presence_result else np.nan
    mp_auc = music_presence_result["auc"] if music_presence_result else np.nan

    print(f"Voice Presence AUC : {'N/A' if not np.isfinite(vp_auc) else f'{vp_auc:.4f}'}")
    print(f"Music Presence AUC : {'N/A' if not np.isfinite(mp_auc) else f'{mp_auc:.4f}'}")

    if np.isfinite(vp_auc) and np.isfinite(mp_auc):
        cps = 0.5 * vp_auc + 0.5 * mp_auc
        print(f"CPS                : {cps:.4f}")
        if np.isfinite(ads):
            print(f"Estimated Score    : {0.9 * ads + 0.1 * cps:.4f}")
    else:
        print("Estimated Score    : N/A (both presence AUCs required)")

    out = args.submission.parent / "mixfake_1200_evaluation_merged.csv"
    df.to_csv(out, index=False)
    print(f"\nMerged evaluation saved: {out}")

if __name__ == "__main__":
    main()
