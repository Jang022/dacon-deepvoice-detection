#!/usr/bin/env python3
import argparse
import csv
import gc
from pathlib import Path

import random
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from Baseline_Dacon_Staged import (
    AUDIO_SAMPLE_RATE,
    load_htdemucs_model,
    separate_voice_and_music,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = BASE_DIR / "music_background_labels.csv"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "data" / "mixfake_music_stems"
DEFAULT_TRAIN_CSV = BASE_DIR / "data" / "mixfake_music_train.csv"
DEFAULT_DEV_CSV = BASE_DIR / "data" / "mixfake_music_dev.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--dev-csv", type=Path, default=DEFAULT_DEV_CSV)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def read_rows(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    required = {"sample_id", "path", "label", "split"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    clean_rows = []

    for row in rows:
        split = str(row["split"]).strip().lower()

        if split not in {"train", "dev"}:
            continue

        sample_id = str(row["sample_id"]).strip()
        audio_path = Path(row["path"])
        label = int(row["label"])

        if not sample_id:
            raise ValueError("Empty sample_id found")

        if label not in (0, 1):
            raise ValueError(f"Invalid label for {sample_id}: {label}")

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        clean_rows.append({
            "sample_id": sample_id,
            "path": audio_path,
            "label": label,
            "split": split,
        })

    return clean_rows


def save_pcm16_wav(audio, output_path):
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if audio.size == 0:
        raise ValueError(f"Empty separated audio: {output_path}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite separated audio: {output_path}")

    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak

    waveform = torch.from_numpy(audio).unsqueeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(
        str(output_path),
        waveform,
        AUDIO_SAMPLE_RATE,
        encoding="PCM_S",
        bits_per_sample=16,
    )


def write_sonics_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "target"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = read_rows(args.input_csv)
    
    random.seed(42)
    train_real = [
        row for row in rows
        if row["split"] == "train"
        and row["label"] == 0
    ]
    
    train_fake = [
        row for row in rows
        if row["split"] == "train"
        and row["label"] == 1
    ]
    
    dev_real = [
        row for row in rows
        if row["split"] == "dev"
        and row["label"] == 0
    ]
    dev_fake = [
        row for row in rows
        if row["split"] == "dev"
        and row["label"] == 1
    ]
    
    train_real = random.sample(train_real, 15000)
    train_fake = random.sample(train_fake, 15000)
    
    dev_real = random.sample(dev_real, 1000)
    dev_fake = random.sample(dev_fake, 1000)
    
    rows = (train_real+train_fake+dev_real+dev_fake)
    random.shuffle(rows)
    
    
    train_count = sum(row["split"] == "train" for row in rows)
    dev_count = sum(row["split"] == "dev" for row in rows)

    print("===== MixFake music preprocessing =====")
    print(f"Device : {device}")
    print(f"Train  : {train_count}")
    print(f"Dev    : {dev_count}")
    print(f"Total  : {len(rows)}")

    if train_count == 0 or dev_count == 0:
        raise ValueError("Both train and dev samples are required")

    args.output_root.mkdir(parents=True, exist_ok=True)

    print("\nLoading HTDemucs...")
    htdemucs_model = load_htdemucs_model()

    sonics_rows = {"train": [], "dev": []}
    created = 0
    skipped = 0

    for row in tqdm(rows, total=len(rows), desc="HTDemucs music stems"):
        sample_id = row["sample_id"]
        split = row["split"]

        output_path = args.output_root / split / f"{sample_id}.wav"

        if output_path.is_file():
            skipped += 1
        else:
            _, music_audio = separate_voice_and_music(
                row["path"],
                htdemucs_model,
                device,
            )

            save_pcm16_wav(music_audio, output_path)
            created += 1

        sonics_rows[split].append({
            "filepath": str(output_path.resolve()),
            "target": row["label"],
        })

    del htdemucs_model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    write_sonics_csv(sonics_rows["train"], args.train_csv)
    write_sonics_csv(sonics_rows["dev"], args.dev_csv)

    print("\n===== Done =====")
    print(f"Created stems : {created}")
    print(f"Skipped stems : {skipped}")
    print(f"Train CSV     : {args.train_csv} ({len(sonics_rows['train'])})")
    print(f"Dev CSV       : {args.dev_csv} ({len(sonics_rows['dev'])})")
    print(f"Stem root     : {args.output_root}")


if __name__ == "__main__":
    main()
