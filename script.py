#!/usr/bin/env python3
"""경진대회 테스트 데이터에 대한 5개 확률값을 생성한다."""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# 제출 실행에서는 로컬 model 폴더만 사용한다.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

import librosa
import numpy as np
import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model
from demucs.separate import load_track
from tqdm import tqdm
from sonics import HFAudioClassifier

# -----------------------------------------------------------------------------
# 경로 설정
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DF_ARENA_DIR = MODEL_DIR / "df_arena_1b"
HTDEMUCS_DIR = MODEL_DIR / "htdemucs"
PANNS_DIR = MODEL_DIR / "panns"
SPECTTTRA_DIR = MODEL_DIR / "spectttra_mixfake_30k_last"

DEFAULT_TEST_DIR = BASE_DIR / "data" / "test"
DEFAULT_SAMPLE_SUBMISSION = BASE_DIR / "data" / "sample_submission.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"


def validate_local_assets():
    required_dirs = [
        DF_ARENA_DIR,
        HTDEMUCS_DIR,
        PANNS_DIR,
        SPECTTTRA_DIR,
    ]
    for path in required_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Required model directory not found: {path}")

    required_files = [
        PANNS_DIR / "Cnn14_mAP=0.431.pth",
        PANNS_DIR / "class_labels_indices.csv",
        PANNS_DIR / "component_labels.json",
        SPECTTTRA_DIR / "config.json",
        SPECTTTRA_DIR / "pytorch_model.bin",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")

# 오디오 처리 설정
AUDIO_SAMPLE_RATE = 16_000
PANNS_SAMPLE_RATE = 32_000
SEGMENT_SAMPLES = 64_600
SILENCE_RMS = 1e-5

PREDICTION_COLUMNS = [
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
]

SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


# -----------------------------------------------------------------------------
# 1. 입력 파일 및 제출 양식 확인
# -----------------------------------------------------------------------------

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate DACON submission probabilities."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE_SUBMISSION
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args(argv)

def select_device(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return torch.device(device_name)


def find_audio_files(test_dir):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    audio_files = []
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            audio_files.append(path)
    audio_files.sort(key=lambda path: path.stem)

    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")

    audio_ids = [path.stem for path in audio_files]
    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError("Audio IDs must be unique")
    return audio_files


def read_sample_submission(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sample submission not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames
        rows = list(reader)

    if column_names is None or not rows:
        raise ValueError(f"Invalid sample submission: {csv_path}")

    required_columns = ["ID"] + PREDICTION_COLUMNS
    missing_columns = [name for name in required_columns if name not in column_names]
    if missing_columns:
        raise ValueError(f"Sample submission is missing columns: {missing_columns}")

    seen_ids = set()
    for row in rows:
        audio_id = str(row["ID"]).strip()
        if not audio_id:
            raise ValueError("Sample submission contains an empty ID")
        if audio_id in seen_ids:
            raise ValueError(f"Duplicate ID in sample submission: {audio_id}")
        seen_ids.add(audio_id)
        row["ID"] = audio_id

    return column_names, rows


def order_audio_files(audio_files, submission_rows):
    audio_by_id = {path.stem: path for path in audio_files}
    submission_ids = [row["ID"] for row in submission_rows]

    missing_ids = [audio_id for audio_id in submission_ids if audio_id not in audio_by_id]
    extra_ids = [audio_id for audio_id in audio_by_id if audio_id not in submission_ids]
    if missing_ids or extra_ids:
        raise ValueError(
            "Test audio and sample submission IDs do not match. "
            f"Missing: {missing_ids[:5]}, Extra: {extra_ids[:5]}"
        )

    return [audio_by_id[audio_id] for audio_id in submission_ids]



def load_audio(audio_path):
    audio, _ = librosa.load(
        audio_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32
    )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {audio_path}")
    return audio


# -----------------------------------------------------------------------------
# 2. 오디오 구간 분할
# -----------------------------------------------------------------------------

def get_segment_starts(audio_length):
    if audio_length <= SEGMENT_SAMPLES:
        return [0]

    last_start = audio_length - SEGMENT_SAMPLES
    starts = list(range(0, last_start + 1, SEGMENT_SAMPLES))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def extract_segment(audio, start):
    if audio.size < SEGMENT_SAMPLES:
        repeat_count = SEGMENT_SAMPLES // audio.size + 1
        audio = np.tile(audio, repeat_count)
        return audio[:SEGMENT_SAMPLES].astype(np.float32)

    end = start + SEGMENT_SAMPLES
    return audio[start:end].astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# 3. PANNs를 이용한 음성·음악 존재 여부 추론
# -----------------------------------------------------------------------------

def prepare_panns_labels():
    source = PANNS_DIR / "class_labels_indices.csv"
    target = Path.home() / "panns_data" / "class_labels_indices.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def load_panns_model(device):
    prepare_panns_labels()
    from panns_inference import AudioTagging, labels

    model = AudioTagging(
        checkpoint_path=str(PANNS_DIR / "Cnn14_mAP=0.431.pth"),
        device=device.type,
    )

    config_path = PANNS_DIR / "component_labels.json"
    label_groups = json.loads(config_path.read_text(encoding="utf-8"))
    label_to_index = {label: index for index, label in enumerate(labels)}
    voice_indices = [label_to_index[label] for label in label_groups["voice"]]
    music_indices = [label_to_index[label] for label in label_groups["music"]]
    return model, voice_indices, music_indices


def make_panns_segments(audio):
    segments = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment = librosa.resample(
            segment,
            orig_sr=AUDIO_SAMPLE_RATE,
            target_sr=PANNS_SAMPLE_RATE,
            res_type="soxr_hq",
        )
        segments.append(segment.astype(np.float32))
    return np.stack(segments)


def predict_presence(model, voice_indices, music_indices, audio):
    segments = make_panns_segments(audio)
    predictions, _ = model.inference(segments)
    voice_probability = float(predictions[:, voice_indices].max())
    music_probability = float(predictions[:, music_indices].max())
    return voice_probability, music_probability


def predict_presence_for_all_files(audio_files, device):
    model, voice_indices, music_indices = load_panns_model(device)
    presence_scores = {}

    for audio_path in tqdm(audio_files, desc="Presence"):
        audio = load_audio(audio_path)
        presence_scores[audio_path.stem] = predict_presence(
            model, voice_indices, music_indices, audio
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return presence_scores


# -----------------------------------------------------------------------------
# 4. HTDemucs를 이용한 음성·음악 분리
# -----------------------------------------------------------------------------

def load_htdemucs_model():
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        # PyTorch 2.6부터 바뀐 기본값에 맞춰 기존 체크포인트를 불러온다.
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = get_model("htdemucs", repo=HTDEMUCS_DIR)
    finally:
        torch.load = original_torch_load
    return model.cpu().eval()


def separate_voice_and_music(audio_path, model, device):
    waveform = load_track(
        audio_path, model.audio_channels, model.samplerate
    ).float()
    mono_waveform = waveform.mean(0)
    mean = mono_waveform.mean()
    std = mono_waveform.std()

    if float(std) < 1e-8:
        length = round(waveform.shape[-1] * AUDIO_SAMPLE_RATE / model.samplerate)
        silence = np.zeros(max(1, length), dtype=np.float32)
        return silence, silence.copy()

    normalized_waveform = (waveform - mean) / std
    with torch.inference_mode():
        sources = apply_model(
            model,
            normalized_waveform[None],
            device=device,
            shifts=0,
            split=True,
            overlap=0.25,
            progress=False,
        )[0]
    sources = sources * std + mean

    vocal_index = model.sources.index("vocals")
    voice_audio = sources[vocal_index].mean(0, keepdim=True)

    music_sources = []
    for index, source_name in enumerate(model.sources):
        if source_name != "vocals":
            music_sources.append(sources[index])
    music_audio = torch.stack(music_sources).sum(0).mean(0, keepdim=True)

    voice_audio = torchaudio.functional.resample(
        voice_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    music_audio = torchaudio.functional.resample(
        music_audio, model.samplerate, AUDIO_SAMPLE_RATE
    )[0]
    return (
        voice_audio.cpu().numpy().astype(np.float32),
        music_audio.cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------------------
# 5. DF-Arena 1B를 이용한 성분별 Fake 추론
# -----------------------------------------------------------------------------

def load_df_arena_model(device):
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))
    from df_arena_1b.modeling_antispoofing import DF_Arena_1B_Antispoofing

    previous_directory = Path.cwd()
    os.chdir(DF_ARENA_DIR)
    try:
        model = DF_Arena_1B_Antispoofing.from_pretrained(
            str(DF_ARENA_DIR),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    finally:
        os.chdir(previous_directory)

    model = model.to(device).eval()
    fake_label_index = int(model.config.label2id["spoof"])
    return model, fake_label_index

#spectttra load
def load_spectttra_model(device):
    model_dir = SPECTTTRA_DIR.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"SpecTTTra model directory not found: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Missing SpecTTTra config: {model_dir / 'config.json'}")
    if not (model_dir / "pytorch_model.bin").is_file():
        raise FileNotFoundError(
            f"Missing SpecTTTra weights: {model_dir / 'pytorch_model.bin'}"
        )

    model = HFAudioClassifier.from_pretrained(str(model_dir))
    return model.to(device).eval()

def prepare_spectttra_audio(audio, max_len):
    audio = np.asarray(
        audio,
        dtype=np.float32,
    ).reshape(-1)
    
    if audio.size < max_len:
        audio = np.pad(
            audio,
            (0, max_len - audio.size)
        )
        
    elif audio.size > max_len:
        start = int(
            (audio.size - max_len) *0.75
        )
        audio = audio[start:start+max_len]
        
    std = max(float(np.std(audio)), 1e-6)
    audio = audio / std
    
    return audio.astype(np.float32)


def calculate_rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def predict_fake(model, fake_label_index, audio, device):
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    segment_scores = []
    for start in get_segment_starts(audio.size):
        segment = extract_segment(audio, start)
        segment_tensor = torch.from_numpy(segment).to(device)

        with torch.inference_mode():
            logits = model(input_values=segment_tensor)["logits"]
            probabilities = torch.softmax(logits.float(), dim=-1)
        segment_scores.append(float(probabilities[0, fake_label_index]))

    return max(segment_scores)
#music predict
def predict_music_fake_spectttra(model, audio, device):
    if calculate_rms(audio) < SILENCE_RMS:
        return 0.0

    max_len = int(model.config.audio.max_len)  # 80000 = 5 sec

    # 5초보다 짧으면 기존처럼 padding
    if audio.size <= max_len:
        segment = prepare_spectttra_audio(audio, max_len)

        tensor = (
            torch.from_numpy(segment)
            .float()
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():
            logits = model(tensor)
            prob = torch.sigmoid(logits.float()).reshape(-1)[0]
        return float(prob)

    # 긴 파일이면 여러 5초 구간 확인
    starts = list(range(0, audio.size - max_len + 1, max_len))

    # 마지막 부분도 반드시 포함
    last_start = audio.size - max_len
    if starts[-1] != last_start:
        starts.append(last_start)

    scores = []

    for start in starts:
        segment = audio[start:start + max_len]

        std = max(float(np.std(segment)), 1e-6)
        segment = (segment / std).astype(np.float32)

        tensor = (
            torch.from_numpy(segment)
            .float()
            .unsqueeze(0)
            .to(device)
        )

        with torch.inference_mode():
            logits = model(tensor)
            prob = torch.sigmoid(logits.float()).reshape(-1)[0]

        scores.append(float(prob))

    return max(scores)
# -----------------------------------------------------------------------------
# 6. 파일 단위 점수 계산 및 제출 파일 저장
# -----------------------------------------------------------------------------

def combine_file_fake_score(voice_fake, music_fake, voice_present, music_present):
    voice_score = voice_present * voice_fake
    music_score = music_present * music_fake
    return max(voice_score, music_score)


def predict_fake_scores_for_all_files(
    audio_files, submission_rows, presence_scores, device
):
    df_arena_model, fake_label_index = load_df_arena_model(device)
    spectttra_model = load_spectttra_model(device)
    htdemucs_model = load_htdemucs_model()

    for index, audio_path in enumerate(tqdm(audio_files, desc="Components")):
    
        voice_audio, music_audio = separate_voice_and_music(
            audio_path, htdemucs_model, device
        )
        voice_fake = predict_fake(
            df_arena_model, fake_label_index, voice_audio, device
        )
        music_fake = predict_music_fake_spectttra(
            spectttra_model, music_audio, device
        )

        voice_present, music_present = presence_scores[audio_path.stem]
        file_fake = combine_file_fake_score(
            voice_fake, music_fake, voice_present, music_present
        )

        row = submission_rows[index]
        row["FILE_FAKE_PROB"] = round(file_fake, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake, 10)
        row["VOICE_PRESENT_PROB"] = round(voice_present, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present, 10)

    return submission_rows



def save_submission(output_path, column_names, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_arguments()
    device = select_device(args.device)
    validate_local_assets()

    print(f"Device: {device}")
    print(f"Test dir: {args.test_dir}")
    print(f"Sample submission: {args.sample_submission}")
    print(f"SpecTTTra: {SPECTTTRA_DIR}")

    # 1. DACON test 오디오와 sample_submission의 ID 순서를 맞춘다.
    audio_files = find_audio_files(args.test_dir)
    column_names, submission_rows = read_sample_submission(
        args.sample_submission
    )
    audio_files = order_audio_files(audio_files, submission_rows)
    print(f"Test samples: {len(audio_files)}")

    # 2. PANNs: 음성/음악 존재 확률
    print("\n===== Stage 1: PANNs =====")
    presence_scores = predict_presence_for_all_files(audio_files, device)

    # 3. HTDemucs 분리 + Voice DF-Arena + Music SpecTTTra
    print("\n===== Stage 2: HTDemucs + DF-Arena + SpecTTTra =====")
    submission_rows = predict_fake_scores_for_all_files(
        audio_files, submission_rows, presence_scores, device
    )

    # 4. MAX Fusion 결과를 포함한 5개 확률 컬럼 저장
    save_submission(args.output, column_names, submission_rows)
    print(f"Saved {len(submission_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
