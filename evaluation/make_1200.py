from pathlib import Path
import shutil
import pandas as pd


# =========================================================
# 경로 설정
# =========================================================

PROJECT_ROOT = Path(r"C:\Temp\dacon_Deepvoice")

# 실제 MixFake 압축 해제 위치에 맞게 수정
MIXFAKE_ROOT = Path(r"D:\MixFake\MixFake\MixFake")

DETAILS_CSV = MIXFAKE_ROOT / "protocols" / "mixed_details.csv"
MIXED_DIR = MIXFAKE_ROOT / "mixed_dataset"

TEST_DIR = PROJECT_ROOT / "data" / "test"
SAMPLE_SUBMISSION = PROJECT_ROOT / "data" / "sample_submission.csv"
SELECTED_CSV = PROJECT_ROOT / "data" / "test_1200_metadata.csv"


# =========================================================
# 확인
# =========================================================

if not DETAILS_CSV.is_file():
    raise FileNotFoundError(
        f"mixed_details.csv 없음: {DETAILS_CSV}"
    )

if not MIXED_DIR.is_dir():
    raise FileNotFoundError(
        f"mixed_dataset 없음: {MIXED_DIR}"
    )


# =========================================================
# mixed_details.csv
# =========================================================

df = pd.read_csv(DETAILS_CSV)

print("전체:", len(df))
print(df["split"].value_counts())


# eval만 사용
eval_df = df[
    df["split"].astype(str).str.lower() == "eval"
].copy()

print("eval:", len(eval_df))

if len(eval_df) < 1200:
    raise ValueError(
        f"eval 데이터가 1200개보다 적음: {len(eval_df)}"
    )


# 고정 seed로 랜덤 1200개
sample = eval_df.sample(
    n=1200,
    random_state=42
).reset_index(drop=True)


# =========================================================
# 기존 test 폴더 정리
# =========================================================

if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# mixed_dataset 파일 인덱스
# =========================================================

print("mixed_dataset 파일 검색 중...")

audio_index = {}

for path in MIXED_DIR.rglob("*"):
    if path.is_file():
        audio_index[path.name] = path

print("검색된 audio:", len(audio_index))


# =========================================================
# 1200개 복사
# =========================================================

submission_rows = []
missing = []

for _, row in sample.iterrows():

    # syn_file에 경로가 들어있더라도 파일명만 사용
    filename = Path(str(row["syn_file"])).name

    src = audio_index.get(filename)

    if src is None:
        missing.append(filename)
        continue

    dst = TEST_DIR / filename

    shutil.copy2(
        src,
        dst
    )

    submission_rows.append({
        "ID": dst.stem,
        "FILE_FAKE_PROB": 0.0,
        "VOICE_FAKE_PROB": 0.0,
        "MUSIC_FAKE_PROB": 0.0,
        "VOICE_PRESENT_PROB": 0.0,
        "MUSIC_PRESENT_PROB": 0.0,
    })


# =========================================================
# 결과 확인
# =========================================================

if missing:
    print("\n못 찾은 파일 예시:")
    for name in missing[:10]:
        print(name)

    raise FileNotFoundError(
        f"{len(missing)}개 파일을 mixed_dataset에서 못 찾음"
    )

if len(submission_rows) != 1200:
    raise RuntimeError(
        f"복사 결과가 1200개가 아님: {len(submission_rows)}"
    )


# ID 중복 확인
ids = [row["ID"] for row in submission_rows]

if len(ids) != len(set(ids)):
    raise ValueError(
        "동일한 ID가 존재함"
    )


# =========================================================
# sample_submission 생성
# =========================================================

pd.DataFrame(
    submission_rows
).to_csv(
    SAMPLE_SUBMISSION,
    index=False
)


# 어떤 샘플을 뽑았는지도 보존
sample.to_csv(
    SELECTED_CSV,
    index=False
)


print()
print("완료")
print("Test audio :", len(submission_rows))
print("Test dir   :", TEST_DIR)
print("Submission :", SAMPLE_SUBMISSION)
print("Metadata   :", SELECTED_CSV)