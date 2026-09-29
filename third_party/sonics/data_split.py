import pandas as pd



if __name__ == "__main__":
    DATA_DIR = "./dataset"

    real_df = pd.read_csv(f"{DATA_DIR}/real_songs.csv")
    real_df["filepath"] = f"{DATA_DIR}/real_songs/" + real_df.filename + ".mp3"
    real_df["target"] = 0

    fake_df = pd.read_csv(f"{DATA_DIR}/fake_songs.csv")
    fake_df["filepath"] = f"{DATA_DIR}/fake_songs/" + fake_df.filename + ".mp3"
    fake_df["target"] = 1

    df = pd.concat([real_df, fake_df])
    df = df[(df.duration >= 30) & (df.no_vocal == False)]

    train_df = df[df.split == 'train']
    train_df.to_csv("train.csv",index=False)

    valid_df = df[df.split == 'valid']
    valid_df.to_csv("valid.csv",index=False)

    test_df = df[df.split == 'test']
    test_df.to_csv("test.csv",index=False)