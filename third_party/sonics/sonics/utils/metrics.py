import numpy as np
import pandas as pd
from sklearn import metrics

np.seterr(divide="ignore", invalid="ignore")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class F1Meter:
    def __init__(self, average="binary"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.f1_score(self.y_true, self.y_pred, average=self.average)

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


class SensitivityMeter:
    def __init__(self, average="binary"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.recall_score(
            self.y_true, self.y_pred, pos_label=1, average=self.average
        )

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


class SpecificityMeter:
    def __init__(self, average="binary"):
        self.average = average
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.recall_score(
            self.y_true, self.y_pred, pos_label=0, average=self.average
        )

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


class AccuracyMeter:
    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        self.y_true = np.concatenate([self.y_true, y_true])
        self.y_pred = np.concatenate([self.y_pred, y_pred])
        self.avg = metrics.balanced_accuracy_score(self.y_true, self.y_pred)

    def reset(self):
        self.y_true = np.array([])
        self.y_pred = np.array([])


def get_part_result(test_pred_df):
    # Create `singer` column to store whether the singer is seen or unseen
    test_pred_df["singer"] = test_pred_df.artist_overlap.map(
        lambda x: "seen" if x else "unseen"
    )

    # Create `fake_type` column to store different types of fake songs
    test_pred_df["fake_type"] = test_pred_df.label

    # Create `length` column to store different duration type songs
    test_pred_df["length"] = test_pred_df["duration_part"] = test_pred_df[
        "duration"
    ].map(lambda t: "short" if t <= 60 else ("long" if t > 120 else "medium"))

    # Initialize an empty DataFrame to store results
    part_result_df = pd.DataFrame()

    # Loop through the specified categories
    for cat in ["algorithm", "singer", "fake_type", "length"]:
        # Filter the dataframe based on the condition for each category
        if cat in ["algorithm", "fake_type"]:
            cat_df = test_pred_df.query("target == 1")
        elif cat == "singer":
            cat_df = test_pred_df.query("target == 0")
        else:
            cat_df = test_pred_df.copy()

        # Compute metrics for each partition
        for part in cat_df[cat].unique():
            part_df = cat_df[cat_df[cat] == part]
            y_true = part_df.y_true.values.astype(int)
            y_pred = (part_df.y_pred.values > 0.5).astype(int)

            # Compute TPR for `algorithm`, `fake_type`; TNR for `singer` and F1 for `length`
            score = (
                metrics.recall_score(
                    y_true, y_pred, pos_label=1 if cat != "singer" else 0
                )
                if cat != "length"
                else metrics.f1_score(y_true, y_pred, average="macro")
            )

            # Create a DataFrame for the current result
            result_df = pd.DataFrame(
                {
                    "category": [cat],
                    "partition": [part],
                    "score": [score],
                    "size": [len(part_df)],
                }
            )

            # Concatenate the result with the existing DataFrame
            part_result_df = pd.concat([part_result_df, result_df], ignore_index=True)

    # Create a dictionary with the results
    result_dict = {
        f"{row['category']}/{row['partition']}": row["score"]
        for _, row in part_result_df.iterrows()
    }

    return part_result_df, result_dict
