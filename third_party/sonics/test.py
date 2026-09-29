import argparse
import logging
import os
import warnings

import pandas as pd
import yaml

import torch

try:
    from torch.amp import autocast

    torch_amp_new = True
except:
    from torch.cuda.amp import autocast

    torch_amp_new = False

from sonics.models.model import AudioClassifier
from sonics.utils.config import dict2cfg
from sonics.utils.dataset import get_dataloader
from sonics.utils.metrics import get_part_result
from sonics.utils.losses import BCEWithLogitsLoss, SigmoidFocalLoss
from sonics.utils.seed import set_seed, worker_init_fn

# Import the valid_loop function from the training script
from train import valid_loop

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("fvcore").setLevel(logging.ERROR)


def arg_parser():
    parser = argparse.ArgumentParser(description="Test a model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to checkpoint file"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = arg_parser()
    dict_ = yaml.safe_load(open(args.config).read())
    cfg = dict2cfg(dict_)

    # Create output directory
    os.makedirs(f"output/{cfg.experiment_name}", exist_ok=True)

    # Setup logger
    print(cfg)

    # Set seed
    set_seed(cfg.environment.seed)

    # Set up device
    if not torch.cuda.is_available():
        print("> Using CPU, this will be slow")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"> Using GPU: {device}")

    # Load test data
    test_df = pd.read_csv(cfg.dataset.test_dataframe)

    # Shuffle test data
    test_df = test_df.sample(frac=1.0, random_state=cfg.environment.seed).reset_index(
        drop=True
    )

    # Store data stats
    cfg.dataset.num_test = len(test_df)
    cfg.dataset.num_test_real = len(test_df.query("target == 0"))
    cfg.dataset.num_test_fake = len(test_df.query("target == 1"))

    # Load dataloader
    test_dataloader = get_dataloader(
        test_df.filepath.tolist(),
        test_df.target.tolist(),
        skip_times=test_df.skip_time.tolist() if cfg.audio.skip_time else None,
        max_len=cfg.audio.max_len,
        batch_size=cfg.validation.batch_size,
        num_classes=cfg.num_classes,
        train=False,
        random_sampling=False,
        num_workers=cfg.environment.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=None,
        distributed=False,
    )

    # Load model
    model = AudioClassifier(cfg)
    model.to(device)

    # Load checkpoint
    if not os.path.exists(args.ckpt_path):
        print(f"> Checkpoint file not found: {args.ckpt_path}")
        raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(f"> Loaded checkpoint from {args.ckpt_path}")

    # Loss
    if cfg.loss.name == "BCEWithLogitsLoss":
        criterion = BCEWithLogitsLoss(label_smoothing=cfg.loss.label_smoothing)
    elif cfg.loss.name == "SigmoidFocalLoss":
        criterion = SigmoidFocalLoss(
            alpha=cfg.loss.alpha,
            gamma=cfg.loss.gamma,
            label_smoothing=cfg.loss.label_smoothing,
        )
    else:
        raise ValueError(f"Unknown loss function: {cfg.loss.name}")

    # Test loop
    (
        test_loss,
        test_acc,
        test_f1,
        test_sens,
        test_spec,
        test_pred_df,
    ) = valid_loop(model, test_dataloader, criterion, device, cfg, desc="Test")

    # Store test results
    best_test_result = {
        "loss": test_loss,
        "acc": test_acc,
        "f1": test_f1,
        "sens": test_sens,
        "spec": test_spec,
    }

    print("> Best Test Result:")
    best_test_result_df = pd.DataFrame([best_test_result])
    print(best_test_result_df.to_markdown(index=False, tablefmt="grid"))
    print()

    test_df = test_df[
        : len(test_pred_df)
    ]  # in case test_df is longer than test_pred_df
    test_pred_df = pd.concat([test_df, test_pred_df], axis=1)

    # Get partition results
    part_result_df, part_result_dict = get_part_result(test_pred_df)
    print("> Test Partition Results:")
    print(part_result_df.to_markdown(index=False))
    print()

    # Save test prediction
    print(
        f"> Saving test predictions to output/{cfg.experiment_name}/test_predictions.csv"
    )
    test_pred_df.to_csv(
        f"output/{cfg.experiment_name}/test_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
