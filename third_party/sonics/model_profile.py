import argparse
import os
import yaml
import torch
import pandas as pd
from sonics.models.model import AudioClassifier
from sonics.utils.perf import profile_model
from sonics.utils.seed import set_seed


def arg_parser():
    parser = argparse.ArgumentParser(description="Profile a model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--batch_size", type=int, default=12, help="Batch size for profiling"
    )
    return parser.parse_args()


def main():
    # Parse arguments
    args = arg_parser()
    dict_ = yaml.safe_load(open(args.config).read())
    cfg = dict2cfg(dict_)
    print(cfg)

    # Set seed
    set_seed(cfg.environment.seed)

    print("\n> Experiment Name:", cfg.experiment_name)

    # Set up device
    if not torch.cuda.is_available():
        print("> Using CPU, this will be slow")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        print(f"> Using GPU: {device}")

    # Load model
    print("> Loading model...")
    model = AudioClassifier(cfg)
    model.to(device)

    # Profile model
    print("> Model Profile:")
    input_tensor = torch.randn((args.batch_size, cfg.audio.max_len)).to(device)
    profile_df = profile_model(model, input_tensor, display=True)

    # Save profile results
    os.makedirs(f"output/{cfg.experiment_name}", exist_ok=True)
    profile_df.to_csv(f"output/{cfg.experiment_name}/model_profile.csv", index=False)
    print(f"> Profile results saved to output/{cfg.experiment_name}/model_profile.csv")


if __name__ == "__main__":
    main()
