from pathlib import Path
import torch
from sonics import HFAudioClassifier

ROOT = Path(r"C:\Temp\dacon_Deepvoice")

BASE_MODEL = ROOT / "model" / "spectttra_beta_5s"
EXP_DIR = ROOT / "output" / "mixfake_spectttra_30k_e8"

exports = {
    "best_checkpoint.pth":
        ROOT / "model" / "spectttra_mixfake_30k_best",

    "last_checkpoint.pth":
        ROOT / "model" / "spectttra_mixfake_30k_last",
}

for checkpoint_name, output_dir in exports.items():

    model = HFAudioClassifier.from_pretrained(
        str(BASE_MODEL),
        map_location="cpu",
    )

    checkpoint = torch.load(
        EXP_DIR / checkpoint_name,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    model.save_pretrained(
        str(output_dir)
    )

    print(checkpoint_name, "->", output_dir)