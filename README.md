# DeepVoice Detection Pipeline

DACON DeepVoice detection competition project.

## Pipeline

Input Audio
→ PANNs
  - Voice Presence
  - Music Presence

→ HTDemucs
  - Voice Stem → DF-Arena 1B
  - Music Stem → Fine-tuned SpecTTTra

→ File-level Fusion
→ 5 probabilities

## Music Model Training

SpecTTTra was fine-tuned on MixFake music samples.

- Train: 30,000
  - Real: 15,000
  - Fake: 15,000
- Validation: 2,000
  - Real: 1,000
  - Fake: 1,000

## Evaluation

MixFake holdout, 1,200 mixed audio samples:

- Voice EER: 0.1458
- Music EER: 0.2554
- File EER: 0.2519
- Music Presence ROC-AUC: 0.9551

The 30k SpecTTTra model improved Music EER
from 0.3184 to 0.2554 compared with the 10k training setup.

## Environment

- Python 3.10
- PyTorch 2.7.1
- CUDA 12.8
- PANNs
- HTDemucs
- DF-Arena 1B
- SONICS SpecTTTra
