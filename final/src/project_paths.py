from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = ROOT / "state/diffusion/checkpoints/mnist_mixup_classifier.pth"
CKPT_PATH = ROOT / 'state/diffusion/checkpoints/ddpm_class'
OUT_DIR = ROOT / 'output/diffusion/v3_score'