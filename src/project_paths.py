from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / 'data'
CLASSIFIER_PATH = ROOT / 'state/diffusion/checkpoints/emnist_mixup_classifier.pth'

BASE_VAE_PATH = ROOT / 'state/VAE/full_VAE.pth'
VAE_OUT_DIR = ROOT / 'output/VAE'
VAE_INTERP_OUT = VAE_OUT_DIR / 'vae_interp_samples.pt'

DIFF_CKPT_PATH = ROOT / 'state/diffusion/checkpoints/ddpm_class_emnist_digits.pt'
DIFF_OUT_DIR = ROOT / 'output/diffusion/v3_score'

# Paths for training the GAN used in determining Value
VALUE_GAN_GEN_PATH = ROOT / 'state/GAN/checkpoints/generators/generator_last.pth'
VALUE_GAN_DISC_PATH = ROOT / 'state/GAN/checkpoints/discriminators/discriminator_last.pth'
VALUE_GAN_OUTPUT_DIR = ROOT / 'output/GAN/checkpoints'

VALUE_CNN_BEST_PATH = ROOT / 'state/mnist_models/is_digit_binary_classifier/cnn/best_cnn.pth'
VALUE_CNN_LAST_PATH = VALUE_CNN_BEST_PATH.parent / 'last_cnn.pth'

NOVELTY_CNN_MIXUP = ROOT / 'state/digit_classifier/emnist_mixup_classifier.pth'
NOVELTY_CNN_PLAIN = ROOT / 'state/digit_classifier/emnist_classifier.pth'
