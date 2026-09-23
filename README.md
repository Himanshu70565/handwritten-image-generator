Deep Image Generation with DCGAN — Assignment

This repository contains a PyTorch Deep Convolutional GAN (DCGAN) for generating handwritten character images from the EMNIST dataset.

Contents
- `README.md` — Project overview, setup instructions, and run commands
- `.gitignore` — Files and folders excluded from version control
- `dataset/EMNIST/` — EMNIST images and label information
- `dataset/Train/` — Character images selected for training
- `src/dcgan.py` — Generator and discriminator model definitions
- `src/utils.py` — Data loading and image transformations
- `src/train.py` — DCGAN training and checkpoint saving
- `src/generate_images.py` — Loads the trained generator and creates 20 images
- `src/TRAINING_models/` — Saved generator and discriminator checkpoints
- `src/TRAINING_results/` — Generated samples saved during training
- `src/DCGAN_results/` — Generator and discriminator loss plots
- `src/GENERATED_images/` — Final generated character images

Prerequisites
- Python 3.12 (recommended)
- PyTorch and torchvision
- Required Python packages: `numpy`, `matplotlib`, and `Pillow`

Install dependencies

```bash
pip install torch torchvision numpy matplotlib Pillow
```

How to run
1. Open a terminal in the `src/` folder.
2. Train the DCGAN:

```bash
python train.py
```

3. Generate 20 images using the trained model:

```bash
python generate_images.py
```

