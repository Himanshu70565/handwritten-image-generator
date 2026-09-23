import os

import torch
from torchvision.utils import save_image

from dcgan import Generator


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

model = Generator().to(device)
model.load_state_dict(torch.load("TRAINING_models/generator_trained.pth", map_location=device))
model.eval()

os.makedirs("GENERATED_images", exist_ok=True)
noise = torch.randn(20, 100, 1, 1, device=device)

with torch.no_grad():
    images = model(noise)

for index, image in enumerate(images, start=1):
    save_image(image, f"GENERATED_images/image_{index:02d}.png", normalize=True, value_range=(-1, 1))

print("Saved 20 images to GENERATED_images/")
