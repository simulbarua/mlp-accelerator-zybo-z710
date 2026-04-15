
import os
from pathlib import Path
from torchvision import datasets
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_IMAGE_DIR = os.path.join(DATA_DIR, "MNIST", "png")

if not os.path.exists(SAMPLE_IMAGE_DIR):
    os.makedirs(SAMPLE_IMAGE_DIR)

ds = datasets.MNIST(DATA_DIR, train=False, download=True)
for i in range(100):
    img, label = ds[i]
    img.save(f"{SAMPLE_IMAGE_DIR}/mnist_{i}_label_{label}.png")

print(f"Saved samples to {SAMPLE_IMAGE_DIR}")