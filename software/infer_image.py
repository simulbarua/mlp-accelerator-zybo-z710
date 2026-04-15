import argparse
import os

from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, datasets


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS = os.path.join(BASE_DIR, "weights", "mlp_best.pth")

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_transform(invert: bool):
    ops = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
    ]
    if invert:
        ops.append(transforms.Lambda(lambda img: ImageOps.invert(img)))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )
    return transforms.Compose(ops)


def load_model(weights_path: str) -> MLP:
    model = MLP()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model: MLP, image_path: str, invert: bool):
    transform = build_transform(invert=invert)
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    return pred, probs


def main():
    parser = argparse.ArgumentParser(description="Run MNIST-style inference on an image.")
    parser.add_argument("image", help="Path to the image to classify.")
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help=f"Path to model weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the image before inference. Use this if the digit is dark on a light background.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = load_model(args.weights)
    pred, probs = predict_image(model, args.image, invert=args.invert)

    print(f"Predicted digit: {pred}")
    print("Class probabilities:")
    for idx, prob in enumerate(probs.tolist()):
        print(f"  {idx}: {prob:.6f}")


if __name__ == "__main__":
    main()
