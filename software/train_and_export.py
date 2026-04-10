"""
MLP Training and Weight Export for Zybo Z7-10 FPGA Accelerator
===============================================================
Architecture: 784 -> 64 -> 32 -> 10 (ReLU hidden, argmax output)
Dataset:      MNIST (60k train / 10k test)
Quantization: Symmetric 8-bit post-training (per-layer)
Output:       fc1_weights.coe, fc1_bias.coe,
              fc2_weights.coe, fc2_bias.coe,
              fc3_weights.coe, fc3_bias.coe
              scales.txt  (scale factors for PS-side fixed-point math)
"""

import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
COE_DIR    = os.path.join(BASE_DIR, "coe_files")

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(COE_DIR,     exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE  = 256
EPOCHS      = 20
LR          = 1e-3
SEED        = 42

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    784 → FC1(64, ReLU) → FC2(32, ReLU) → FC3(10)
    No softmax: hardware does argmax on raw logits.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64,  32)
        self.fc3 = nn.Linear(32,  10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)           # flatten 28×28 → 784
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)               # raw logits; argmax in hardware
        return x


# ── Data ───────────────────────────────────────────────────────────────────────
def get_loaders():
    # Normalize to [0,1]; hardware receives uint8 pixels scaled by 1/255 on PS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean/std
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, test_loader


# ── Training ───────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n * 100.0


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset) * 100.0


def train(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc  = 0.0
    best_path = os.path.join(WEIGHTS_DIR, "mlp_best.pth")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Test Acc':>10}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        test_acc              = evaluate(model, test_loader)
        scheduler.step()

        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2f}%  {test_acc:>9.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    if best_acc < 95.0:
        print("WARNING: accuracy below 95% target — consider more epochs or QAT.")
    return best_path


# ── Quantization ───────────────────────────────────────────────────────────────
def quantize_tensor(tensor: torch.Tensor, bits: int = 8):
    """
    Symmetric linear quantization.
      scale  = max(|x|) / 127
      x_q    = round(x / scale)  clipped to [-128, 127]
    Returns (int8_tensor, scale_float).
    """
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return torch.zeros_like(tensor, dtype=torch.int8), 1.0
    scale   = max_val / 127.0
    x_q     = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return x_q, scale


def quantize_model(model):
    """
    Quantize all FC layer weights and biases.
    Biases are quantized to 16-bit in the hardware accumulator, but we store
    them as int8 in .coe for simplicity; the RTL sign-extends them.
    Returns dict: {layer_name: {'weight': (int8, scale), 'bias': (int8, scale)}}
    """
    model.cpu().eval()
    layers = {"fc1": model.fc1, "fc2": model.fc2, "fc3": model.fc3}
    quant  = {}
    for name, layer in layers.items():
        w_q, w_scale = quantize_tensor(layer.weight.data.detach())
        b_q, b_scale = quantize_tensor(layer.bias.data.detach())
        quant[name] = {
            "weight": (w_q, w_scale),
            "bias":   (b_q, b_scale),
        }
        print(f"  {name}: weight scale={w_scale:.6f}  bias scale={b_scale:.6f}")
    return quant


# ── COE Export ─────────────────────────────────────────────────────────────────
def write_coe(int8_tensor: torch.Tensor, filepath: str):
    """
    Write a Vivado Block Memory Generator .coe file.
    Radix 16, one hex byte per address (unsigned reinterpretation of int8).
    Row-major order: for a weight matrix [out_features, in_features],
    each row is one output neuron's weight vector — matches RTL address mapping.
    """
    flat     = int8_tensor.flatten().numpy().astype(np.uint8)  # two's complement
    hex_vals = [f"{v:02x}" for v in flat]

    with open(filepath, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(hex_vals))
        f.write(";\n")

    print(f"  Wrote {len(hex_vals)} bytes → {os.path.relpath(filepath, BASE_DIR)}")


def export_coe(quant: dict):
    scales = {}
    for name, tensors in quant.items():
        w_q, w_scale = tensors["weight"]
        b_q, b_scale = tensors["bias"]

        write_coe(w_q, os.path.join(COE_DIR, f"{name}_weights.coe"))
        write_coe(b_q, os.path.join(COE_DIR, f"{name}_bias.coe"))

        scales[f"{name}_weight_scale"] = w_scale
        scales[f"{name}_bias_scale"]   = b_scale

    # Save scales — needed by PS-side C code to reconstruct fixed-point output
    scales_path = os.path.join(COE_DIR, "scales.txt")
    with open(scales_path, "w") as f:
        for k, v in scales.items():
            f.write(f"{k}={v:.8f}\n")
    print(f"  Wrote scale factors → {os.path.relpath(scales_path, BASE_DIR)}")


# ── Verification: software fixed-point inference ───────────────────────────────
def fixed_point_inference(quant: dict, sample_image: np.ndarray) -> int:
    """
    Simulate what the FPGA will compute using integer arithmetic only.
    Uses int16 accumulators (matching the RTL's 16-bit accumulator).
    Returns predicted class index.
    """
    x = sample_image.flatten().astype(np.int16)   # uint8 pixels as int16

    for name in ["fc1", "fc2", "fc3"]:
        w = quant[name]["weight"][0].numpy().astype(np.int16)  # [out, in]
        b = quant[name]["bias"][0].numpy().astype(np.int16)    # [out]
        # MAC: accumulate in int16 (mirrors RTL)
        acc = w @ x + b                                        # [out]
        if name in ("fc1", "fc2"):
            acc = np.maximum(acc, 0)                           # ReLU
        x = acc.astype(np.int16)

    return int(np.argmax(x))


def verify_quantized(model, quant, test_loader, n_samples=1000):
    """Compare float model vs fixed-point simulation on n_samples test images."""
    model.cpu().eval()
    float_correct = 0
    fixed_correct = 0
    count = 0

    # Undo normalization to recover uint8-like values for fixed-point sim
    mean = torch.tensor([0.1307])
    std  = torch.tensor([0.3081])

    with torch.no_grad():
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                if count >= n_samples:
                    break
                # Float inference
                out = model(img.unsqueeze(0))
                float_pred = out.argmax(1).item()

                # Fixed-point: denormalize → uint8 → int16 pixels
                raw = (img * std + mean).clamp(0, 1) * 255
                raw_np = raw.squeeze().numpy().astype(np.uint8)
                fixed_pred = fixed_point_inference(quant, raw_np)

                float_correct += (float_pred == lbl.item())
                fixed_correct += (fixed_pred == lbl.item())
                count += 1
            if count >= n_samples:
                break

    print(f"\nVerification on {count} samples:")
    print(f"  Float model accuracy:       {float_correct/count*100:.2f}%")
    print(f"  Fixed-point sim accuracy:   {fixed_correct/count*100:.2f}%")
    if fixed_correct / count < 0.94:
        print("  WARNING: quantized accuracy < 94% — consider QAT.")
    else:
        print("  Quantized accuracy OK (>= 94%).")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  MLP Accelerator — Train & Export")
    print("  Target: Zybo Z7-10 (Zynq-7010)")
    print("=" * 55)

    # 1. Data
    print("\n[1/5] Loading MNIST...")
    train_loader, test_loader = get_loaders()

    # 2. Train
    print("\n[2/5] Training MLP (784→64→32→10)...")
    model = MLP().to(DEVICE)
    best_path = train(model, train_loader, test_loader)

    # 3. Load best checkpoint
    print(f"\n[3/5] Loading best checkpoint: {os.path.relpath(best_path, BASE_DIR)}")
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()

    # 4. Quantize
    print("\n[4/5] Quantizing weights to INT8...")
    quant = quantize_model(model)

    # 5. Export
    print("\n[5/5] Exporting .coe files...")
    export_coe(quant)

    # Bonus: verify fixed-point simulation matches float
    verify_quantized(model, quant, test_loader)

    print("\nDone. Files ready for Vivado BRAM initialization:")
    for f in sorted(os.listdir(COE_DIR)):
        path = os.path.join(COE_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:<30} {size:>7,} bytes")


if __name__ == "__main__":
    main()
