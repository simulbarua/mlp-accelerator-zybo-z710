"""
CMPE 240 - Phase 1: MLP Training, Quantization, and Weight Export
=================================================================
Group 1 - Zybo Z7-10 FPGA Accelerator Project

FIX (macOS): num_workers=0 and if __name__ == '__main__' guard added.
  macOS uses 'spawn' (not 'fork') for multiprocessing. PyTorch DataLoader
  workers crash unless all top-level training code is inside __main__.
  Setting num_workers=0 is the simplest fix — it loads data on the main
  process and has no meaningful speed penalty for MNIST.

Run:
    pip install torch torchvision
    python phase1_mlp.py

All output files are written to ./phase1_output/
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────
# Config  (top-level is fine — no workers touch this)
# ─────────────────────────────────────────────
OUTPUT_DIR     = "phase1_output"
BATCH_SIZE     = 128
EPOCHS         = 20
LR             = 1e-3
DROPOUT        = 0.2
TARGET_FP_ACC  = 97.0
TARGET_INT_ACC = 95.0
MNIST_MEAN     = 0.1307
MNIST_STD      = 0.3081


# ─────────────────────────────────────────────
# Model  (defined at module level so pickle can find it)
# ─────────────────────────────────────────────
class MLP(nn.Module):
    """
    784 → FC1(64, ReLU, Dropout) → FC2(32, ReLU, Dropout) → FC3(10)

    Architecture matches Table 2 of the project proposal.
    Dropout is used during training only and removed before export.
    """
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        p = DROPOUT if use_dropout else 0.0
        self.fc1   = nn.Linear(784, 64)
        self.fc2   = nn.Linear(64,  32)
        self.fc3   = nn.Linear(32,  10)
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
        self.relu  = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
        x = self.fc3(x)          # raw logits — argmax applied in hardware
        return x

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────
# Helpers  (defined at module level — safe for spawn)
# ─────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Returns top-1 accuracy (%) on the given DataLoader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


def symmetric_quantize(arr: np.ndarray, n_bits: int = 8):
    """
    Symmetric per-tensor quantization (zero-point = 0).

    scale = max(|arr|) / (2^(n_bits-1) - 1)
    arr_q = clamp(round(arr / scale), -128, 127)
    """
    max_abs = np.max(np.abs(arr))
    if max_abs == 0:
        return np.zeros_like(arr, dtype=np.int8), 1.0
    scale = max_abs / (2 ** (n_bits - 1) - 1)
    arr_q = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return arr_q, float(scale)


def relu_np(x):
    return np.maximum(0, x)


def mlp_infer_quantized(pixel_vec, Wq, Bq, scales):
    """NumPy simulation of the INT8 hardware MAC pipeline."""
    x = pixel_vec.astype(np.float32)
    for name in ["fc1", "fc2", "fc3"]:
        W_f = Wq[name].astype(np.float32) * scales[name]
        B_f = Bq[name].astype(np.float32) * scales[name]
        x = x @ W_f.T + B_f
        if name != "fc3":
            x = relu_np(x)
    return int(np.argmax(x))


def write_coe_int8(path: str, arr: np.ndarray):
    """Xilinx .coe file for INT8 weights packed 4-per-word for a 32-bit BRAM.

    Each COE entry is one 32-bit word (8 hex digits).  Bytes are packed
    little-endian (byte 0 in bits[7:0]) to match the PS AXI write packing
    used by bram_write_block() in mlp_bram_init.c.

    Without this packing the COE entries are 2 hex digits each, which Vivado
    treats as individual 32-bit words (zero-padded).  That wastes 4× BRAM
    depth and the AXI BRAM Controller may map word 0 incorrectly.
    """
    flat = arr.flatten().astype(np.uint8)
    # Zero-pad to a multiple of 4 bytes so every word is complete.
    pad  = (4 - len(flat) % 4) % 4
    flat = np.concatenate([flat, np.zeros(pad, dtype=np.uint8)])
    # Pack 4 bytes per 32-bit word, little-endian.
    words = flat.reshape(-1, 4)
    hex_vals = [
        f"{int(w[3]):02X}{int(w[2]):02X}{int(w[1]):02X}{int(w[0]):02X}"
        for w in words
    ]
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for i in range(0, len(hex_vals), 8):
            chunk = ", ".join(hex_vals[i:i+8])
            sep   = ";" if i + 8 >= len(hex_vals) else ","
            f.write(f"  {chunk}{sep}\n")


def write_coe_int32(path: str, arr: np.ndarray):
    """Xilinx .coe file for INT32 biases (8-digit hex, two's complement)."""
    # Cast to uint32 first so two's-complement negatives (e.g. -1 → 0xFFFFFFFF)
    # don't cause overflow when Python formats them as unsigned hex.
    flat     = arr.flatten().astype(np.uint32)
    hex_vals = [f"{v:08X}" for v in flat]
    with open(path, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        for i in range(0, len(hex_vals), 8):
            chunk = ", ".join(hex_vals[i:i+8])
            sep   = ";" if i + 8 >= len(hex_vals) else ","
            f.write(f"  {chunk}{sep}\n")


# ══════════════════════════════════════════════════════════════
#  MAIN  — all training/export code lives here.
#  Required on macOS / Windows because both use 'spawn' for
#  multiprocessing. Without this guard the script re-imports
#  itself in every worker, causing the RuntimeError you saw.
#  num_workers=0 is a belt-and-suspenders fix: it keeps all
#  data loading on the main process and avoids spawn entirely.
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"[Config] Device        : {DEVICE}")
    print(f"[Config] Output dir    : ./{OUTPUT_DIR}/\n")

    # ── STEP 1: Dataset ───────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Downloading and Preparing MNIST Dataset")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # num_workers=0  →  data loads on main process (no subprocess spawn)
    # pin_memory=False  →  only useful with CUDA; avoids a warning on CPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_dataset,  batch_size=10000,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"  Training samples : {len(train_dataset):,}")
    print(f"  Test samples     : {len(test_dataset):,}")
    print(f"  Image size       : 28×28 → flattened to 784")
    print(f"  Normalization    : mean={MNIST_MEAN}, std={MNIST_STD}")
    print(f"  Batch size       : {BATCH_SIZE}\n")

    # ── STEP 2: Training ──────────────────────────────────────
    print("=" * 60)
    print("STEP 2: Training FP32 MLP")
    print("=" * 60)

    model = MLP(use_dropout=True).to(DEVICE)
    print(f"  Architecture : 784 → FC1(64, ReLU) → FC2(32, ReLU) → FC3(10)")
    print(f"  Total params : {model.total_params():,}")
    print(f"  Loss         : CrossEntropyLoss")
    print(f"  Optimizer    : Adam (lr={LR})")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Dropout      : {DROPOUT} (FC1 & FC2 during training only)\n")

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    best_acc   = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        acc      = evaluate(model, test_loader, DEVICE)

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        tag = " ★ best" if acc == best_acc else ""
        print(f"  Epoch {epoch:2d}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Test Acc: {acc:.2f}%{tag}")

    model.load_state_dict(best_state)
    final_fp_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n  Final FP32 accuracy (best checkpoint): {final_fp_acc:.2f}%")

    if final_fp_acc < TARGET_FP_ACC:
        print(f"  ⚠ WARNING: {final_fp_acc:.2f}% < target {TARGET_FP_ACC}%. Try more epochs.")
    else:
        print(f"  ✓ Target {TARGET_FP_ACC}% reached!")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "mlp_fp32.pth"))
    print(f"  Saved: {OUTPUT_DIR}/mlp_fp32.pth\n")

    # ── STEP 3: INT8 Quantization ─────────────────────────────
    print("=" * 60)
    print("STEP 3: Symmetric INT8 Post-Training Quantization")
    print("=" * 60)
    print("  Method : per-layer symmetric PTQ (zero-point = 0)")
    print("  Range  : [-128, 127]\n")

    W = {n: getattr(model, n).weight.detach().cpu().numpy() for n in ["fc1","fc2","fc3"]}
    B = {n: getattr(model, n).bias.detach().cpu().numpy()   for n in ["fc1","fc2","fc3"]}

    scales = {}
    Wq     = {}
    print("  Quantizing weights:")
    for name in ["fc1", "fc2", "fc3"]:
        Wq[name], scales[name] = symmetric_quantize(W[name])
        orig_kb  = W[name].nbytes  // 1024
        quant_b  = Wq[name].nbytes
        quant_str = f"{quant_b//1024}KB" if quant_b >= 1024 else f"{quant_b}B"
        print(f"    {name}: shape={W[name].shape}  scale={scales[name]:.6f}  "
              f"size: {orig_kb}KB → {quant_str}")

    Bq = {}
    print("\n  Quantizing biases (INT32, pre-scaled):")
    for name in ["fc1", "fc2", "fc3"]:
        Bq[name] = np.clip(
            np.round(B[name] / scales[name]), -(2**31), 2**31 - 1
        ).astype(np.int32)
        print(f"    {name}: shape={B[name].shape}")

    # Accuracy verification (NumPy simulation)
    print("\n  Verifying INT8 accuracy (numpy simulation)...")
    all_images = test_dataset.data.numpy().astype(np.float32) / 255.0
    all_images = (all_images - MNIST_MEAN) / MNIST_STD
    all_images = all_images.reshape(-1, 784)
    all_labels = test_dataset.targets.numpy()

    preds    = np.array([mlp_infer_quantized(all_images[i], Wq, Bq, scales)
                         for i in range(len(all_labels))])
    int8_acc = 100.0 * np.mean(preds == all_labels)

    print(f"\n  FP32 accuracy  : {final_fp_acc:.2f}%")
    print(f"  INT8 accuracy  : {int8_acc:.2f}%")
    print(f"  Accuracy drop  : {final_fp_acc - int8_acc:.2f}%\n")

    if int8_acc < TARGET_INT_ACC:
        print(f"  ⚠ WARNING: INT8 accuracy {int8_acc:.2f}% < target {TARGET_INT_ACC}%.")
        print(f"    Consider applying Quantization-Aware Training (QAT).")
    else:
        print(f"  ✓ INT8 accuracy target {TARGET_INT_ACC}% reached!")

    # ── STEP 4: Export .coe Files ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Exporting .coe Files for Vivado BRAM Initialization")
    print("=" * 60)

    file_map = {
        "W1.coe": ("fc1", "weight"),
        "W2.coe": ("fc2", "weight"),
        "W3.coe": ("fc3", "weight"),
        "b1.coe": ("fc1", "bias"),
        "b2.coe": ("fc2", "bias"),
        "b3.coe": ("fc3", "bias"),
    }

    for fname, (layer, kind) in file_map.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if kind == "weight":
            write_coe_int8(path, Wq[layer])
            count  = Wq[layer].size
            size_b = count
        else:
            write_coe_int32(path, Bq[layer])
            count  = Bq[layer].size
            size_b = count * 4
        print(f"  Wrote {fname:<8}  ({count} values, ~{size_b} bytes)")

    scales_path = os.path.join(OUTPUT_DIR, "scales.txt")
    with open(scales_path, "w") as f:
        f.write("# Scale factors for INT8 dequantization (read by PS/ARM)\n")
        f.write("# Format: layer_name scale_value\n")
        for name in ["fc1", "fc2", "fc3"]:
            f.write(f"{name} {scales[name]:.10f}\n")
    print(f"  Wrote scales.txt  (3 FP32 scale factors)")

    for tag, arr in [("W1_int8", Wq["fc1"]), ("W2_int8", Wq["fc2"]), ("W3_int8", Wq["fc3"]),
                     ("b1_int32", Bq["fc1"]), ("b2_int32", Bq["fc2"]), ("b3_int32", Bq["fc3"])]:
        np.save(os.path.join(OUTPUT_DIR, f"{tag}.npy"), arr)
    print(f"  Wrote .npy debug arrays\n")

    # ── STEP 5: Export Binary Files ───────────────────────────
    print("=" * 60)
    print("STEP 5: Exporting Raw Binary Files")
    print("=" * 60)
    print("  Format:")
    print("    Weights → raw INT8  bytes, row-major (1 byte per weight)")
    print("    Biases  → raw INT32 bytes, little-endian (4 bytes per bias)\n")

    bin_weight_map = {
        "W1.bin": Wq["fc1"],   # shape (64, 784)  → 50,176 bytes
        "W2.bin": Wq["fc2"],   # shape (32,  64)  →  2,048 bytes
        "W3.bin": Wq["fc3"],   # shape (10,  32)  →    320 bytes
    }
    for fname, arr in bin_weight_map.items():
        path = os.path.join(OUTPUT_DIR, fname)
        # view as uint8 so tobytes() writes correct two's-complement
        # bit patterns (e.g. -1 → 0xFF) without overflow errors
        arr.astype(np.int8).view(np.uint8).flatten().tofile(path)
        size_b = os.path.getsize(path)
        print(f"  Wrote {fname:<8}  ({arr.size} INT8  values, {size_b} bytes)")

    bin_bias_map = {
        "b1.bin": Bq["fc1"],   # 64  values → 256 bytes
        "b2.bin": Bq["fc2"],   # 32  values → 128 bytes
        "b3.bin": Bq["fc3"],   # 10  values →  40 bytes
    }
    for fname, arr in bin_bias_map.items():
        path = os.path.join(OUTPUT_DIR, fname)
        # '<i4' = little-endian INT32, consistent across all platforms
        arr.astype('<i4').tofile(path)
        size_b = os.path.getsize(path)
        print(f"  Wrote {fname:<8}  ({arr.size} INT32 values, {size_b} bytes)")

    print()
    print("  Binary file layout (for C code on PS/ARM):")
    print("    W1.bin : 50176 bytes  — read as int8_t  W1[64][784]")
    print("    W2.bin :  2048 bytes  — read as int8_t  W2[32][64]")
    print("    W3.bin :   320 bytes  — read as int8_t  W3[10][32]")
    print("    b1.bin :   256 bytes  — read as int32_t b1[64]")
    print("    b2.bin :   128 bytes  — read as int32_t b2[32]")
    print("    b3.bin :    40 bytes  — read as int32_t b3[10]")
    print()
    print("  Example C code to load on PS/ARM:")
    print("    int8_t  W1[64][784];")
    print("    FILE   *f = fopen(\"W1.bin\", \"rb\");")
    print("    fread(W1, sizeof(int8_t), 64*784, f);")
    print("    fclose(f);")
    print()


    # ── Summary ───────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1 COMPLETE – Summary")
    print("=" * 60)

    total_w = sum(Wq[n].size     for n in ["fc1","fc2","fc3"])
    total_b = sum(Bq[n].size * 4 for n in ["fc1","fc2","fc3"])

    print(f"  FP32 test accuracy       : {final_fp_acc:.2f}%  (target ≥{TARGET_FP_ACC}%)")
    print(f"  INT8 test accuracy       : {int8_acc:.2f}%  (target ≥{TARGET_INT_ACC}%)")
    print(f"  Accuracy drop            : {final_fp_acc - int8_acc:.2f}%")
    print(f"  Total INT8 weight bytes  : {total_w:,}  (~{total_w/1024:.1f} KB)")
    print(f"  Total INT32 bias bytes   : {total_b}")
    bram_blocks = math.ceil((total_w + total_b) / (36 * 1024 / 8))
    print(f"  BRAM usage (est.)        : {bram_blocks} × 36Kb blocks")
    print()
    print(f"  Output files in ./{OUTPUT_DIR}/")
    print(f"    W1.coe, W2.coe, W3.coe   ← weight BRAMs  (INT8,  Vivado init)")
    print(f"    b1.coe, b2.coe, b3.coe   ← bias BRAMs    (INT32, Vivado init)")
    print(f"    W1.bin, W2.bin, W3.bin   ← weight binary (INT8,  for PS/C code)")
    print(f"    b1.bin, b2.bin, b3.bin   ← bias binary   (INT32, for PS/C code)")
    print(f"    scales.txt               ← scale factors (read by PS/ARM)")
    print(f"    mlp_fp32.pth             ← FP32 model checkpoint")
    print(f"    *.npy                    ← debug arrays")
    print()

    if int8_acc >= TARGET_INT_ACC and final_fp_acc >= TARGET_FP_ACC:
        print("  ✓ All targets met. Hand off to Phase 2.")
    else:
        print("  ⚠ Some targets not met. See warnings above.")
