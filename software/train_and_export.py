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
              firmware/weights_biases.h  (ARM C header for BRAM loading)
"""

import os
import math
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR  = os.path.join(BASE_DIR, "weights_and_biases")
COE_DIR      = os.path.join(BASE_DIR, "coe_files")
FIRMWARE_DIR = os.path.join(BASE_DIR, "..", "hardware", "firmware")
RTL_DIR      = os.path.join(BASE_DIR, "..", "hardware", "rtl")

os.makedirs(DATA_DIR,     exist_ok=True)
os.makedirs(WEIGHTS_DIR,  exist_ok=True)
os.makedirs(COE_DIR,      exist_ok=True)
os.makedirs(FIRMWARE_DIR, exist_ok=True)
os.makedirs(RTL_DIR,      exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE  = 256
EPOCHS      = 20
LR          = 1e-3
SEED        = 42
NUM_WORKERS = 2 if os.name == "nt" else 0     # for DataLoader; adjust based on your CPU cores

# ── Input quantization constants (MNIST normalization) ─────────────────────────
# The model receives inputs normalized by (pixel/255 - mean) / std.
# The hardware PS side will normalize uint8 pixels the same way, then
# quantize to INT8 using INPUT_SCALE before writing to the Input BRAM.
INPUT_MEAN  = 0.1307
INPUT_STD   = 0.3081
# Symmetric INT8 scale: max |normalized value| / 127
# max = (1.0 - 0.1307) / 0.3081 ≈ 2.822
INPUT_SCALE = max(abs((1.0 - INPUT_MEAN) / INPUT_STD),
                  abs((0.0 - INPUT_MEAN) / INPUT_STD)) / 127.0

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    784 → FC1(64, ReLU, Dropout) → FC2(32, ReLU, Dropout) → FC3(10)
    No softmax: hardware does argmax on raw logits.
    Dropout is applied during training only and disabled during export/inference.
    """
    def __init__(self, dropout_p: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64,  32)
        self.fc3 = nn.Linear(32,  10)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_p)
        self.drop2 = nn.Dropout(dropout_p)

    def forward(self, x):
        x = x.view(-1, 784)           # flatten 28×28 → 784
        x = self.drop1(self.relu(self.fc1(x)))
        x = self.drop2(self.relu(self.fc2(x)))
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
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
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
def calibrate_activation_scales(model, loader, n_batches: int = 50):
    """
    Run a forward pass over n_batches of calibration data and record the
    maximum post-ReLU activation value at each hidden layer output.

    Returns (act1_scale, act2_scale) where scale = max_val / 127.
    These are used to compute the inter-layer requantization right-shifts
    that bring INT32 accumulators back to INT8 range between FC layers.

    Without this step, every positive accumulator (which is ~10 000× larger
    than its float equivalent) gets clipped to 127, making the hidden layer
    effectively binary and collapsing accuracy to ~27%.
    """
    model.cpu().eval()
    max_act1, max_act2 = 0.0, 0.0
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(loader):
            if batch_idx >= n_batches:
                break
            x  = images.view(-1, 784)
            a1 = torch.relu(model.fc1(x))
            a2 = torch.relu(model.fc2(a1))
            max_act1 = max(max_act1, a1.abs().max().item())
            max_act2 = max(max_act2, a2.abs().max().item())
    return max_act1 / 127.0, max_act2 / 127.0


def compute_req_shift(w_scale: float, x_scale: float, act_scale: float) -> int:
    """
    Compute the integer right-shift k such that 2^(-k) ≈ M_req.

    M_req = (w_scale × x_scale) / act_scale
          = scale of one INT32 accumulator unit / scale of one output INT8 unit

    k = round(log2(1 / M_req)) = round(log2(act_scale / (w_scale × x_scale)))

    After the shift, the INT32 accumulator is in the same numerical range
    as INT8 [-128, 127], so ReLU + clip is meaningful.
    """
    M_req = (w_scale * x_scale) / act_scale
    return max(0, int(round(math.log2(1.0 / M_req))))


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


def quantize_model(model, train_loader):
    """
    Quantize FC layer weights to INT8 and biases to INT32 in accumulator.
    Scales, and compute inter-layer requantization right-shifts.

    Why shifts are needed
    ---------------------
    After a FC layer MAC:  acc_int32 ≈ acc_float / (w_scale * x_scale)
    For typical MNIST values this is ~10 000 to 50 000.  Clipping that
    directly to [0, 127] makes every positive neuron output 127 (binary),
    collapsing accuracy.  The right-shift k brings acc_int32 back to INT8
    range:  act_int8 = clip(ReLU(acc_int32 >> k), 0, 127)
    where k = round(log2(act_scale / (w_scale * x_scale))).

    Returns dict with keys:
      layer_name → {'weight': (int8, scale), 'bias': (int32, bias_scale)}
      'input_scale'    → float
      'act1_scale'     → float  (FC1 post-ReLU calibrated scale)
      'act2_scale'     → float  (FC2 post-ReLU calibrated scale)
      'req_shift_fc1'  → int    (right-shift for FC1 output requantization)
      'req_shift_fc2'  → int    (right-shift for FC2 output requantization)
    """
    model.cpu().eval()
    layers = {"fc1": model.fc1, "fc2": model.fc2, "fc3": model.fc3}
    quant  = {"input_scale": INPUT_SCALE}
    for name, layer in layers.items():
        w_q, w_scale = quantize_tensor(layer.weight.data.detach())
        quant[name] = {
            "weight": (w_q, w_scale),
        }
        print(f"  {name}: weight scale={w_scale:.6f}")
    print(f"  input_scale={INPUT_SCALE:.6f}  (INPUT_MEAN={INPUT_MEAN}, INPUT_STD={INPUT_STD})")

    # Calibrate post-ReLU activation ranges
    print("  Calibrating activation scales (50 batches)...")
    act1_scale, act2_scale = calibrate_activation_scales(model, train_loader)
    quant["act1_scale"] = act1_scale
    quant["act2_scale"] = act2_scale

    # Compute requantization shifts
    # FC1: input is quantized image (scale = INPUT_SCALE)
    # FC2: input is act1 (scale = act1_scale)
    w1_scale = quant["fc1"]["weight"][1]
    w2_scale = quant["fc2"]["weight"][1]
    req_shift_fc1 = compute_req_shift(w1_scale, INPUT_SCALE,  act1_scale)
    req_shift_fc2 = compute_req_shift(w2_scale, act1_scale,   act2_scale)
    quant["req_shift_fc1"] = req_shift_fc1
    quant["req_shift_fc2"] = req_shift_fc2

    # Quantize biases into the INT32 accumulator domain.
    # bias_scale must match the scale of one accumulator LSB:
    #   FC1 accumulator scale = w1_scale * input_scale
    #   FC2 accumulator scale = w2_scale * act1_scale
    #   FC3 accumulator scale = w3_scale * act2_scale
    bias_input_scales = {
        "fc1": INPUT_SCALE,
        "fc2": act1_scale,
        "fc3": act2_scale,
    }
    for name in ["fc1", "fc2", "fc3"]:
        w_scale = quant[name]["weight"][1]
        b_scale = w_scale * bias_input_scales[name]
        b_float = layers[name].bias.data.detach().cpu().numpy()
        b_q = np.clip(
            np.round(b_float / b_scale),
            -(2**31),
            2**31 - 1,
        ).astype(np.int32)
        quant[name]["bias"] = (b_q, b_scale)
        print(f"  {name}: bias scale={b_scale:.8f}  ({b_q.shape[0]} INT32 values)")

    print(f"  act1_scale={act1_scale:.6f}  req_shift_fc1={req_shift_fc1}")
    print(f"  act2_scale={act2_scale:.6f}  req_shift_fc2={req_shift_fc2}")
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

# Write INT32 .coe files for biases (after scaling to accumulator domain)
def write_coe_int32(int32_tensor, filepath: str):
    """
    Write a Vivado Block Memory Generator .coe file for INT32 values.
    Radix 16, one 8-hex-digit word per address (two's complement).
    """
    flat = np.asarray(int32_tensor, dtype=np.int32).flatten().astype(np.uint32)
    hex_vals = [f"{v:08X}" for v in flat]

    with open(filepath, "w") as f:
        f.write("memory_initialization_radix=16;\n")
        f.write("memory_initialization_vector=\n")
        f.write(",\n".join(hex_vals))
        f.write(";\n")

    print(f"  Wrote {len(hex_vals)} words → {os.path.relpath(filepath, BASE_DIR)}")


def export_coe(quant: dict):
    layer_names = ["fc1", "fc2", "fc3"]
    scales = {
        "input_scale":    quant["input_scale"],
        "input_mean":     INPUT_MEAN,
        "input_std":      INPUT_STD,
        "req_shift_fc1":  quant["req_shift_fc1"],
        "req_shift_fc2":  quant["req_shift_fc2"],
    }
    for name in layer_names:
        w_q, w_scale = quant[name]["weight"]
        b_q, b_scale = quant[name]["bias"]

        write_coe(w_q, os.path.join(COE_DIR, f"{name}_weights.coe"))
        write_coe_int32(b_q, os.path.join(COE_DIR, f"{name}_bias.coe"))

        scales[f"{name}_weight_scale"] = w_scale
        scales[f"{name}_bias_scale"]   = b_scale

    # Save scales — needed by PS-side C code to quantize the input
    scales_path = os.path.join(COE_DIR, "scales.txt")
    with open(scales_path, "w", encoding="utf-8") as f:
        for k, v in scales.items():
            line = f"{k}={v}\n" if isinstance(v, int) else f"{k}={v:.8f}\n"
            f.write(line)
    print(f"  Wrote scale factors → {os.path.relpath(scales_path, BASE_DIR)}")


# ── C Header Export ────────────────────────────────────────────────────────────
def _pack_int8_to_uint32(int8_tensor: torch.Tensor) -> list:
    """
    Pack a flat int8 tensor into a list of uint32 values (little-endian,
    4 bytes per word).  Zero-pads to the next 4-byte boundary.
    Byte order within each word:
      word[i] = byte[4i+0] | (byte[4i+1]<<8) | (byte[4i+2]<<16) | (byte[4i+3]<<24)
    byte[4i+0] lands at the lowest BRAM address — matches ARM Cortex-A9
    little-endian AXI writes.
    """
    raw = np.asarray(int8_tensor, dtype=np.int8).flatten().tobytes()
    pad = (-len(raw)) % 4          # bytes needed to reach next 4-byte boundary
    raw += b'\x00' * pad
    return list(struct.unpack(f'<{len(raw) // 4}I', raw))


def export_header(quant: dict, out_dir: str = FIRMWARE_DIR):
    """
    Write firmware/weights_biases.h.

    Contains:
      - Network dimension #defines
      - uint32_t word-count #defines for each parameter block
      - BRAM byte-offset #defines (contiguous layout, weights before biases
        within each layer: fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b)
      - float32 quantization scale arrays
      - static const uint32_t arrays for every weight matrix and bias vector
        (INT8 values packed 4-per-word, little-endian, row-major [out, in])

    The file is meant to be #included by mlp_bram_init.c only.
    """
    os.makedirs(out_dir, exist_ok=True)

    layer_names = ["fc1", "fc2", "fc3"]

    # Pack all tensors into uint32 word lists
    packed_w = {n: _pack_int8_to_uint32(quant[n]["weight"][0]) for n in layer_names}
    packed_b = {
        n: list(np.asarray(quant[n]["bias"][0], dtype=np.int32).astype("<i4").view(np.uint32))
        for n in layer_names
    }

    # Build contiguous BRAM offset map (bytes)
    bram_offsets: dict = {}
    cursor = 0
    for n in layer_names:
        bram_offsets[f"{n}_weight"] = cursor;  cursor += len(packed_w[n]) * 4
        bram_offsets[f"{n}_bias"]   = cursor;  cursor += len(packed_b[n]) * 4
    total_bytes = cursor

    def fmt_array(words: list, cols: int = 8) -> str:
        """Format uint32 list as indented C hex literals."""
        lines = []
        for i in range(0, len(words), cols):
            chunk = words[i : i + cols]
            lines.append("    " + ", ".join(f"0x{w:08X}U" for w in chunk))
        return ",\n".join(lines)

    out: list = []

    # ── File header ────────────────────────────────────────────────────────────
    out += [
        "/* weights_biases.h — AUTO-GENERATED by train_and_export.py */",
        "/* DO NOT EDIT: re-run train_and_export.py to regenerate.   */",
        "/*                                                           */",
        "/* INT8 weights/biases packed as uint32_t (little-endian,   */",
        "/* 4 bytes per word).  Row-major [out_features, in_features] */",
        "/* for weight matrices.  Zero-padded to 4-byte boundary.    */",
        "",
        "#ifndef WEIGHTS_BIASES_H",
        "#define WEIGHTS_BIASES_H",
        "",
        "#include <stdint.h>",
        "",
    ]

    # ── Network dimensions ─────────────────────────────────────────────────────
    out += [
        "/* ── Network dimensions ──────────────────────────────────────── */",
        "#define MLP_FC1_IN    784U",
        "#define MLP_FC1_OUT    64U",
        "#define MLP_FC2_IN     64U",
        "#define MLP_FC2_OUT    32U",
        "#define MLP_FC3_IN     32U",
        "#define MLP_FC3_OUT    10U",
        "",
    ]

    # ── Array sizes ────────────────────────────────────────────────────────────
    out += ["/* ── Array sizes (uint32_t words per parameter block) ────────── */"]
    for n in layer_names:
        out.append(f"#define {n.upper()}_WEIGHT_WORDS  {len(packed_w[n])}U")
        out.append(f"#define {n.upper()}_BIAS_WORDS    {len(packed_b[n])}U")
    out.append("")

    # ── BRAM byte offsets ──────────────────────────────────────────────────────
    out += [
        "/* ── BRAM byte offsets from MLP_BRAM_BASE_ADDR ─────────────────── */",
        "/* Set MLP_BRAM_BASE_ADDR in mlp_bram_init.h to match the address  */",
        "/* assigned to your AXI BRAM Controller in Vivado's Address Editor. */",
    ]
    for n in layer_names:
        out.append(f"#define {n.upper()}_WEIGHT_BRAM_OFFSET  0x{bram_offsets[f'{n}_weight']:08X}UL")
        out.append(f"#define {n.upper()}_BIAS_BRAM_OFFSET    0x{bram_offsets[f'{n}_bias']:08X}UL")
    out.append(f"#define MLP_PARAM_TOTAL_BYTES   {total_bytes}U  /* {total_bytes / 1024:.1f} KB */")
    out.append("")

    # ── Scale factors ──────────────────────────────────────────────────────────
    w_sc = ", ".join(f"{quant[n]['weight'][1]:.8f}f" for n in layer_names)
    b_sc = ", ".join(f"{quant[n]['bias'][1]:.8f}f"   for n in layer_names)
    out += [
        "/* ── Input normalization constants (MNIST) ───────────────────────── */",
        "/* PS C code must normalize each uint8 pixel before writing to BRAM: */",
        "/*   x_float = (pixel / 255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD     */",
        "/*   x_int8  = clamp(roundf(x_float / MLP_INPUT_SCALE), -128, 127)   */",
        f"#define MLP_INPUT_MEAN   {INPUT_MEAN:.4f}f",
        f"#define MLP_INPUT_STD    {INPUT_STD:.4f}f",
        f"#define MLP_INPUT_SCALE  {quant['input_scale']:.8f}f",
        "",
        "/* ── Inter-layer requantization right-shifts ─────────────────────── */",
        "/* RTL applies:  act_int8 = clip(ReLU(acc_int32 >> k), 0, 127)       */",
        "/* k is computed so that 2^(-k) approximates M_req =                 */",
        "/*   (w_scale * x_scale) / act_scale                                 */",
        "/* Without these shifts, every positive FC accumulator (magnitude    */",
        "/* ~10 000-50 000) clips to 127, collapsing hidden layers to binary. */",
        f"#define MLP_REQ_SHIFT_FC1  {quant['req_shift_fc1']}U",
        f"#define MLP_REQ_SHIFT_FC2  {quant['req_shift_fc2']}U",
        "",
        "/* ── Weight / bias quantization scale factors (fc1, fc2, fc3 order) */",
        "/* weight_scale[i]: dequantize via  w_float = w_int8 * scale          */",
        f"static const float mlp_weight_scales[3] = {{ {w_sc} }};",
        f"static const float mlp_bias_scales[3]   = {{ {b_sc} }};",
        "",
    ]

    # ── Parameter arrays ───────────────────────────────────────────────────────
    out.append("/* ── Parameter arrays ────────────────────────────────────────── */")
    out.append("/* Include this header in exactly ONE .c file (mlp_bram_init.c). */")
    out.append("")

    for n in layer_names:
        w_shape = quant[n]["weight"][0].shape   # (out, in)
        b_shape = quant[n]["bias"][0].shape     # (out,)

        out.append(f"/* {n} weights  [{w_shape[0]}×{w_shape[1]} INT8]"
                   f"  →  {len(packed_w[n])} uint32_t words */")
        out.append(f"static const uint32_t {n}_weights[{n.upper()}_WEIGHT_WORDS] = {{")
        out.append(fmt_array(packed_w[n]))
        out.append("};")
        out.append("")

        out.append(f"/* {n} bias  [{b_shape[0]} INT32]"
                   f"  →  {len(packed_b[n])} uint32_t words */")
        out.append(f"static const int32_t {n}_bias[{n.upper()}_BIAS_WORDS] = {{")
        out.append("    " + ", ".join(str(int(v)) for v in np.asarray(quant[n]["bias"][0]).flatten()))
        out.append("};")
        out.append("")

    out.append("#endif /* WEIGHTS_BIASES_H */")

    header_path = os.path.join(out_dir, "weights_biases.h")
    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

    print(f"  Wrote {total_bytes / 1024:.1f} KB params → "
          f"{os.path.relpath(header_path, BASE_DIR)}")
    print(f"  BRAM layout:")
    for n in layer_names:
        print(f"    0x{bram_offsets[f'{n}_weight']:08X}  {n}_weights"
              f"  ({len(packed_w[n])*4} bytes)")
        print(f"    0x{bram_offsets[f'{n}_bias']:08X}  {n}_bias"
              f"  ({len(packed_b[n])*4} bytes)")
    print(f"    Total: {total_bytes} bytes ({total_bytes/1024:.1f} KB)")

    # Also write Verilog parameter header so RTL picks up shifts without
    # the user having to manually edit mlp_engine.v after each retraining.
    export_verilog_params(quant, out_dir)


def export_verilog_params(quant: dict, out_dir: str = FIRMWARE_DIR):
    """
    Write rtl/mlp_params.vh with `define constants for the inter-layer
    requantization right-shifts.  mlp_engine.v `includes this file so
    the shifts are automatically updated whenever weights are retrained.
    """
    vh_path   = os.path.join(RTL_DIR, "mlp_params.vh")
    lines = [
        "// mlp_params.vh — AUTO-GENERATED by train_and_export.py",
        "// DO NOT EDIT: re-run train_and_export.py to regenerate.",
        "//",
        "// Inter-layer requantization right-shifts for mlp_engine.v",
        "// act_int8 = clip(ReLU(acc_int32 >>> shift), 0, 127)",
        f"`define MLP_REQ_SHIFT_FC1  5'd{quant['req_shift_fc1']}",
        f"`define MLP_REQ_SHIFT_FC2  5'd{quant['req_shift_fc2']}",
    ]
    with open(vh_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote RTL params → {os.path.relpath(vh_path, BASE_DIR)}")


# ── Verification: software fixed-point inference ───────────────────────────────
def fixed_point_inference(quant: dict, sample_image: np.ndarray) -> int:
    """
    Simulate exactly what the FPGA will compute.

    Input path (mirrors C driver + RTL):
      uint8 pixel → normalize → quantize to INT8 → write to Input BRAM
      RTL reads INT8, sign-extends, multiplies with INT8 weight → INT32 acc

    Key corrections vs the old (broken) version:
      1. Inputs are normalized and quantized to INT8, not raw uint8.
         Raw uint8 (0–255) fed to weights trained on ~[-0.42, 2.82] float inputs
         causes every accumulator to overflow catastrophically.
      2. Accumulators are INT32, not INT16.
         FC1: 784 terms × max 127×127 = 12.6M >> int16 range (32 767).
      3. Post-ReLU activations are clipped to [0, 127] (INT8) before the
         next layer, matching the RTL's relu_clip() function.
    """
    input_scale    = quant["input_scale"]
    req_shift_fc1  = quant["req_shift_fc1"]
    req_shift_fc2  = quant["req_shift_fc2"]

    # Normalize uint8 pixels exactly as the PS C driver will do
    x_float = (sample_image.flatten().astype(np.float32) / 255.0
               - INPUT_MEAN) / INPUT_STD
    # Quantize to INT8
    x = np.clip(np.round(x_float / input_scale), -128, 127).astype(np.int8)

    for name in ["fc1", "fc2", "fc3"]:
        w   = quant[name]["weight"][0].numpy().astype(np.int32)
        b   = np.asarray(quant[name]["bias"][0], dtype=np.int32)
        acc = w @ x.astype(np.int32) + b           # INT32 accumulator

        if name == "fc1":
            # Requantize: right-shift brings acc from INT32 range back to INT8 range.
            # Without this, every positive accumulator (≈10 000–50 000) clips to 127.
            # k = req_shift_fc1 ≈ round(log2(act1_scale / (w1_scale × input_scale)))
            acc = np.clip(np.right_shift(np.maximum(acc, 0), req_shift_fc1), 0, 127).astype(np.int8)
        elif name == "fc2":
            acc = np.clip(np.right_shift(np.maximum(acc, 0), req_shift_fc2), 0, 127).astype(np.int8)
        # FC3: keep as INT32 for argmax
        x = acc

    return int(np.argmax(x))


def verify_quantized(model, quant, test_loader, n_samples=1000):
    """Compare float model vs fixed-point simulation on n_samples test images."""
    model.cpu().eval()
    float_correct = 0
    fixed_correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                if count >= n_samples:
                    break

                # Float inference (normalized tensor, as trained)
                float_pred = model(img.unsqueeze(0)).argmax(1).item()

                # Fixed-point: recover uint8 pixels → pass to fixed_point_inference
                # (which normalizes and quantizes internally, mirroring C driver)
                raw_np = ((img * INPUT_STD + INPUT_MEAN)
                          .clamp(0, 1)
                          .mul(255)
                          .squeeze()
                          .numpy()
                          .astype(np.uint8))
                fixed_pred = fixed_point_inference(quant, raw_np)

                float_correct += (float_pred == lbl.item())
                fixed_correct += (fixed_pred == lbl.item())
                count += 1
            if count >= n_samples:
                break

    print(f"\nVerification on {count} samples:")
    print(f"  Float model accuracy:       {float_correct/count*100:.2f}%")
    print(f"  Fixed-point sim accuracy:   {fixed_correct/count*100:.2f}%")
    gap = (float_correct - fixed_correct) / count * 100
    print(f"  Accuracy gap:               {gap:.2f}pp")
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
    model = MLP(dropout_p=0.2).to(DEVICE)
    best_path = train(model, train_loader, test_loader)

    # 3. Load best checkpoint
    print(f"\n[3/5] Loading best checkpoint: {os.path.relpath(best_path, BASE_DIR)}")
    model.load_state_dict(torch.load(best_path, map_location="cpu"))
    model.eval()

    # 4. Quantize + calibrate activation scales
    print("\n[4/5] Quantizing weights/biases and calibrating activation scales...")
    quant = quantize_model(model, train_loader)

    # 5. Export
    print("\n[5/6] Exporting .coe files...")
    export_coe(quant)

    print("\n[6/6] Generating firmware/weights_biases.h...")
    export_header(quant)

    # Bonus: verify fixed-point simulation matches float
    verify_quantized(model, quant, test_loader)

    print("\nDone. Files ready for Vivado BRAM initialization and PS/RTL integration:")
    for f in sorted(os.listdir(COE_DIR)):
        path = os.path.join(COE_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f:<30} {size:>7,} bytes")


if __name__ == "__main__":
    main()
