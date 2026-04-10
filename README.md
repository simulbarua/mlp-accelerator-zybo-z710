# FPGA-Based MLP Accelerator — Zybo Z7-10

CMPE 240 Sec 01 · Advanced Computer Design · Group 1

An end-to-end implementation of a Multi-Layer Perceptron (MLP) hardware accelerator for handwritten digit recognition on the **Digilent Zybo Z7-10** (Zynq-7010 SoC).

---

## Architecture

```
Input (784)  →  FC1 (64, ReLU)  →  FC2 (32, ReLU)  →  FC3 (10, argmax)
```

- **Dataset:** MNIST (60k train / 10k test)
- **Quantization:** Symmetric 8-bit post-training quantization (per-layer)
- **Target:** Zynq-7010 PL fabric, controlled via PS over AXI

---

## Repository Structure

```
mlp_accelerator_zybo_z710/
├── proposal/               # Project proposal (LaTeX source + PDF)
├── software/               # Phase 1 — Model training & weight export
│   ├── train_and_export.py # Train MLP, quantize, and export .coe files
│   └── requirements.txt    # Python dependencies
└── hardware/               # Phase 2 — RTL accelerator (Vivado project)
    └── (in progress)
```

---

## Software (Phase 1)

### Setup

```bash
cd software
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Train and Export

```bash
python train_and_export.py
```

This will:
1. Download MNIST into `software/data/`
2. Train the MLP for 20 epochs (targeting ≥ 95% test accuracy)
3. Save the best checkpoint to `software/weights/mlp_best.pth`
4. Quantize all FC layer weights/biases to INT8
5. Export Vivado-ready `.coe` files to `software/coe_files/`
6. Run a fixed-point simulation to verify quantized accuracy ≥ 94%

### Output Files

| File | Description |
|------|-------------|
| `coe_files/fc1_weights.coe` | FC1 weight BRAM init (64×784 INT8) |
| `coe_files/fc1_bias.coe` | FC1 bias BRAM init (64 INT8) |
| `coe_files/fc2_weights.coe` | FC2 weight BRAM init (32×64 INT8) |
| `coe_files/fc2_bias.coe` | FC2 bias BRAM init (32 INT8) |
| `coe_files/fc3_weights.coe` | FC3 weight BRAM init (10×32 INT8) |
| `coe_files/fc3_bias.coe` | FC3 bias BRAM init (10 INT8) |
| `coe_files/scales.txt` | Per-layer scale factors for PS-side dequantization |

---

## Hardware (Phase 2)

Vivado RTL design targeting the Zynq-7010 PL. Coming soon.

---

## Team

| # | Name | Student ID |
|---|------|-----------|
| 1 | Simul Barua | 018315661 |
| 2 | Sriram Sridhar | 016661411 |
| 3 | Manisha Varthamanan | 020095920 |
| 4 | Urmi Shah | 019107595 |
| 5 | Anusha Subramanian | 014122459 |
| 6 | Nkono Andrew Mwase | 014224548 |
