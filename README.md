# FPGA-Based MLP Accelerator — Zybo Z7-10

CMPE 240 Sec 01 · Advanced Computer Design · Group 1

An end-to-end implementation of a Multi-Layer Perceptron (MLP) hardware accelerator for handwritten digit recognition on the **Digilent Zybo Z7-10** (Zynq-7010 SoC).

---

## Network Architecture

```
Input (784)  →  FC1 (64, ReLU)  →  FC2 (32, ReLU)  →  FC3 (10, argmax)
```

- **Dataset:** MNIST (60k train / 10k test)
- **Quantization:** Symmetric INT8 post-training quantization (per-layer)
- **Target:** Zynq-7010 PL fabric, controlled by PS Cortex-A9 over AXI
- **PL clock:** 100 MHz (FCLK_CLK0 from PS)

---

## System Architecture

```
Zynq PS (ARM Cortex-A9)
   │  M_AXI_GP0
   ▼
AXI SmartConnect (1 master → 3 slaves)
   │
   ├──► AXI BRAM Ctrl 0  ──► BMG 0 Port A (32-bit AXI write)
   │    0x4000_0000 64 KB     BMG 0 Port B (8-bit) ──► mlp_engine param port
   │                          [Param BRAM: INT8 weights + INT32 biases]
   │
   ├──► AXI BRAM Ctrl 1  ──► BMG 1 Port A (32-bit AXI write)
   │    0x4001_0000  4 KB     BMG 1 Port B (8-bit) ──► mlp_engine input port
   │                          [Input BRAM: 784 INT8 pixel bytes]
   │
   └──► mlp_top AXI-Lite  ──► axilite_ctrl  (CTRL / STATUS / RESULT regs)
        0x4002_0000  4 KB     mlp_engine     (serial MAC FSM, INT32 accum)
```

### Inference flow

1. PS writes INT8 weights + INT32 biases to Param BRAM via `mlp_bram_init()`
2. PS normalizes and writes 784 INT8 pixels to Input BRAM
3. PS writes `0x1` to CTRL register to start inference
4. PS polls STATUS register until done bit is set
5. PS reads 4-bit predicted class from RESULT register

---

## Repository Structure

```
mlp_accelerator_zybo_z710/
├── hardware/
│   ├── rtl/                        # Canonical RTL (single source of truth)
│   │   ├── mlp_top.v               # Top-level: AXI-Lite + BRAM port wrapper
│   │   ├── mlp_engine.v            # Serial MAC inference FSM
│   │   ├── axilite_ctrl.v          # AXI4-Lite slave (ctrl/status/result regs)
│   │   └── mlp_params.vh           # Network constants (auto-generated)
│   ├── firmware/                   # Canonical C application source
│   │   ├── main.c                  # Top-level: init, load, infer, cleanup
│   │   ├── mlp_bram_init.c/.h      # BRAM loader + readback check
│   │   └── weights_biases.h        # Packed weight arrays (auto-generated)
│   └── mlp_accel_sys/              # Vivado project + Vitis workspace
│       ├── mlp_accel_sys.tcl       # Vivado rebuild script  ← source this
│       ├── mlp_accel_sys.hw/       # Exported hardware (.lpr + .xsa)
│       ├── mlp_system_wrapper.xsa  # Hardware export for Vitis
│       ├── mlp_accel_sys.srcs/constrs_1/  # Board constraints XDC
│       ├── app_component/src/      # Vitis C sources (compiled by Vitis)
│       └── platform/               # Vitis platform definition
├── software/                       # Python training pipeline
│   ├── train_and_export_mlp.py     # Train → quantize → export weights_biases.h
│   ├── infer_image.py              # Run inference on a saved image
│   ├── collect_sample_images.py    # Save sample MNIST images to disk
│   ├── requirements.txt            # Python dependencies
│   └── coe_files/                  # Generated Vivado BRAM init files
├── docs/
│   └── project_proposal.pdf            # Project proposal (tracked in git)
└── constraints/
    └── zybo_z710_mlp.xdc           # Board pin/timing constraints
```

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Vivado + Vitis | 2025.2.1 | Unified installer from AMD — use the unified Vitis IDE, not classic |
| Python | 3.10 + | 3.14 used during development |
| PyTorch | 2.11+ | GPU not required |
| Digilent Zybo Z7-10 board | — | XC7Z010-1CLG400C |
| USB-UART driver | — | Needed for serial console |

---

## Setup

### 1 — Clone the repository

```bash
git clone <repo-url>
cd mlp_accelerator_zybo_z710
```

### 2 — Python environment (model training & weight export)

```bash
cd software
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the model and export the weight header used by the firmware:

```bash
python train_and_export_mlp.py
```

This will:
1. Download MNIST into `software/data/` (first run only)
2. Train the MLP for up to 20 epochs (targets ≥ 95% test accuracy)
3. Save the best checkpoint to `software/weights_and_biases/mlp_best.pth`
4. Quantize all FC layers to INT8
5. Write `hardware/firmware/weights_biases.h` — the packed C array header
6. Write `hardware/rtl/mlp_params.vh` — Verilog network-dimension constants
7. Write `.coe` files to `software/coe_files/` for optional BRAM init in Vivado

Copy the generated header into the Vitis application source before building:

```bash
cp hardware/firmware/weights_biases.h \
   hardware/mlp_accel_sys/app_component/src/weights_biases.h
```

### 3 — Install the Zybo Z7-10 board files (first time only)

The Digilent board definition is not bundled with Vivado and must be installed once before recreating the project.

1. Open Vivado 2025.2.1 (no project needed — just the welcome screen)
2. In the **Tcl Console** at the bottom, run:
   ```tcl
   xhub::refresh_catalog [xhub::get_xstores xilinx_board_store]
   xhub::install [xhub::get_xitems digilentinc.com:xilinx_board_store:zybo-z7-10:1.1]
   ```
3. **Close and restart Vivado completely** — board files are only picked up on startup

You only need to do this once per Vivado installation.

### 4 — Vivado: open the project and generate the bitstream

> Skip this step if you are using the committed `.xsa` and have not changed the hardware.

1. Open Vivado 2025.2.1 (after restarting from step 3 above)
2. **File → Open Project** → select `hardware/mlp_accel_sys/mlp_accel_sys.xpr`
3. In the **Reports** menu → **Report IP Status** → click **Upgrade All**
4. Right-click the block design in Sources → **Generate Output Products**
5. Right-click the block design → **Create HDL Wrapper** → **Let Vivado manage**
6. **Run Synthesis → Run Implementation → Generate Bitstream**
7. When complete: **File → Export → Export Hardware** (check *Include bitstream*) → overwrite `mlp_system_wrapper.xsa`

### 5 — Vitis: build and run the application

1. Open **Vitis 2025.2.1** (the unified IDE, not the classic IDE)
2. When prompted for a workspace, select the `hardware/mlp_accel_sys/` folder and click **Launch**
3. Create the platform component from scratch (required on first clone):
   - Go to **File → New Component → Platform**
   - Set **Component name** to exactly `platform`
   - Set **Component location** to `hardware/mlp_accel_sys/`
   - Under **Hardware Design**, browse to `hardware/mlp_accel_sys/mlp_system_wrapper.xsa` → **Next**
   - Leave OS as **standalone**, processor as **ps7_cortexa9_0** → **Finish**
4. In the Flow Navigator under **PLATFORM**, click **Build** — wait for it to complete
5. In the Flow Navigator under **APPLICATION**, select **app_component** → click **Build**
6. Connect the Zybo Z7-10 over USB
7. Open a serial terminal at **115200 baud, 8N1** on the board's USB-UART port
8. In the Flow Navigator click **Run** → **Launch Hardware (Single Application Debug)**

### 6 — Expected serial output

```
initialize_device: zeroing Param BRAM...
initialize_device: zeroing Input BRAM...
initialize_device: done.
Loading MLP parameters into BRAM...
mlp_bram_init: all parameter blocks written.
mlp_bram_readback_check:
  FC1 weights: PASS (0x????????)
  FC1 biases : PASS (0x????????)
  FC2 weights: PASS (0x????????)
  FC2 biases : PASS (0x????????)
  FC3 weights: PASS (0x????????)
  FC3 biases : PASS (0x????????)
  Overall: PASS
BRAM loaded OK.
Predicted class: <digit>
cleanup_device: done.
```

---

## AXI Address Map

| Peripheral | Base | Size | Purpose |
|---|---|---|---|
| AXI BRAM Ctrl 0 (Param BRAM) | `0x4000_0000` | 64 KB | Weights + biases |
| AXI BRAM Ctrl 1 (Input BRAM) | `0x4001_0000` | 4 KB | 784 input pixels |
| mlp_top AXI-Lite | `0x4002_0000` | 4 KB | CTRL / STATUS / RESULT |

### AXI-Lite register map

| Offset | Name | Access | Description |
|---|---|---|---|
| `0x00` | CTRL | W | Write `0x1` to start inference |
| `0x04` | STATUS | R | Bit 0: done flag (read-to-clear) |
| `0x08` | RESULT | R | Bits [3:0]: predicted class (0–9) |

---

## Param BRAM layout

| Block | Offset | Size | Format |
|---|---|---|---|
| FC1 weights | `0x0000` | 12 544 words | INT8, 4 per word, little-endian |
| FC1 biases | `0xC400` | 64 words | INT32, 1 per word |
| FC