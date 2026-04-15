/* mlp_bram_init.h
 *
 * Declares the BRAM initialisation and readback-check functions for the
 * MLP accelerator on the Zybo Z7-10.
 *
 * BRAM layout (byte-addressed from MLP_BRAM_BASE_ADDR):
 *   FC1 weights : INT8 packed 4-per-word  (50 176 B = 12 544 × uint32_t)
 *   FC1 biases  : INT32 one-per-word      (   256 B =    64 × uint32_t)
 *   FC2 weights : INT8 packed 4-per-word  ( 2 048 B =   512 × uint32_t)
 *   FC2 biases  : INT32 one-per-word      (   128 B =    32 × uint32_t)
 *   FC3 weights : INT8 packed 4-per-word  (   320 B =    80 × uint32_t)
 *   FC3 biases  : INT32 one-per-word       (    40 B =    10 × int32_t)
 *
 * All byte offsets are defined in weights_biases.h (auto-generated).
 * MLP_BRAM_BASE_ADDR must match the address assigned to the AXI BRAM
 * Controller in Vivado's Address Editor.
 */

#ifndef MLP_BRAM_INIT_H
#define MLP_BRAM_INIT_H

#include <stdint.h>

/* ── Portable rounding without libm ─────────────────────────────────────────
 * Replaces roundf() so the firmware does not need -lm at link time.
 * Works correctly for both positive and negative values.
 */
static inline int mlp_roundf(float x)
{
    return (x >= 0.0f) ? (int)(x + 0.5f) : (int)(x - 0.5f);
}

/* ── Base address of the Param BRAM ──────────────────────────────────────────
 * Change this to match Vivado → Address Editor → your AXI BRAM Controller.
 * Typical default for GP0 slave: 0x40000000.
 */
#define MLP_PARAM_BRAM_BASE_ADDR  0x40000000UL

/* ── Base address of the Input BRAM ─────────────────────────────────────────
 * The PS writes 784 INT8 pixels here before asserting start.
 * Change to match your Input BRAM Controller address in the Address Editor.
 */
#define MLP_INPUT_BRAM_BASE_ADDR  0x40010000UL

/* ── Write all weights and biases from weights_biases.h into Param BRAM ─────
 * Call once at start-up, after PL is programmed and before the first
 * inference.  Takes ~0.4 ms at 650 MHz (PS clock).
 */
void mlp_bram_init(void);

/* ── Spot-check the first word of each of the six parameter blocks ───────────
 * Reads back and prints PASS/FAIL via xil_printf.
 * Useful for verifying BRAM wiring before running inference.
 */
int  mlp_bram_readback_check(void);   /* returns 0 on PASS, -1 on any FAIL */

/* ── Write one quantized input image into the Input BRAM ────────────────────
 * pixels : array of 784 raw uint8 pixel values (row-major, 0–255)
 * Normalises and quantises each pixel to INT8:
 *   x_float = (pixel/255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD
 *   x_int8  = clamp(roundf(x_float / MLP_INPUT_SCALE), -128, 127)
 * Packs 4 bytes per uint32_t word (little-endian) for the AXI 32-bit bus.
 */
void mlp_write_input(const uint8_t *pixels);

#endif /* MLP_BRAM_INIT_H */
