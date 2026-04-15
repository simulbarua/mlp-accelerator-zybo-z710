/* mlp_bram_init.c
 *
 * Loads MLP weights and biases from the embedded C header into the Param BRAM,
 * and provides a helper to write a normalised input image to the Input BRAM.
 *
 * Weight/bias layout:
 *   FC1/FC2/FC3 weights : INT8, packed 4-per-uint32_t word (little-endian)
 *   FC1/FC2/FC3 biases  : INT32, one value per uint32_t word
 *
 * Include weights_biases.h in exactly ONE translation unit (this file).
 * All other files that need network constants should include only
 * mlp_bram_init.h, which keeps the large static arrays out of them.
 */

#include "mlp_bram_init.h"
#include "weights_biases.h"  /* defines fc1_weights[], fc1_biases[], etc. */

/* math.h / roundf not needed — mlp_roundf() is defined in mlp_bram_init.h */
#include "xil_io.h"          /* Xil_Out32, Xil_In32 */
#include "xil_printf.h"      /* xil_printf */

/* ── Internal helpers ────────────────────────────────────────────────────────
 * Write `n_words` consecutive uint32_t values to BRAM starting at
 * (base_addr + byte_offset), using 32-bit AXI writes.
 */
static void bram_write_block(uintptr_t      base_addr,
                              uint32_t       byte_offset,
                              const uint32_t *data,
                              uint32_t       n_words)
{
    uint32_t i;
    /* Wrap-boundary checkpoints: 0, 1024, 2048, 4096, 8192, 12288.
     * If the physical BRAM is smaller than the 64KB AXI window, a write at
     * index B (where B = physical_words) will alias back to BRAM address 0,
     * overwriting the value written at i=0.  Each checkpoint prints the AXI
     * address, the source data, and reads back BRAM[base+offset] after the
     * write so we can see exactly when address 0 gets clobbered. */
    static const uint32_t chk[] = {0, 1023, 1024, 2047, 2048, 4095, 4096,
                                    8191, 8192, 12287, 12288, 12543};
    uint32_t c;
    for (i = 0; i < n_words; i++) {
        int is_chk = 0;
        for (c = 0; c < sizeof(chk)/sizeof(chk[0]); c++) {
            if (i == chk[c]) { is_chk = 1; break; }
        }
        if (is_chk) {
            xil_printf("[BW] pre  i=%5u axi=0x%08X src=0x%08X\r\n",
                       i, (uint32_t)(base_addr + byte_offset + (i << 2)), data[i]);
        }
        Xil_Out32(base_addr + byte_offset + (i << 2), data[i]);
        if (is_chk) {
            __asm volatile("dsb" ::: "memory");
            xil_printf("[BW] post i=%5u BRAM[base+off]=0x%08X\r\n",
                       i, Xil_In32(base_addr + byte_offset));
        }
    }
}

/* Same as above but for signed INT32 bias arrays.
 * int32_t and uint32_t have identical bit representations; the cast is safe.
 */
static void bram_write_int32_block(uintptr_t     base_addr,
                                   uint32_t      byte_offset,
                                   const int32_t *data,
                                   uint32_t      n_words)
{
    bram_write_block(base_addr, byte_offset,
                     (const uint32_t *)data, n_words);
}

/* ── mlp_bram_init ───────────────────────────────────────────────────────────
 * Writes all six parameter blocks to the Param BRAM in order.
 *
 * All biases are INT32 (one INT32 per uint32_t word).
 * All weights are INT8 (four INT8 per uint32_t word).
 *
 * The byte offsets (FC1_WEIGHT_BRAM_OFFSET, etc.) come from weights_biases.h
 * and must match the localparam addresses in mlp_engine.v exactly.
 */
void mlp_bram_init(void)
{
    const uintptr_t base = MLP_PARAM_BRAM_BASE_ADDR;

    /* ── AXI BRAM controller priming write ───────────────────────────────────
     * The Xilinx AXI BRAM Controller (v4.1) silently drops the very first
     * AXI write transaction after the PL comes out of reset: the controller's
     * write-acceptance state machine needs one extra AXI clock after
     * s_axi_aresetn de-asserts before it is ready.  FC1 weights start at
     * BRAM offset 0x0000 — the very first write in this function — so word 0
     * was consistently reading back stale data from the previous session.
     *
     * This dummy write absorbs the dropped transaction.  The DSB stalls the
     * CPU until the BRAM controller has acknowledged with BVALID, ensuring
     * the controller is fully ready before the real data writes begin. */
    Xil_Out32(base, 0U);
    __asm volatile("dsb" ::: "memory");
    xil_printf("[D1] after priming write: BRAM[0]=0x%08X\r\n", Xil_In32(base));

    /* FC1 weights: 64×784 INT8 → 12 544 words */
    xil_printf("[D1b] fc1_weights DDR=0x%08X [0]=0x%08X words=%u\r\n",
               (uint32_t)(uintptr_t)fc1_weights, fc1_weights[0], FC1_WEIGHT_WORDS);
    /* Print wrap-boundary source values BEFORE the write so we know what
     * will land at BRAM address 0 if the physical depth is 1024/2048/4096/8192. */
    xil_printf("[WRAP-SRC] [1024]=0x%08X [2048]=0x%08X [4096]=0x%08X "
               "[8192]=0x%08X [12288]=0x%08X [12543]=0x%08X\r\n",
               fc1_weights[1024], fc1_weights[2048], fc1_weights[4096],
               fc1_weights[8192], fc1_weights[12288], fc1_weights[12543]);
    bram_write_block(base, FC1_WEIGHT_BRAM_OFFSET,
                     fc1_weights, FC1_WEIGHT_WORDS);
    __asm volatile("dsb" ::: "memory");
    /* Snapshot BRAM at the start of each 1KB/2KB/4KB/8KB/16KB sub-range so we
     * can see which physical boundary the controller actually wraps at. */
    xil_printf("[POST-FC1] BRAM @0x0000=0x%08X @0x1000=0x%08X @0x2000=0x%08X "
               "@0x4000=0x%08X @0x8000=0x%08X @0xC000=0x%08X @0xC3FC=0x%08X\r\n",
               Xil_In32(base + 0x0000), Xil_In32(base + 0x1000),
               Xil_In32(base + 0x2000), Xil_In32(base + 0x4000),
               Xil_In32(base + 0x8000), Xil_In32(base + 0xC000),
               Xil_In32(base + 0xC3FC));
    xil_printf("[D2] BRAM[0]=0x%08X expect fc1_weights[0]=0x%08X\r\n",
               Xil_In32(base + FC1_WEIGHT_BRAM_OFFSET), fc1_weights[0]);

    /* FC1 biases: 64 INT32, signed, one value per word */
    bram_write_int32_block(base, FC1_BIAS_BRAM_OFFSET,
                           fc1_bias, FC1_BIAS_WORDS);

    /* FC2 weights: 32×64 INT8 → 512 words */
    bram_write_block(base, FC2_WEIGHT_BRAM_OFFSET,
                     fc2_weights, FC2_WEIGHT_WORDS);

    /* FC2 biases: 32 INT32, signed, one value per word */
    bram_write_int32_block(base, FC2_BIAS_BRAM_OFFSET,
                           fc2_bias, FC2_BIAS_WORDS);

    /* FC3 weights: 10×32 INT8 → 80 words */
    bram_write_block(base, FC3_WEIGHT_BRAM_OFFSET,
                     fc3_weights, FC3_WEIGHT_WORDS);

    /* FC3 biases: 10 INT32, signed, one value per word */
    bram_write_int32_block(base, FC3_BIAS_BRAM_OFFSET,
                           fc3_bias, FC3_BIAS_WORDS);

    /* Data Synchronisation Barrier — mandatory before any readback.
     *
     * The Cortex-A9 AXI master buffers Device-memory writes.  Xil_Out32()
     * issues an AXI write transaction (AW + W channels) but returns before
     * the BRAM controller sends BVALID (write response).  Without a barrier,
     * a subsequent Xil_In32() can race ahead on the independent AR/R channel
     * and be delivered to the BRAM controller BEFORE the very first write has
     * committed — returning stale data.  DSB stalls the CPU until every
     * outstanding write response (BVALID) has been received, guaranteeing
     * that all BRAM words are visible to any following read. */
    __asm volatile("dsb" ::: "memory");

    xil_printf("mlp_bram_init: all parameter blocks written.\r\n");
}

/* ── mlp_bram_readback_check ─────────────────────────────────────────────────
 * Reads back the first uint32_t word of each parameter block and compares
 * against the expected value from weights_biases.h.
 * Prints PASS or FAIL for each block via xil_printf.
 */
int mlp_bram_readback_check(void)
{
    const uintptr_t base = MLP_PARAM_BRAM_BASE_ADDR;
    int pass = 1;

#define CHECK_BLOCK(name, offset, arr)                                      \
    do {                                                                    \
        uint32_t rd = Xil_In32(base + (offset));                            \
        if (rd == (arr)[0]) {                                               \
            xil_printf("  " name ": PASS (0x%08X)\r\n", rd);               \
        } else {                                                            \
            xil_printf("  " name ": FAIL (got 0x%08X, expected 0x%08X)\r\n",\
                       rd, (arr)[0]);                                       \
            pass = 0;                                                       \
        }                                                                   \
    } while (0)

    xil_printf("mlp_bram_readback_check:\r\n");
    CHECK_BLOCK("FC1 weights", FC1_WEIGHT_BRAM_OFFSET, (const uint32_t *)fc1_weights);
    CHECK_BLOCK("FC1 biases ", FC1_BIAS_BRAM_OFFSET,   (const uint32_t *)fc1_bias);
    CHECK_BLOCK("FC2 weights", FC2_WEIGHT_BRAM_OFFSET, (const uint32_t *)fc2_weights);
    CHECK_BLOCK("FC2 biases ", FC2_BIAS_BRAM_OFFSET,   (const uint32_t *)fc2_bias);
    CHECK_BLOCK("FC3 weights", FC3_WEIGHT_BRAM_OFFSET, (const uint32_t *)fc3_weights);
    CHECK_BLOCK("FC3 biases ", FC3_BIAS_BRAM_OFFSET,   (const uint32_t *)fc3_bias);

#undef CHECK_BLOCK

    xil_printf("  Overall: %s\r\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : -1;
}

/* ── mlp_write_input ─────────────────────────────────────────────────────────
 * Normalise 784 raw uint8 MNIST pixels to INT8 and write to the Input BRAM.
 *
 * Quantisation:
 *   x_float = (pixel / 255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD
 *   x_int8  = clamp(round(x_float / MLP_INPUT_SCALE), -128, 127)
 *
 * Bytes are packed little-endian 4-per-word matching the INT8 packing used
 * for weights.  The PL engine reads each byte individually via the 8-bit
 * Port B of the Input BRAM.
 *
 * pixels: pointer to 784 uint8_t values (row-major, top-left first).
 */
void mlp_write_input(const uint8_t *pixels)
{
    const uintptr_t base = MLP_INPUT_BRAM_BASE_ADDR;
    uint32_t word_idx;
    uint32_t word;
    int b;
    int8_t  q[784];
    int i;

    /* Step 1: quantise all 784 pixels to INT8 */
    for (i = 0; i < 784; i++) {
        float x = ((float)pixels[i] / 255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD;
        float xq = (float)mlp_roundf(x / MLP_INPUT_SCALE);
        if      (xq < -128.0f) xq = -128.0f;
        else if (xq >  127.0f) xq =  127.0f;
        q[i] = (int8_t)xq;
    }

    /* Step 2: pack 4 INT8 bytes per uint32_t word (little-endian) and write */
    for (word_idx = 0; word_idx < 196; word_idx++) {  /* 784/4 = 196 words */
        word = 0;
        for (b = 0; b < 4; b++) {
            word |= ((uint32_t)(uint8_t)q[word_idx * 4 + b]) << (b * 8);
        }
        Xil_Out32(base + (word_idx << 2), word);
    }
}
