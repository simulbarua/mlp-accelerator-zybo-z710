#include <stdint.h>
#include "xil_io.h"
#include "xil_printf.h"
#include "mlp_bram_init.h"
#include "weights_biases.h"   /* provides MLP_INPUT_MEAN, MLP_INPUT_STD, MLP_INPUT_SCALE */

/* AXI-Lite register addresses — must match Vivado Address Editor */
#define MLP_AXILITE_BASE  0x40020000UL
#define MLP_CTRL_ADDR    (MLP_AXILITE_BASE + 0x00UL)
#define MLP_STATUS_ADDR  (MLP_AXILITE_BASE + 0x04UL)
#define MLP_RESULT_ADDR  (MLP_AXILITE_BASE + 0x08UL)

/* Input BRAM base address (AXI BRAM Controller 1) */
#define INPUT_BRAM_BASE  MLP_INPUT_BRAM_BASE_ADDR

/* BRAM sizes — must match the Address Editor ranges in Vivado.
 * Param BRAM : 0x40000000, 64 KB  → 16 384 × 32-bit words
 * Input BRAM : 0x40010000,  4 KB  →  1 024 × 32-bit words
 */
#define PARAM_BRAM_WORDS  (0x10000U >> 2)
#define INPUT_BRAM_WORDS  (0x1000U  >> 2)

/* Clamp x to [lo, hi] */
static inline int clamp_int(int x, int lo, int hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

/*
 * zero_bram()
 * Write 0x00000000 to every 32-bit word of a BRAM over the AXI bus.
 *
 * A priming write is issued first because the Xilinx AXI BRAM Controller
 * v4.1 silently drops the very first write transaction after reset.  The
 * DSB stalls the CPU until that priming write receives BVALID, guaranteeing
 * the controller is fully ready before the sweep begins.  Word 0 is then
 * written again inside the loop, so the entire BRAM including word 0 is
 * guaranteed to be zero on return.
 */
static void zero_bram(uintptr_t base, uint32_t n_words)
{
    uint32_t i;
    Xil_Out32(base, 0U);                      /* absorb potential first-write drop */
    __asm volatile("dsb" ::: "memory");
    for (i = 0; i < n_words; i++) {
        Xil_Out32(base + (i << 2), 0U);
    }
    __asm volatile("dsb" ::: "memory");
}

/*
 * initialize_device()
 * Zero both BRAMs before loading weights.
 *
 * Guarantees a known-clean state so that any mismatch in
 * mlp_bram_readback_check() is provably a write failure in the current
 * session, not a false pass from stale data left by a previous run.
 * The zero sweep also primes both AXI BRAM controllers so that
 * mlp_bram_init() does not encounter the first-write-dropped issue.
 */
static void initialize_device(void)
{
    xil_printf("initialize_device: zeroing Param BRAM...\r\n");
    zero_bram(MLP_PARAM_BRAM_BASE_ADDR, PARAM_BRAM_WORDS);

    xil_printf("initialize_device: zeroing Input BRAM...\r\n");
    zero_bram(MLP_INPUT_BRAM_BASE_ADDR, INPUT_BRAM_WORDS);

    xil_printf("initialize_device: done.\r\n");
}

/*
 * cleanup_device()
 * Zero both BRAMs before exiting.
 *
 * BRAM content is non-volatile across bitstream reloads when
 * Load_Init_File = false.  Clearing here ensures the next session
 * starts from all-zeros, so stale weights or pixel data cannot mask
 * write failures during readback verification.
 */
static void cleanup_device(void)
{
    xil_printf("cleanup_device: clearing Param BRAM...\r\n");
    zero_bram(MLP_PARAM_BRAM_BASE_ADDR, PARAM_BRAM_WORDS);

    xil_printf("cleanup_device: clearing Input BRAM...\r\n");
    zero_bram(MLP_INPUT_BRAM_BASE_ADDR, INPUT_BRAM_WORDS);

    xil_printf("cleanup_device: done.\r\n");
}

/*
 * normalize_pixels()
 * Convert 784 raw uint8 pixels to INT8 quantized values.
 *   1. Normalize:  x_f = (pixel/255.0 - MEAN) / STD
 *   2. Quantize:   x_i = round(x_f / INPUT_SCALE), clipped to [-128, 127]
 *   3. Write as two's-complement bytes to the Input BRAM.
 *
 * The RTL sign-extends each byte when reading Port B, so negative values
 * are handled correctly in hardware.
 */
static void normalize_pixels(const uint8_t *pixels, uint32_t n)
{
    volatile uint8_t *bram = (volatile uint8_t *)INPUT_BRAM_BASE;
    for (uint32_t i = 0; i < n; i++) {
        float x_f = ((float)pixels[i] / 255.0f - MLP_INPUT_MEAN) / MLP_INPUT_STD;
        int   x_i = mlp_roundf(x_f / MLP_INPUT_SCALE);
        bram[i]   = (uint8_t)(int8_t)clamp_int(x_i, -128, 127);
    }
}

int main(void) {

    /* ── Step 1: Zero both BRAMs — eliminates stale data from previous sessions */
    initialize_device();

    /* ── BRAM address-0 diagnostic ───────────────────────────────────────────
     * Determine whether writes to BRAM word 0 (0x40000000) ever commit.
     * Reports results before mlp_bram_init() so its priming write is not
     * a confounding factor.  Remove this block once the issue is resolved. */
    {
        uint32_t v;

        v = Xil_In32(MLP_PARAM_BRAM_BASE_ADDR + 0);
        xil_printf("[DIAG] After zero_bram: BRAM[0x0000] = 0x%08X (expect 0x00000000)\r\n", v);

        v = Xil_In32(MLP_PARAM_BRAM_BASE_ADDR + 4);
        xil_printf("[DIAG] After zero_bram: BRAM[0x0004] = 0x%08X (expect 0x00000000)\r\n", v);

        Xil_Out32(MLP_PARAM_BRAM_BASE_ADDR + 0, 0xDEADBEEFUL);
        Xil_Out32(MLP_PARAM_BRAM_BASE_ADDR + 4, 0xCAFEBABEUL);
        __asm volatile("dsb" ::: "memory");

        v = Xil_In32(MLP_PARAM_BRAM_BASE_ADDR + 0);
        xil_printf("[DIAG] After direct write: BRAM[0x0000] = 0x%08X (expect 0xDEADBEEF)\r\n", v);

        v = Xil_In32(MLP_PARAM_BRAM_BASE_ADDR + 4);
        xil_printf("[DIAG] After direct write: BRAM[0x0004] = 0x%08X (expect 0xCAFEBABE)\r\n", v);
    }
    /* ── end diagnostic ───────────────────────────────────────────────────── */

    /* ── Step 2: Load weights/biases into Param BRAM ─────────────────────── */
    xil_printf("Loading MLP parameters into BRAM...\r\n");
    mlp_bram_init();
    if (mlp_bram_readback_check() != 0) {
        xil_printf("ERROR: BRAM readback failed. Check MLP_BRAM_BASE_ADDR.\r\n");
        cleanup_device();
        return -1;
    }
    xil_printf("BRAM loaded OK.\r\n");

    /* ── Step 3: Normalize and write 784 pixels to Input BRAM ────────────── */
    extern uint8_t my_image[784];   /* raw uint8 pixels, 0–255 */
    normalize_pixels(my_image, 784);

    /* ── Step 4: Trigger inference ────────────────────────────────────────── */
    Xil_Out32(MLP_CTRL_ADDR, 0x00000001UL);

    /* ── Step 5: Poll STATUS[0] (done flag, read-to-clear) ───────────────── */
    while ((Xil_In32(MLP_STATUS_ADDR) & 0x1U) == 0U)
        ;

    /* ── Step 6: Read predicted class ────────────────────────────────────── */
    uint32_t cls = Xil_In32(MLP_RESULT_ADDR) & 0xFU;
    xil_printf("Predicted class: %d\r\n", (int)cls);

    /* ── Step 7: Clear BRAMs so next session starts from a known-zero state ─ */
    cleanup_device();
    return 0;
}
