// =============================================================================
// axilite_ctrl.v
//
// AXI4-Lite slave — control/status interface for mlp_engine
// Register map (word-addressed, 4-byte aligned):
//
//   Offset 0x00  CTRL   [0]   start  — write 1 to begin inference (auto-clears)
//   Offset 0x04  STATUS [1]   busy   — 1 while engine is running
//                       [0]   done   — 1 after inference completes (read-to-clear)
//   Offset 0x08  RESULT [3:0] class  — predicted digit 0–9 (valid when done=1)
//
// This slave keeps the interface intentionally simple:
//   - writes complete only when AWVALID and WVALID are both high in the same
//     cycle, avoiding stale-address hazards
//   - reads are registered
// done_latch is set on the engine's one-cycle done pulse and cleared when the
// PS reads the STATUS register (read-to-clear semantics).
// =============================================================================
`timescale 1ns / 1ps

module axilite_ctrl (
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,

    // Write address channel
    input  wire [3:0]  s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,

    // Write data channel
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,

    // Write response channel
    output reg  [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,

    // Read address channel
    input  wire [3:0]  s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,

    // Read data channel
    output reg  [31:0] s_axi_rdata,
    output reg  [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready,

    // Engine interface
    output reg         start,    // one-cycle pulse to mlp_engine
    input  wire        busy,     // engine busy flag (from mlp_top SR latch)
    input  wire        done,     // one-cycle done pulse from mlp_engine
    input  wire [3:0]  result    // predicted class from mlp_engine
);

// ── Internal registers ────────────────────────────────────────────────────────
reg [3:0] ar_addr_lat;  // latched read address
reg       done_latch;   // sticky done flag (read-to-clear)

// Detect when PS is reading the STATUS register (offset 0x04 → addr[3:2]=01)
wire status_read = s_axi_rvalid && s_axi_rready && (ar_addr_lat[3:2] == 2'b01);

// =============================================================================
// Write path
// =============================================================================
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        s_axi_awready <= 1'b0;
        s_axi_wready  <= 1'b0;
        s_axi_bvalid  <= 1'b0;
        s_axi_bresp   <= 2'b00;
        start         <= 1'b0;
    end else begin
        start <= 1'b0;  // auto-clear each cycle

        // Accept write address/data together. This is AXI-Lite compliant and
        // avoids decoding WDATA against an old latched AWADDR.
        if (!s_axi_bvalid && s_axi_awvalid && s_axi_wvalid) begin
            s_axi_awready <= 1'b1;
            s_axi_wready  <= 1'b1;
            // CTRL register (offset 0x00 → addr[3:2] == 2'b00)
            if (s_axi_awaddr[3:2] == 2'b00 && s_axi_wdata[0])
                start <= 1'b1;
            // STATUS and RESULT registers are read-only; writes ignored
        end else begin
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
        end

        // Send write response after a successful AW/W handshake.
        if (!s_axi_bvalid && s_axi_awvalid && s_axi_wvalid) begin
            s_axi_bvalid <= 1'b1;
            s_axi_bresp  <= 2'b00;  // OKAY
        end else if (s_axi_bvalid && s_axi_bready) begin
            s_axi_bvalid <= 1'b0;
        end
    end
end

// =============================================================================
// done_latch — set on done pulse, clear on STATUS read
// Single always block to avoid multi-driver issues.
// =============================================================================
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn)
        done_latch <= 1'b0;
    else if (done)
        done_latch <= 1'b1;
    else if (status_read)
        done_latch <= 1'b0;
end

// =============================================================================
// Read path
// =============================================================================
always @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
    if (!s_axi_aresetn) begin
        s_axi_arready <= 1'b0;
        s_axi_rvalid  <= 1'b0;
        s_axi_rdata   <= 32'h0;
        s_axi_rresp   <= 2'b00;
        ar_addr_lat   <= 4'h0;
    end else begin
        // Accept read address
        if (s_axi_arvalid && !s_axi_arready) begin
            s_axi_arready <= 1'b1;
            ar_addr_lat   <= s_axi_araddr;
        end else begin
            s_axi_arready <= 1'b0;
        end

        // Drive read data
        if (s_axi_arready && s_axi_arvalid && !s_axi_rvalid) begin
            s_axi_rvalid <= 1'b1;
            s_axi_rresp  <= 2'b00;  // OKAY
            case (s_axi_araddr[3:2])
                2'b00:   s_axi_rdata <= 32'h0;                          // CTRL (write-only view)
                2'b01:   s_axi_rdata <= {30'h0, busy, done_latch};       // STATUS
                2'b10:   s_axi_rdata <= {28'h0, result};                 // RESULT
                default: s_axi_rdata <= 32'hDEAD_BEEF;
            endcase
        end else if (s_axi_rvalid && s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end
    end
end

endmodule
