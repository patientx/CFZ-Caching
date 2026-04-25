# CFZ-Caching

A ComfyUI custom node pack for caching conditioning, debugging workflows, and tuning MIOpen settings.

## Installation

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/patientx/CFZ-Caching
```

---

## Nodes

### Save / Load Conditioning
Cache your CLIP text encoder output to disk and reload it in future runs - skipping re-encoding entirely.

> **Note:** Does not work with models that require VAE input in CLIP text (e.g. `qwen-image-edit`).

| Node | Description |
|---|---|
| CFZ Save Conditioning | Connect to a CLIP Text Encode node to save conditioning to disk |
| CFZ Load Conditioning | Load previously saved conditioning by name |

---

### Print Marker
Mark specific points in your workflow in the terminal. Useful for debugging execution order and measuring timing between steps.

**Features:**
- Custom messages printed to terminal
- Timer support between start/end marker pairs
- Elapsed time formatting (ms / s / m)
- Optional terminal clear before output
- MIOpen toggle

---

### CUDNN
Simple node to enable or disable MIOpen and its benchmark mode.

---

### CUDNN Advanced
Extended control over PyTorch, MIOpen, and Triton backend settings.

| Setting | Description |
|---|---|
| `cudnn_enabled` | Enable/disable MIOpen and benchmark mode |
| `pytorch_tunableop_enabled` | Enable PyTorch TunableOp |
| `pytorch_tunableop_tuning` | Enable TunableOp tuning pass |
| `miopen_debug_conv_direct` | Set `MIOPEN_DEBUG_CONV_DIRECT` |
| `miopen_debug_conv_naive_fwd` | Set `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD` |
| `miopen_find_enforce` | Set `MIOPEN_FIND_ENFORCE` (0 = disabled, 1-5) |
| `miopen_find_mode` | Set `MIOPEN_FIND_MODE` (0 = disabled, 1-5) |
| `triton_print_autotuning` | Set `TRITON_PRINT_AUTOTUNING` |
| `triton_cache_autotuning` | Set `TRITON_CACHE_AUTOTUNING` |

---

### MIOpen Nodes

| Node | Description |
|---|---|
| CFZ MIOpen Profile | Apply a preset architecture profile (RDNA2, RDNA3, etc.) |
| CFZ MIOpen Settings | Fine-grained control over MIOpen environment variables |
| CFZ MIOpen Solvers | Enable/disable individual MIOpen solvers |
| CFZ MIOpen Paths | Set MIOpen system DB, rocBLAS, and TunableOp cache paths |
| CFZ MIOpen DB Info | Inspect current MIOpen DB and cache state |
