# Title

`nrLDPCEncode(algo="sionna") fails for valid BG2 K=640 input because _gen_submat uses k_b instead of nsys`

# Body

Hi, thanks for maintaining py3gpp.

I found a reproducible issue in `nrLDPCEncode(..., algo="sionna")` for a valid BG2 low-rate case.

## Environment

- py3gpp: 0.6.0
- Python: 3.12
- Platform: macOS

## Minimal reproduction

```python
import numpy as np
from py3gpp import nrCRCEncode, nrCodeBlockSegmentLDPC, nrDLSCHInfo, nrLDPCEncode

tbs = 552
R = 120 / 1024

info = nrDLSCHInfo(tbs, R)
print(info)
# Expected:
# {'CRC': '16', 'L': 16, 'BGN': 2, 'C': 1, 'Lcb': 0, 'F': 72, 'Zc': 64, 'K': 640, 'N': 3200}

rng = np.random.default_rng(7)
tb = rng.integers(0, 2, size=tbs, dtype=np.int8)
tb_crc = nrCRCEncode(tb, info["CRC"])[:, 0].astype(np.int8)
cbs = nrCodeBlockSegmentLDPC(tb_crc, info["BGN"])

nrLDPCEncode(cbs, info["BGN"])
```

## Actual result

This raises:

```text
ValueError: matmul: dimension mismatch with signature (n,k=576),(k=640,1?)->(n,1?)
```

## Root cause

From reading `nrLDPCEncode.py`, it looks like `k_b` is being used for two different purposes:

1. selecting the lifting size set (`i_ls`)
2. selecting the systematic submatrix width passed into `_gen_submat(...)`

For BG2 and `K=640`, the code chooses:

- `k_b = 9` for lifting-size selection

That part seems fine.

But then the same `k_b` is passed here:

```python
pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(bm, k_b, Zc, bgn)
```

For BG2, `_gen_submat(...)` should still use the base graph systematic width (`nsys = 10`), not the reduced `k_b = 9` used only for lifting-size selection.

Because of that, `A` is sliced with width 9 instead of 10, and the encoder eventually tries to multiply a 576-wide matrix with a 640-length message vector.

## Suggested fix

Use `nsys` when calling `_gen_submat(...)`, while keeping `k_b` only for lifting-size selection.

So this:

```python
pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(bm, k_b, Zc, bgn)
```

should likely become:

```python
pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(bm, nsys, Zc, bgn)
```

## Additional note

This happens for a standards-valid case:

- BG2
- K = 640
- Zc = 64

So it would be great to add a regression test for this input.
