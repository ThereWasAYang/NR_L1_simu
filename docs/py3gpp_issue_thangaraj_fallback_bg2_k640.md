# Title

`nrLDPCEncode(algo="thangaraj") is not a reliable fallback for the valid BG2 K=640 low-rate case`

# Body

Hi, I am opening this as a separate issue from the `algo="sionna"` crash, because this one is about correctness rather than an exception.

## Environment

- py3gpp: 0.6.0
- Python: 3.12
- Platform: macOS

## Context

For the following valid case:

- `tbs = 552`
- `R = 120 / 1024`
- `nrDLSCHInfo(...) -> BGN = 2, K = 640, Zc = 64, N = 3200`

the default `nrLDPCEncode(..., algo="sionna")` crashes with a dimension mismatch.

As a workaround, I tried `algo="thangaraj"`.

## Problem

`algo="thangaraj"` does return an encoded codeword, but it does not behave like a reliable drop-in fallback in this case.

I tested the following full chain:

1. `nrLDPCEncode(..., algo="thangaraj")`
2. `nrRateMatchLDPC(...)`
3. ideal hard/soft bits
4. `nrRateRecoverLDPC(...)`
5. `nrLDPCDecode(...)`
6. `nrCodeBlockDesegmentLDPC(...)`
7. `nrCRCDecode(...)`

The final CRC still fails.

## Important observation

I also checked whether rate matching / rate recovery are scrambling the bit order.

After:

- encoding with `algo="thangaraj"`
- rate matching
- ideal soft bits
- rate recovery

the recovered hard bits match the encoded bits exactly, ignoring filler positions.

So the problem does **not** appear to be caused by:

- modulation mapping
- bit interleaving
- rate matching
- rate recovery

The inconsistency seems to be deeper:

- either `algo="thangaraj"` does not produce a codeword fully compatible with the current decoder path for this case
- or there is a decoder-side assumption that does not hold for this encoding branch

## Minimal reproduction

```python
import numpy as np
import contextlib
import io
from py3gpp import (
    nrCRCEncode,
    nrCodeBlockSegmentLDPC,
    nrDLSCHInfo,
    nrLDPCEncode,
    nrRateMatchLDPC,
    nrRateRecoverLDPC,
    nrLDPCDecode,
    nrCodeBlockDesegmentLDPC,
    nrCRCDecode,
)

tbs = 552
R = 120 / 1024
info = nrDLSCHInfo(tbs, R)

rng = np.random.default_rng(7)
tb = rng.integers(0, 2, size=tbs, dtype=np.int8)

tb_crc = nrCRCEncode(tb, info["CRC"])[:, 0].astype(np.int8)
cbs = nrCodeBlockSegmentLDPC(tb_crc, info["BGN"])

enc = nrLDPCEncode(cbs, info["BGN"], algo="thangaraj")
rm = nrRateMatchLDPC(enc, outlen=4608, rv=0, mod="QPSK", nLayers=1).astype(np.int8)

llrs = (1 - 2 * rm) * 50.0
rec = nrRateRecoverLDPC(
    llrs,
    trblklen=tbs,
    R=R,
    rv=0,
    mod="QPSK",
    nLayers=1,
)

# sanity check: recovered hard bits match encoded bits
hard = (rec[:, 0] < 0).astype(np.int8)
valid = enc[:, 0] != -1
print(
    "hard mismatches after rate recover vs encoded bits:",
    int(np.sum(hard[valid] != enc[:, 0][valid])),
)

decoded_cbs, _ = nrLDPCDecode(rec, info["BGN"], maxNumIter=25)

with contextlib.redirect_stdout(io.StringIO()):
    tbw, _ = nrCodeBlockDesegmentLDPC(decoded_cbs, info["BGN"], tbs + info["L"])

decoded, crc_error = nrCRCDecode(tbw.astype(np.int8), info["CRC"])
print("crc_error:", crc_error)
```

## Actual result

- recovered hard bits vs encoded bits: `0 mismatches`
- final CRC: fails

## Expected result

If `algo="thangaraj"` is intended to be a valid alternative encoder implementation, then this end-to-end chain should decode successfully for the same standards-valid input.

## Request

Could you please clarify whether:

1. `algo="thangaraj"` is expected to be fully equivalent to the default encoder path, or
2. this branch is only partially implemented / not guaranteed for all valid cases?

If it is expected to be equivalent, then this seems to be a correctness bug and would benefit from a regression test using:

- `BGN = 2`
- `K = 640`
- `Zc = 64`
- `outlen = 4608`
- `QPSK`
- `rv = 0`
