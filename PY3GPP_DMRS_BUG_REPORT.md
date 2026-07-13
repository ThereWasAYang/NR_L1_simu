# py3gpp 0.6.0 known DMRS and OFDM issues

This document consolidates three independently reproduced upstream issues. They are not defects in `NR_L1_simu`; the local implementation follows 3GPP TS 38.211 V18.8.0 and deliberately limits py3gpp waveform interoperability checks to cases where py3gpp is a valid reference.

Verified against:

- installed package: py3gpp 0.6.0;
- upstream repository: `catkira/py3gpp` main at `a4cc7cff1f3f6c780bdeea6937818f6d24179eea` (2026-07-03).

## 1. PDSCH mapping-A adds DMRS at symbol 7 when addPos=0

`PDSCHDMRSSyms` unconditionally appends symbol 7 for allocation lengths 8 and 9:

```python
if sym_alloc in [8, 9]:
    occupied_syms = np.append(occupied_syms, 7)
```

For single-symbol PDSCH DMRS mapping type A, TS 38.211 Table 7.4.1.1.2-3 requires only the front-loaded `l0` when `DMRSAdditionalPosition=0`.

Minimal correction:

```diff
-if sym_alloc in [8, 9]:
-    occupied_syms = np.append(occupied_syms, 7)
+if sym_alloc in [8, 9] and add_pos >= 1:
+    occupied_syms = np.append(occupied_syms, 7)
```

## 2. PDSCH mapping-A `ld=12` uses the wrong table row

For `DMRSTypeAPosition=2`, py3gpp 0.6.0/current main returns:

| Additional position | Actual | TS 38.211 Table 7.4.1.1.2-3 |
|---|---|---|
| pos1 | `[2, 11]` | `[2, 9]` |
| pos2 | `[2, 7, 11]` | `[2, 6, 9]` |
| pos3 | `[2, 5, 8, 11]` | `[2, 5, 8, 11]` |

The `sym_alloc == 12` branch should be:

```diff
 if add_pos == 1:
-    occupied_syms = np.append(occupied_syms, 11)
+    occupied_syms = np.append(occupied_syms, 9)
 elif add_pos == 2:
-    occupied_syms = np.append(occupied_syms, [7, 11])
+    occupied_syms = np.append(occupied_syms, [6, 9])
```

Reproduction:

```python
import numpy as np
from py3gpp.nrPDSCHDMRS import PDSCHDMRSSyms
from py3gpp.configs.nrPDSCHConfig import nrPDSCHConfig

cfg = nrPDSCHConfig()
cfg.SymbolAllocation = [0, 12]
cfg.DMRS.DMRSTypeAPosition = 2
cfg.DMRS.DMRSLength = 1
for add_pos in (1, 2, 3):
    cfg.DMRS.DMRSAdditionalPosition = add_pos
    print(add_pos, sorted(int(x) for x in np.unique(PDSCHDMRSSyms(cfg))))
```

## 3. `nrOFDMModulate(initialNSlot != 0)` treats a slot as a symbol offset

The public argument is named `initialNSlot`, but the implementation uses it directly as a symbol offset:

```python
for i in range(initialNSlot):
    sample_pos_in_slot += Nfft + N_cp[i]

sym_pos_in_slot = (sym_pos_in_grid + initialNSlot) % carrier.SymbolsPerSlot
```

For a one-slot grid and `initialNSlot=1`, the first grid column is therefore modulated as OFDM symbol 1 instead of symbol 0. CP selection and clause 5.4 phase timing are both shifted. Values at or above `SymbolsPerSlot` can also index beyond `N_cp` in the pre-loop.

Independent measurement with 30 kHz SCS, 1024-point FFT, 3.5 GHz carrier and a fixed random grid:

| Slot | Maximum absolute difference from the independent 38.211 implementation |
|---|---|
| 0 | `8.9e-11` |
| 1 | `1.77e-1` |

The fix needs to keep two concepts separate:

1. `initialNSlot` selects the absolute slot/subframe timing and that slot's CP pattern;
2. grid column index selects the OFDM symbol within each generated slot.

For normal CP at μ≥2, the long-prefix locations must also be derived from absolute OFDM symbol positions 0 and `7 * 2**μ` in each subframe; multiplying every slot by one constant slot length is not sufficient.

## Upstream submission checklist

- Search current issues and pull requests for duplicates.
- Submit the two PDSCH table defects together because they affect the same function and table.
- Submit the OFDM slot/timing defect separately because it requires broader tests and a different implementation path.
- Add public-API regression tests for PDSCH lengths 8/9/12 and OFDM slots 0 through at least one full subframe.
