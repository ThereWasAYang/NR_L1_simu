# PR Title

`Fix nrLDPCEncode(algo="sionna") for valid BG2/K=640 inputs`

# Commit Message

`Fix nrLDPCEncode sionna path for BG2 K=640 case`

# PR Description

## Summary

This PR fixes a bug in `nrLDPCEncode(..., algo="sionna")` for valid BG2 low-rate inputs such as:

- `tbs = 552`
- `R = 120 / 1024`
- `BGN = 2`
- `K = 640`
- `Zc = 64`

Previously, this case failed with a matrix dimension mismatch during encoding.

## Root cause

In the current implementation, `k_b` is used for two different purposes:

1. selecting the lifting-size set (`i_ls`)
2. selecting the number of systematic base-graph columns passed into `_gen_submat(...)`

For BG2 and `K=640`, the code correctly selects:

- `k_b = 9`

but this value is only intended for lifting-size selection.

When constructing the encoding submatrices, `_gen_submat(...)` should still use the base-graph systematic width:

- `nsys = 10` for BG2

Using `k_b=9` here causes the base graph to be sliced incorrectly, which leads to a dimension mismatch later in `_encode(...)`.

## Fix

Keep `k_b` for lifting-size selection, but use `nsys` when calling `_gen_submat(...)`.

### Before

```python
pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(bm, k_b, Zc, bgn)
```

### After

```python
pcm_a, pcm_b_inv, pcm_c1, pcm_c2 = _gen_submat(bm, nsys, Zc, bgn)
```

I also removed the unused `pcm = _lift_basegraph(bm, Zc)` line from this branch.

## Regression test

This PR adds a regression test covering a valid BG2/K=640 case:

- `tbs = 552`
- `R = 120 / 1024`

The test verifies that `nrLDPCEncode(..., algo="sionna")` no longer crashes and returns an output with the expected shape.

## Observed error before this fix

For the valid input above, the encoder failed with:

```text
ValueError: matmul: dimension mismatch with signature (n,k=576),(k=640,1?)->(n,1?)
```

## Why this is correct

For BG2, the reduced `k_b` values (`10/9/8/6`) are part of the lifting-size selection procedure in 38.212. They do not change the actual number of systematic base-graph columns, which remains fixed by the base graph itself.

So:

- `k_b` should affect `i_ls` selection
- `nsys` should affect `_gen_submat(...)` slicing
