# Feature spec v1 (contract between Python reference and C)

## 1. Input data
- Source: UCI HAR `total_acc` x/y/z, 128 samples @ 50 Hz, units g.
- Quantization (Python only, done once): q = clip(floor(g*1000 + 0.5), -32768, 32767) -> int16 (unit: mg).
- The board stores only these int16 base windows (128 x 3). **Both sides compute features from the int16 values.**
  C never converts from g.

## 2. Window and rate (derived in C from the 128-sample base window)
| Window | Rate | Samples used | N |
|---|---|---|---|
| w128 | r50 | indices 0..127 | 128 |
| w128 | r25 | indices 0,2,4..126 | 64 |
| w64  | r50 | indices 0..63 | 64 |
| w64  | r25 | indices 0,2,4..62 | 32 |
"Window" = time span (w128 = 2.56 s, w64 = first 1.28 s). r25 = every 2nd sample, no filter.

## 3. Signals
x, y, z (int16 values, as int32/int64 in sums), and magnitude
m[i] = sqrtf((float)(x*x + y*y + z*z)). The sum is at most 1.5e7 < 2^24, so it is exact in float32,
and IEEE sqrtf is correctly rounded, so m is identical in Python (float32) and C.

## 4. Features: 6 per signal, 24 total
Order is fixed: for sig in [x, y, z, mag]: [mean, std, min, max, energy, zc]
-> index = 6*sig_idx + feat_idx (x:0-5, y:6-11, z:12-17, mag:18-23).

For x, y, z (integer-exact; sums in int64, N = sample count):
- S = sum(x), Q = sum(x*x)
- mean   = S / N
- std    = sqrt( (N*Q - S*S) / (N*N) )      population std (ddof=0), numerator is an exact int64
- min, max = integer min/max
- energy = Q / N                            mean of squares, in mg^2
- zc     = count of i in [0, N-2] where d[i]*d[i+1] < 0, with d[i] = N*x[i] - S (exact int64)
           i.e. strict sign change around the window mean; a sample equal to the mean never counts.

For mag (float32 values m[i]):
- mean, std (ddof=0), min, max as above; energy = mean(m*m) = mean of (x*x+y*y+z*z)
- zc = strict sign changes of (m[i] - mean)

## 5. Output types and tolerance
- min, max, zc: int, must match exactly (**exception: mag zc may differ by +-1 due to float rounding; allowed**).
- mean, std, energy, mag stats: float32, relative error <= 1e-4 (abs <= 1e-3 near zero).
- Features stay in mg units (no rescaling). Scaling for M2/M3 (mean/std per feature) is exported as constants.

## 6. Models and precision
- fp32: all features and weights float32.
- int8 (M2/M3 only): int8 weights, features quantized with per-feature scale from training; scales exported.
  Details are defined in the export step. int8 saves flash only (no MCU speedup).

## 7. Test windows on the board
- 150 windows from the 9 held-out test subjects, stratified by class, fixed seed (42), stored as int16 [150][128][3].
- window_idx indexes this list; labels are in labels.h.

## 8. Golden vectors
`golden/golden_v1.json`: 20 windows (from the 150) with int16 data and expected 24 features for
each of the 4 window/rate combinations. A's C code must match all of them within section 5.