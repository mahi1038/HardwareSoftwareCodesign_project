#pragma once
#include <stdint.h>

#define N_FEAT 24

// combo: 0 = w128/r50, 1 = w128/r25, 2 = w64/r50, 3 = w64/r25
// win points to a 128x3 int16 base window (x,y,z interleaved)
// out: 24 floats, index = 6*sig + feat, sig = x,y,z,mag
//      feat = mean, std, min, max, energy, zc
void compute_features(const int16_t (*win)[3], int combo, float *out);