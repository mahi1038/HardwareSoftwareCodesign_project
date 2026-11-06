#include "features.h"
#include <math.h>

void compute_features(const int16_t (*win)[3], int combo, float *out) {
    const int span = (combo >= 2) ? 64 : 128;
    const int step = (combo & 1) ? 2 : 1;   // r25 = every 2nd sample, start at index 0
    const int N = span / step;

    int64_t S[3] = {0, 0, 0}, Q[3] = {0, 0, 0}, QM = 0;
    int32_t mn[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
    float m[128];
    float msum = 0.0f, mmin = 0.0f, mmax = 0.0f;

    // pass 1: sums, min/max, magnitude
    for (int k = 0; k < N; k++) {
        const int16_t *p = win[k * step];
        int64_t s2 = 0;
        for (int a = 0; a < 3; a++) {
            int32_t v = p[a];
            S[a] += v;
            int64_t sq = (int64_t)v * v;
            Q[a] += sq;
            s2 += sq;
            if (k == 0 || v < mn[a]) mn[a] = v;
            if (k == 0 || v > mx[a]) mx[a] = v;
        }
        QM += s2;
        m[k] = sqrtf((float)s2);
        msum += m[k];
        if (k == 0 || m[k] < mmin) mmin = m[k];
        if (k == 0 || m[k] > mmax) mmax = m[k];
    }

    // x, y, z
    for (int a = 0; a < 3; a++) {
        float *o = out + 6 * a;
        int64_t num = (int64_t)N * Q[a] - S[a] * S[a];
        o[0] = (float)S[a] / (float)N;
        o[1] = sqrtf((float)num / (float)((int64_t)N * N));
        o[2] = (float)mn[a];
        o[3] = (float)mx[a];
        o[4] = (float)Q[a] / (float)N;

        int zc = 0;
        int64_t dprev = 0;
        for (int k = 0; k < N; k++) {
            int64_t d = (int64_t)N * win[k * step][a] - S[a];
            if (k > 0 && dprev * d < 0) zc++;   // strict sign change around the mean
            dprev = d;
        }
        o[5] = (float)zc;
    }

    // magnitude
    float *o = out + 18;
    float mean = msum / (float)N;
    float acc = 0.0f;
    for (int k = 0; k < N; k++) {
        float d = m[k] - mean;
        acc += d * d;
    }
    o[0] = mean;
    o[1] = sqrtf(acc / (float)N);
    o[2] = mmin;
    o[3] = mmax;
    o[4] = (float)QM / (float)N;

    int zc = 0;
    float dprev = 0.0f;
    for (int k = 0; k < N; k++) {
        float d = m[k] - mean;
        if (k > 0 && ((dprev > 0.0f && d < 0.0f) || (dprev < 0.0f && d > 0.0f))) zc++;
        dprev = d;
    }
    o[5] = (float)zc;
}