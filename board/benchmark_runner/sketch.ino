#include "Arduino_RouterBridge.h"
#include <stdint.h>
#include <string.h>
#include "features.h"

// Combos: 0=w128_r50, 1=w128_r25, 2=w64_r50, 3=w64_r25 (only matters for P4 configs)
// If the board stops registering functions (image too big), set two of these to 0
// and run the other two in a second pass. The runner skips P4 configs it cannot run.
#define USE_C0 0
#define USE_C1 0
#define USE_C2 0
#define USE_C3 1

extern "C" float expf(float x) {
    if (x > 88.0f) x = 88.0f;
    if (x < -87.0f) return 0.0f;
    const float LN2 = 0.69314718056f;
    int k = (int)(x / LN2 + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)k * LN2;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f/6 + r * (1.0f/24 +
              r * (1.0f/120 + r * (1.0f/720 + r * (1.0f/5040)))))));
    union { float f; uint32_t u; } s;
    s.u = (uint32_t)(k + 127) << 23;
    return p * s.f;
}

#include "test_windows.h"

#if USE_C0
#include "model_M1_w128_r50_fp32.h"
#include "model_M2_w128_r50_fp32.h"
#include "model_M3_w128_r50_fp32.h"
#include "model_M2_w128_r50_int8.h"
#include "model_M3_w128_r50_int8.h"
#endif
#if USE_C1
#include "model_M1_w128_r25_fp32.h"
#include "model_M2_w128_r25_fp32.h"
#include "model_M3_w128_r25_fp32.h"
#include "model_M2_w128_r25_int8.h"
#include "model_M3_w128_r25_int8.h"
#endif
#if USE_C2
#include "model_M1_w64_r50_fp32.h"
#include "model_M2_w64_r50_fp32.h"
#include "model_M3_w64_r50_fp32.h"
#include "model_M2_w64_r50_int8.h"
#include "model_M3_w64_r50_int8.h"
#endif
#if USE_C3
#include "model_M1_w64_r25_fp32.h"
#include "model_M2_w64_r25_fp32.h"
#include "model_M3_w64_r25_fp32.h"
#include "model_M2_w64_r25_int8.h"
#include "model_M3_w64_r25_int8.h"
#endif

// k: 0=M1 fp32, 1=M2 fp32, 2=M3 fp32, 3=M2 int8, 4=M3 int8
#define DISPATCH(CN) \
    switch (k) { \
        case 0: return m1_##CN##_fp32_predict(f, conf); \
        case 1: return m2_##CN##_fp32_predict(f, conf); \
        case 2: return m3_##CN##_fp32_predict(f, conf); \
        case 3: return m2_##CN##_int8_predict(f, conf); \
        case 4: return m3_##CN##_int8_predict(f, conf); \
    }

static int run_model(int combo, int k, const float *f, float *conf) {
    switch (combo) {
#if USE_C0
        case 0: DISPATCH(w128_r50) break;
#endif
#if USE_C1
        case 1: DISPATCH(w128_r25) break;
#endif
#if USE_C2
        case 2: DISPATCH(w64_r50) break;
#endif
#if USE_C3
        case 3: DISPATCH(w64_r25) break;
#endif
    }
    *conf = 0;
    return -1;
}

static const char HEXD[] = "0123456789abcdef";
static void put16(String &s, uint16_t u) {
    for (int sh = 12; sh >= 0; sh -= 4) s += HEXD[(u >> sh) & 0xF];
}
static void put32(String &s, uint32_t u) {
    for (int sh = 28; sh >= 0; sh -= 4) s += HEXD[(u >> sh) & 0xF];
}

int ping_len(String s) { return s.length(); }

int combo_mask() {
    return USE_C0 | (USE_C1 << 1) | (USE_C2 << 2) | (USE_C3 << 3);
}

// P4: code = idx*100 + combo*10 + k  ->  "label,conf*10000,mcu_us"
String mcu_p4(int code) {
    int idx = code / 100, combo = (code / 10) % 10, k = code % 10;
    if (idx < 0 || idx >= N_TEST_WINDOWS || combo > 3 || k > 4) return "ERR";
    float f[N_FEAT], conf = 0;
    unsigned long t0 = micros();
    compute_features(TEST_WINDOWS[idx], combo, f);
    int label = run_model(combo, k, f, &conf);
    unsigned long dt = micros() - t0;
    if (label < 0) return "ERR";
    String s;
    s += label; s += ','; s += (int)(conf * 10000.0f + 0.5f); s += ','; s += (long)dt;
    return s;
}

// P3: code = idx*10 + combo  ->  "24 float32 hex values|mcu_us"
String mcu_feat(int code) {
    int idx = code / 10, combo = code % 10;
    if (idx < 0 || idx >= N_TEST_WINDOWS || combo > 3) return "ERR";
    float f[N_FEAT];
    unsigned long t0 = micros();
    compute_features(TEST_WINDOWS[idx], combo, f);
    unsigned long dt = micros() - t0;
    String s;
    s.reserve(N_FEAT * 9 + 12);
    for (int i = 0; i < N_FEAT; i++) {
        uint32_t u; memcpy(&u, &f[i], 4);
        put32(s, u);
        if (i < N_FEAT - 1) s += ',';
    }
    s += '|'; s += (long)dt;
    return s;
}

// P1: full 128x3 base window as int16 hex (1536 chars) + "|mcu_us"
String mcu_raw(int idx) {
    if (idx < 0 || idx >= N_TEST_WINDOWS) return "ERR";
    String s;
    s.reserve(128 * 3 * 4 + 12);
    unsigned long t0 = micros();
    for (int k = 0; k < 128; k++)
        for (int a = 0; a < 3; a++) put16(s, (uint16_t)TEST_WINDOWS[idx][k][a]);
    unsigned long dt = micros() - t0;
    s += '|'; s += (long)dt;
    return s;
}

// P2: window already cut and decimated for the combo (N x 3 int16 hex) + "|mcu_us"
// code = idx*10 + combo
String mcu_cut(int code) {
    int idx = code / 10, combo = code % 10;
    if (idx < 0 || idx >= N_TEST_WINDOWS || combo > 3) return "ERR";
    int span = (combo >= 2) ? 64 : 128;
    int step = (combo & 1) ? 2 : 1;
    String s;
    s.reserve((span / step) * 3 * 4 + 12);
    unsigned long t0 = micros();
    for (int k = 0; k < span; k += step)
        for (int a = 0; a < 3; a++) put16(s, (uint16_t)TEST_WINDOWS[idx][k][a]);
    unsigned long dt = micros() - t0;
    s += '|'; s += (long)dt;
    return s;
}

void setup() {
    Bridge.begin();
    Bridge.provide("ping_len", ping_len);
    Bridge.provide("combo_mask", combo_mask);
    Bridge.provide("mcu_p4", mcu_p4);
    Bridge.provide("mcu_feat", mcu_feat);
    Bridge.provide("mcu_raw", mcu_raw);
    Bridge.provide("mcu_cut", mcu_cut);
}

void loop() {}