#include "Arduino_RouterBridge.h"
#include <string.h>
#include "features.h"
#include "test_windows.h"
#include "golden_v1.h"

// Array names from B's generated headers
#define TEST_WIN  TEST_WINDOWS
#define N_TEST    N_TEST_WINDOWS
#define GOLD_WIN  GOLDEN_WINDOWS      // GUESS: confirm with grep, change if different
#define N_GOLD    GOLDEN_N

static volatile float g_sink;

// n floats -> "hhhhhhhh,hhhhhhhh,..." (raw float32 bits, no printf needed)
static String to_hex(const float *f, int n) {
    static const char H[] = "0123456789abcdef";
    String s;
    s.reserve(n * 9);
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &f[i], 4);
        for (int sh = 28; sh >= 0; sh -= 4) s += H[(u >> sh) & 0xF];
        if (i < n - 1) s += ',';
    }
    return s;
}

// code = window_idx * 4 + combo
String feat_test(int code) {
    if (code < 0) return "ERR";
    int idx = code / 4, combo = code % 4;
    if (idx >= N_TEST) return "ERR";
    float f[N_FEAT];
    compute_features(TEST_WIN[idx], combo, f);
    return to_hex(f, N_FEAT);
}

String feat_golden(int code) {
    if (code < 0) return "ERR";
    int idx = code / 4, combo = code % 4;
    if (idx >= N_GOLD) return "ERR";
    float f[N_FEAT];
    compute_features(GOLD_WIN[idx], combo, f);
    return to_hex(f, N_FEAT);
}

// total microseconds for 10 repeats of feature extraction on test window 0
int feat_time_us(int combo) {
    float f[N_FEAT];
    unsigned long t0 = micros();
    for (int r = 0; r < 10; r++) {
        compute_features(TEST_WIN[0], combo, f);
        g_sink = f[0];
    }
    return (int)(micros() - t0);
}

void setup() {
    Bridge.begin();
    Bridge.provide("feat_test", feat_test);
    Bridge.provide("feat_golden", feat_golden);
    Bridge.provide("feat_time_us", feat_time_us);
}

void loop() {}