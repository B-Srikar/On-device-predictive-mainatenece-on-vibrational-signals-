#include "../include/hi_core.h"
#include "../include/hi_baseline.h"  // Generate from real normal CWRU windows first.

static float samples[HI_WINDOW_SIZE], fft_real[HI_WINDOW_SIZE], fft_imag[HI_WINDOW_SIZE];
static unsigned int sample_count = 0; static int ema_initialized = 0; static float previous_ema = 0.0f;

/* Replace with the selected accelerometer driver's sample read, in g. */
static float read_accelerometer_g(void) { return 0.0f; }

void setup() { Serial.begin(115200); }
void loop() {
  samples[sample_count++] = read_accelerometer_g();
  if (sample_count < HI_WINDOW_SIZE) return;
  hi_features_t features;
  if (hi_compute_features(samples, fft_real, fft_imag, &features) == 0) {
    float raw_hi = hi_zscore_mean(&features, HI_BASELINE_MEAN, HI_BASELINE_STD);
    previous_ema = hi_ema(raw_hi, previous_ema, &ema_initialized);
    Serial.printf("HI raw=%.5f ema=%.5f\n", raw_hi, previous_ema);
  }
  sample_count = 0;  // Non-overlapping windows; add a static overlap buffer if required.
}
