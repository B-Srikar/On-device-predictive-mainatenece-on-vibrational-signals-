#include "../include/hi_core.h"
#include "../include/hi_baseline.h"

static float samples[HI_WINDOW_SIZE], fft_real[HI_WINDOW_SIZE], fft_imag[HI_WINDOW_SIZE];
static unsigned int sample_count = 0u;
static hi_monitor_t monitor_state;
static const hi_monitor_config_t monitor_cfg = {
  { HI_BASELINE_MEAN[0], HI_BASELINE_MEAN[1], HI_BASELINE_MEAN[2], HI_BASELINE_MEAN[3],
    HI_BASELINE_MEAN[4], HI_BASELINE_MEAN[5], HI_BASELINE_MEAN[6], HI_BASELINE_MEAN[7] },
  { HI_BASELINE_STD[0], HI_BASELINE_STD[1], HI_BASELINE_STD[2], HI_BASELINE_STD[3],
    HI_BASELINE_STD[4], HI_BASELINE_STD[5], HI_BASELINE_STD[6], HI_BASELINE_STD[7] },
  HI_BASELINE_MEDIAN_RMS,
  HI_MACHINE_OFF_RMS_RATIO,
  HI_EMA_ALPHA,
  HI_THRESHOLD,
  HI_BASELINE_HI_MEAN,
  HI_SLOPE_THRESHOLD
};

/* Replace with the selected accelerometer driver's sample read, in g. */
static float read_accelerometer_g(void) { return 0.0f; }

void setup() {
  Serial.begin(115200);
  hi_monitor_init(&monitor_state);
}

void loop() {
  samples[sample_count++] = read_accelerometer_g();
  if (sample_count < HI_WINDOW_SIZE) return;
  hi_features_t features;
  hi_monitor_output_t out;
  if (hi_compute_features(samples, fft_real, fft_imag, &features) == 0) {
    hi_monitor_update(&monitor_state, &monitor_cfg, &features, &out);
    Serial.printf("gated=%d raw=%.5f ema=%.5f fix=%d slp=%d\n",
                  out.gated, out.raw_hi, out.ema_hi, out.fixed_alarm, out.slope_alarm);
  }
  sample_count = 0u;  // Non-overlapping windows; add a static overlap buffer if required.
}
