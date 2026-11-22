#ifndef HI_CONFIG_H
#define HI_CONFIG_H

/* Generated from config.yaml values on 2026-11-10; regenerate if config changes. */
#define HI_WINDOW_SIZE 1024u
#define HI_SAMPLE_RATE_HZ 3200.0f
#define HI_EMA_ALPHA 0.1f
#define HI_BAND_COUNT 4u

typedef struct { float low_hz; float high_hz; } hi_band_t;
static const hi_band_t HI_BANDS[HI_BAND_COUNT] = {
    {0.0f, 400.0f}, {400.0f, 800.0f}, {800.0f, 1200.0f}, {1200.0f, 1600.0f}
};

#endif
