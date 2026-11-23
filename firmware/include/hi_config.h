#ifndef HI_CONFIG_H
#define HI_CONFIG_H

/* Generated from config.yaml locked parameters (CWRU 3200 Hz default; IMS 20480 Hz with -DHI_CONFIG_IMS). */
#define HI_WINDOW_SIZE 1024u
#define HI_HOP_SIZE 512u
#define HI_EMA_ALPHA 0.1f

#define HI_SIGMA_FLOOR 1.0e-4f
#define HI_SLOPE_WINDOW 12u
#define HI_CONSEC_ALARMS 3u
#define HI_MACHINE_OFF_RMS_RATIO 0.10f

typedef struct {
    float low_hz;
    float high_hz;
} hi_band_t;

#ifdef HI_CONFIG_IMS

#define HI_SAMPLE_RATE_HZ 20480.0f
#define HI_BAND_COUNT 5u
#define HI_SNAPSHOT_SAMPLES 20480u
#define HI_SNAPSHOT_HOP 512u
#define HI_SNAPSHOT_WINDOWS 39u

static const hi_band_t HI_BANDS[HI_BAND_COUNT] = {
    {0.0f, 500.0f},
    {500.0f, 1500.0f},
    {1500.0f, 3000.0f},
    {3000.0f, 6000.0f},
    {6000.0f, 10240.0f}
};

#else

#define HI_SAMPLE_RATE_HZ 3200.0f
#define HI_BAND_COUNT 4u

static const hi_band_t HI_BANDS[HI_BAND_COUNT] = {
    {0.0f, 400.0f},
    {400.0f, 800.0f},
    {800.0f, 1200.0f},
    {1200.0f, 1600.0f}
};

#endif

#endif
