#ifndef HI_BASELINE_H
#define HI_BASELINE_H

#include "hi_core.h"

/* Stage 3c primary fold @ 3200 Hz: pooled normal_0..2 baseline n=107, holdout n=44. */
static const float HI_BASELINE_MEAN[HI_FEATURE_COUNT] = {0.0396656995f, 0.247868595f, 3.28820821f, 2.85604067f, -8.01609087f, -8.20470677f, -9.1267012f, -9.94679032f};
static const float HI_BASELINE_STD[HI_FEATURE_COUNT] = {0.00261041395f, 0.0184008947f, 0.243551359f, 0.150006715f, 0.381811805f, 0.136654641f, 0.250863668f, 0.38845189f};
#define HI_BASELINE_MEDIAN_RMS 0.0398039291f
#define HI_BASELINE_HI_MEAN    0.387418972f
#define HI_THRESHOLD           1.01187521f
#define HI_SLOPE_THRESHOLD     0.0212129833f  /* healthy normal_0-2 holdout slopes (n=14) */

#endif
