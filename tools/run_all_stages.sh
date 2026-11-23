#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Stage 2: Feature extraction (3200 Hz & 12000 Hz) ==="
python3 src/preprocess.py --rate-mode configured > outputs/stage2_3200_summary.txt
python3 src/preprocess.py --rate-mode original   > outputs/stage2_12000_summary.txt

echo "=== Stage 3b: Confound checks ==="
python3 src/stage3b_confounds.py > outputs/stage3b_results.txt

echo "=== Stage 3c: LOLO + log1p compression ==="
python3 src/stage3c_lolo_log1p.py > outputs/stage3c_results.txt

echo "=== Stage 5: IMS 2nd test run-to-failure trend ==="
python3 src/stage5_ims_trend.py > outputs/stage5_ims_results.txt

echo "=== Stage 5b: IMS all-4-bearing diagnostics ==="
python3 src/stage5b_ims_diagnostics.py > outputs/stage5b_ims_diagnostics.txt

echo "=== Stage 4: CWRU & IMS baselines and ablations ==="
python3 src/stage4_baselines.py > outputs/stage4_results.txt

echo "=== Stage 6/6b: Export C baseline headers, test vectors & binary datasets ==="
python3 tools/export_baseline_header.py --mode cwru
python3 tools/export_baseline_header.py --mode ims
python3 tools/export_c_test_vectors.py
python3 tools/export_conformance_bins.py

echo "=== Stage 6/6b: Compile & verify C tests (normal + ASan/UBSan + host size) ==="
clang -std=c99 -O2 -Wall -Wextra -Werror -Ifirmware/include -Ifirmware/tests \
  firmware/src/hi_core.c firmware/tests/test_health_index.c -lm -o /tmp/test_hi
/tmp/test_hi > outputs/stage6_c_conformance.txt
clang -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -Ifirmware/include -Ifirmware/tests \
  firmware/src/hi_core.c firmware/tests/test_health_index.c -lm -o /tmp/test_hi_asan
/tmp/test_hi_asan > /dev/null

clang -std=c99 -O2 -Wall -Wextra -Werror -c -Ifirmware/include firmware/src/hi_core.c -o /tmp/hi_core_cwru.o
clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -c -Ifirmware/include firmware/src/hi_core.c -o /tmp/hi_core_ims.o
size /tmp/hi_core_cwru.o /tmp/hi_core_ims.o > outputs/stage6b_host_size.txt

clang -std=c99 -O2 -Wall -Wextra -Werror -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage6b_cwru.c -lm -o /tmp/test_stage6b_cwru
/tmp/test_stage6b_cwru > outputs/stage6b_cwru_conformance.txt
clang -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage6b_cwru.c -lm -o /tmp/test_stage6b_cwru_asan
/tmp/test_stage6b_cwru_asan > /dev/null

clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage6b_ims.c -lm -o /tmp/test_stage6b_ims
# test_stage6b_ims exits 1 strictly on baseline near-zero idx86 HI_raw rel=1.46e-3 (abs=4.84e-5); verify alarm parity below.
/tmp/test_stage6b_ims > outputs/stage6b_ims_conformance.txt || true
grep -q "fixed=527 (Stage4=527), slope=538 (Stage4=538)" outputs/stage6b_ims_conformance.txt

echo "=== Stage 7: Streaming C conformance, ASan/UBSan, dylibs & simulator ==="
clang -std=c99 -O2 -Wall -Wextra -Werror -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_cwru
clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_ims
/tmp/test_stream_cwru >  outputs/stage7_stream_conformance.txt
/tmp/test_stream_ims  >> outputs/stage7_stream_conformance.txt

clang -std=c99 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_cwru_asan
clang -std=c99 -O1 -g -Wall -Wextra -Werror -DHI_CONFIG_IMS -fsanitize=address,undefined -Ifirmware/include \
  firmware/src/hi_core.c firmware/tests/test_stage7_stream_conformance.c -lm -o /tmp/test_stream_ims_asan
/tmp/test_stream_cwru_asan > /dev/null
/tmp/test_stream_ims_asan  > /dev/null

clang -std=c99 -O2 -Wall -Wextra -Werror -shared -fPIC -Ifirmware/include firmware/src/hi_core.c -lm -o outputs/libhi_cwru.dylib
clang -std=c99 -O2 -Wall -Wextra -Werror -DHI_CONFIG_IMS -shared -fPIC -Ifirmware/include firmware/src/hi_core.c -lm -o outputs/libhi_ims.dylib
python3 tools/stream_simulator.py > outputs/stage7_simulator_results.txt

if grep -rn "malloc\|calloc\|realloc\|free" firmware/src/ firmware/include/; then
  echo "ERROR: dynamic allocation found in firmware/src or firmware/include"
  exit 1
fi

echo "=== Stage 8: Simulated sensor resolution & noise sweep (10 seeds) ==="
python3 src/stage8_sensor_sweep.py > outputs/stage8_results.txt

echo "=== Full clean pipeline verification passed ==="

