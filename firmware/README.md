# ESP32 HI core

`src/hi_core.c` is plain C with float math and caller-owned memory. For a 1024-sample
window it requires three 1024-float buffers (input, FFT real, FFT imaginary): **12,288
bytes**. Features, baseline mean/std, and control state add under 256 bytes.

The radix-2 FFT uses 5,120 complex butterflies (about 20,480 float multiplications and
30,720 additions/subtractions), plus 1,023 runtime `sinf`/`cosf` evaluations in the
current compact implementation. The time-domain pass is roughly 10,000 basic float
operations and the band-energy scan has 513 bins. Precomputing twiddles trades about
4 KiB of flash/RAM for fewer trigonometric calls if profiling requires it.

After real CWRU features exist, run `python3 tools/export_baseline_header.py`, generate
the stored-window test-vector header, then compile the harness with:

`cc -std=c99 -Ifirmware/include firmware/src/hi_core.c firmware/tests/test_health_index.c -lm -o /tmp/test_hi`

The Arduino sketch is in `arduino/health_monitor.ino`; replace its accelerometer stub
with the board-specific driver before flashing.
