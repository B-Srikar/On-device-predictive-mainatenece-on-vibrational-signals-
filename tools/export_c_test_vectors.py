"""Export two real CWRU windows and Python expectations for the C conformance test."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.health_index import matrix, split_normal
from src.preprocess import de_signal_and_metadata, extract_features, load_config, matched_rate_signal, window_signal

def c_array(values: np.ndarray) -> str:
    return ', '.join(f'{float(value):.9g}f' for value in values)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--mat', type=Path, default=Path('data/raw/cwru/normal_0.mat'))
    parser.add_argument('--features', type=Path, default=Path('data/processed/features_3200.csv'))
    parser.add_argument('--output', type=Path, default=Path('firmware/tests/hi_test_vectors.h'))
    args = parser.parse_args(); config = load_config(args.config)
    from src.health_index import load_rows
    rows, names = load_rows(args.features); baseline, _ = split_normal(rows, float(config['baseline_fraction']))
    mean, std = matrix(baseline, names).mean(axis=0), matrix(baseline, names).std(axis=0, ddof=0)
    signal, _ = de_signal_and_metadata(args.mat)
    signal, rate = matched_rate_signal(signal, args.mat.name, 'normal', int(config['source_sampling_rates'].get(args.mat.name, config['sampling_rate'])), int(config['sampling_rate']), int(config['target_rate']))
    windows = list(window_signal(signal, int(config['window_size']), float(config['overlap'])))[:2]
    if len(windows) != 2: raise ValueError('Need at least two windows')
    expected = [extract_features(window, rate, config['fft_bands_hz']) for window in windows]
    feature_vectors = [np.array([item[name] for name in names]) for item in expected]
    his = [float(np.mean((item - mean) / np.where(std == 0, 1, std))) for item in feature_vectors]
    ema_1 = his[0]; ema_2 = float(config['ema_alpha']) * his[1] + (1.0 - float(config['ema_alpha'])) * ema_1
    text = '\n'.join((
        '#ifndef HI_TEST_VECTORS_H', '#define HI_TEST_VECTORS_H', '#include "hi_core.h"',
        f'static const float HI_TEST_WINDOW_0[HI_WINDOW_SIZE] = {{{c_array(windows[0])}}};',
        f'static const float HI_TEST_WINDOW_1[HI_WINDOW_SIZE] = {{{c_array(windows[1])}}};',
        f'static const float HI_EXPECTED_FEATURES_0[HI_FEATURE_COUNT] = {{{c_array(feature_vectors[0])}}};',
        f'static const float HI_EXPECTED_FEATURES_1[HI_FEATURE_COUNT] = {{{c_array(feature_vectors[1])}}};',
        f'#define HI_EXPECTED_HI_0 {his[0]:.9g}f', f'#define HI_EXPECTED_HI_1 {his[1]:.9g}f',
        f'#define HI_EXPECTED_EMA_0 {ema_1:.9g}f', f'#define HI_EXPECTED_EMA_1 {ema_2:.9g}f', '#endif', ''
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text, encoding='utf-8')
    print(f'Exported two stored windows at {rate} Hz to {args.output}')
if __name__ == '__main__': main()
