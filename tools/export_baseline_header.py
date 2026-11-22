"""Export real healthy-feature baseline statistics as firmware/include/hi_baseline.h."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.health_index import load_rows, matrix, split_normal
from src.preprocess import load_config

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=Path('config.yaml'))
    parser.add_argument('--features', type=Path, default=Path('data/processed/features_3200.csv'))
    parser.add_argument('--output', type=Path, default=Path('firmware/include/hi_baseline.h'))
    args = parser.parse_args()
    config = load_config(args.config)
    rows, names = load_rows(args.features)
    baseline, _ = split_normal(rows, float(config['baseline_fraction']))
    values = matrix(baseline, names)
    if len(names) != 10:
        raise ValueError(f'Expected 10 features, found {len(names)}')
    mean, std = values.mean(axis=0), values.std(axis=0, ddof=0)
    body = '\n'.join((
        '#ifndef HI_BASELINE_H', '#define HI_BASELINE_H', '#include "hi_core.h"',
        'static const float HI_BASELINE_MEAN[HI_FEATURE_COUNT] = {' + ', '.join(f'{x:.9g}f' for x in mean) + '};',
        'static const float HI_BASELINE_STD[HI_FEATURE_COUNT] = {' + ', '.join(f'{x:.9g}f' for x in std) + '};', '#endif', ''
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding='utf-8')
    print(f'Exported {len(baseline)} healthy windows to {args.output}')
if __name__ == '__main__': main()
