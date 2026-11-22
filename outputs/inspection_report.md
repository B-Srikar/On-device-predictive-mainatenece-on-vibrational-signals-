# Stage 1 — CWRU data inspection

**Inspection date:** 2026-11-10  
**Raw-data path checked:** `data/raw/cwru/`

## Local-file availability

The workspace did not contain `data/raw/` at the time of inspection. Consequently,
none of the requested source recordings is present and no raw data was changed.

| Requested file | Present locally | `scipy.io.loadmat` inspection | DE statistics / plot |
| --- | ---: | --- | --- |
| `normal_0.mat` | No | Not run — file absent | Not available |
| `IR007_0.mat` | No | Not run — file absent | Not available |
| `B007_0.mat` | No | Not run — file absent | Not available |
| `OR007@6_0.mat` | No | Not run — file absent | Not available |

No variable names, array shapes, sample counts, RPM field values, or DE-signal
statistics can be reported without inventing results. Likewise, no 0.2-second plots
were generated because no DE signals are available. The plots directory is retained
for the plots to be created after the recordings are supplied.

## CWRU catalog metadata

The official CWRU catalog identifies the requested 0 HP recordings as follows:

| Requested local label | Official catalog file / catalog entry | Condition | Catalog RPM | Sampling-rate determination |
| --- | --- | --- | ---: | --- |
| `normal_0.mat` | `97.mat` / `Normal_0` | Normal, 0 HP | 1797 | 12 kHz for the standard `97.mat` DE recording; confirm against the supplied file before use. |
| `IR007_0.mat` | `105.mat` / `IR007_0` | 0.007 in inner-race fault, 0 HP | 1797 | 12 kHz — listed on CWRU's **12k Drive End Bearing Fault Data** page. |
| `B007_0.mat` | `118.mat` / `B007_0` | 0.007 in ball fault, 0 HP | 1797 | 12 kHz — listed on CWRU's **12k Drive End Bearing Fault Data** page. |
| `OR007@6_0.mat` | `130.mat` / `OR007@6_0` | 0.007 in outer-race fault at 6:00, 0 HP | 1797 | 12 kHz — listed on CWRU's **12k Drive End Bearing Fault Data** page. |

Sources: CWRU's [download overview](https://engineering.case.edu/bearingdatacenter/download-data-file),
[normal baseline table](https://engineering.case.edu/bearingdatacenter/normal-baseline-data), and
[12 kHz drive-end fault table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data).

## Sampling-rate consistency check

No local sampling-rate field or time base could be checked because all four files are
absent. The requested fault labels are specifically from the CWRU 12 kHz drive-end
catalog. The normal catalog page itself does not print its sample rate, so it must not
be assumed to be 48 kHz merely because CWRU also distributes 48 kHz drive-end data.
For the standard `97.mat` normal recording paired with `105.mat`, `118.mat`, and
`130.mat`, the expected common rate is 12 kHz. `config.yaml` therefore uses 12000 Hz
for this defined set, subject to validation once the actual local files are present.

## Outputs and stage boundary

- Created `config.yaml` with `sampling_rate: 12000`, `window_size: 1024`,
  `overlap: 0.5`, blank `target_rate`, and `ema_alpha: 0.1`.
- Created `outputs/inspection_report.md` and reserved `outputs/plots/`.
- Feature extraction has **not** started.

