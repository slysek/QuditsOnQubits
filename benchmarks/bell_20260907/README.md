# Bell benchmark: offline reproduction

This bundle contains **the complete earlier analysis and the Fez/Garnet repeat with 5000 shots per setting**. Bell values, bootstrap SE/95% CI, F3/baseline differences, and historical ZNE can be recomputed from saved real-QPU results. Replaying the analysis requires no IBM/IQM credentials and consumes no QPU credits.

## Results summary

The repeat campaign completed **48/48 variants**: 3 states (`two_qutrit`, `ghz3`,
`ame43`) x (baseline + 3 candidate encodings) x 2 F3 versions x 2 real processors
(IBM Fez, IQM Garnet). Each of the 544 Bell settings received exactly 5000 shots,
2 970 000 shots in total including calibrations. ZNE was omitted in this repeat.

### Highest measured values across all bases and both F3 versions

The values below are **post-selected, without readout correction**; `+/-` is one
bootstrap SE. Each row is the maximum over the eight (basis x F3) cells for that
backend and state. **Selection of the maximum is exploratory; the intervals carry
no multiple-comparison correction.**

| Backend | State | Basis | F3 | Bell |
| --- | --- | --- | --- | --- |
| ibm | two_qutrit | sup023_P021_ph000 | optimal | 5.0309 +/- 0.0351 |
| ibm | ghz3 | sup023_P021_ph001 | optimal | 4.5739 +/- 0.0337 |
| ibm | ame43 | canonical_ez | optimal | 5.3524 +/- 0.0404 |
| iqm | two_qutrit | canonical_ez | standard | 4.7397 +/- 0.0350 |
| iqm | ghz3 | sup023_P021_ph001 | optimal | 3.6363 +/- 0.0350 |
| iqm | ame43 | canonical_ez | optimal | 2.0291 +/- 0.0468 |

For `two_qutrit` the ideal value is 6 and the nominal classical bound is
5.6381557. **No raw result exceeded the nominal classical bound.** A post-selected
or mitigated value is not a loophole-free proof of nonlocality; the parties'
qubits sit on the same processor.

Best candidate against the baseline, F3 optimal: IBM improves for `two_qutrit`
(+0.2293 +/- 0.0498, +4.78%) and `ghz3` (+0.1502 +/- 0.0478, +3.40%) but degrades
for `ame43` (-0.1485 +/- 0.0578, -2.78%); IQM improves
for `ghz3` (+0.1181 +/- 0.0496, +3.36%) but degrades for `two_qutrit`
(-1.2187 +/- 0.0527, -25.93%) and `ame43` (-0.8996 +/- 0.0666, -44.33%).

Against the earlier series the median post-selected SE fell by 56.15% (IBM) and
64.16% (IQM). Fez reached higher values, but `rep_delay` also changed from 250 to
75 us and twirling from 16 to 8 randomizations, so the change cannot be attributed
to the shot count alone. IQM shows a decline in part of the results, especially
AME; checks of instructions, mapping and calibration did not establish a cause,
and the results were neither discarded nor replaced by the earlier ones.

QPU cost of the repeat: IBM 127 s; IQM upper bound 526.660 s, i.e. 263.330
credits under the author's own conversion factor, not a provider rate. Whole
campaign: IBM 438/540 s; IQM upper bound 745.598/1000 s, i.e. 372.799/500 credits.
The IQM figure is a timeline-derived limit, not an invoice.

## Reports and data

The full reports below are the primary evidence. They were written during the
original campaign and **retain their original language (Polish)**; the summary
above covers their headline results in English.

- [5000-shot report: Fez + IQM](reports/rerun5000/RAPORT.md) (Polish)
- [First benchmark: Kingston, Marrakesh, Fez, IQM, and ZNE](reports/RAPORT_KONCOWY.md) (Polish)
- [Repeat methodology and limitations](reports/rerun5000/METODY.md) (Polish)
- [Encoding matrices](reports/coding_bases.json)
- [Data manifest and SHA-256 hashes](manifest.json)

## Installation — Python 3.12

Run these commands from the root of a normal repository checkout, such as `main` or the `v0.1.0` tag once available.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-bell
& .venv-bell/Scripts/python.exe -m pip install -r benchmarks/bell_20260907/requirements.txt
& .venv-bell/Scripts/python.exe -m pip install --no-deps -e .
& .venv-bell/Scripts/python.exe benchmarks/bell_20260907/reproduce.py --bootstrap
```

Linux/macOS:

```bash
python3.12 -m venv .venv-bell
.venv-bell/bin/python -m pip install -r benchmarks/bell_20260907/requirements.txt
.venv-bell/bin/python -m pip install --no-deps -e .
.venv-bell/bin/python benchmarks/bell_20260907/reproduce.py --bootstrap
```

Installation downloads dependencies; subsequent replay works offline. `requirements.txt` pins the scientific libraries and SDKs used for the measurements. `constraints.txt` additionally pins transitive dependencies from the validated environment. This is the benchmark environment, not an installation of every optional repository feature. Full bootstrap replay can take several minutes, depending on the CPU. Outputs go to `artifacts/reproduced-bell-20260907/`. The script refuses to overwrite an existing directory; use a new `--output` for subsequent runs.

## Replay modes

These commands assume an active environment and its `python` interpreter:

```bash
# Archive integrity only; requires only the Python standard library.
python benchmarks/bell_20260907/reproduce.py --verify-only

# All Bell values from counts; SE/CI from saved bootstrap replicas.
python benchmarks/bell_20260907/reproduce.py --output artifacts/bell-quick

# All values, 2000 bootstrap replicas, and SE/CI agreement checks.
python benchmarks/bell_20260907/reproduce.py --bootstrap --output artifacts/bell-full

# Only the 5000-shot repeat.
python benchmarks/bell_20260907/reproduce.py --series rerun5000 --bootstrap --output artifacts/bell-5000
```

Use `--bootstrap` for full reproduction. Quick mode explicitly records `bootstrap_recomputed: false`; it does not present historical replicas as new calculations. Full mode compares each job's values, SE, and CI endpoints with the publication at tolerance `1e-9`. It also checks all 144 published primary results: 96 from the earlier campaign and 48 from the repeat. Bootstrap assumptions are documented in the reports; numerical agreement does not remove device systematic errors.

## Output files

- `verification.json`: verified job count, scope, and agreement-check results.
- `results.csv`: 240 aggregate rows across all series and scales; four estimators, SE, CI, shot counts, and postselection rejection fractions. Series and backends have separate keys.
- `differences.csv`: optimized F3 − standard, candidate − baseline with the same F3, optimized candidate − standard baseline, and repeat − earlier series.
- `zne.csv`: only the earlier series with complete scales 1/3/5; linear intercepts and curvature diagnostics. ZNE was omitted from the 5000-shot repeat.
- `data/`: extracted original counts, receipts, QPY circuits, calibrations, and reference values. Full bootstrap adds recomputed analyses beside the corresponding jobs.

Static reports, plots, and original CSVs are already available in `reports/`; reading them requires no execution. Column names and row organization in the replay export may differ from historical reports; results are identified by series, backend, state, encoding, F3, and scale.

## Contents and provenance

`data.zip` is approximately 19 MB and contains 383 files. The manifest covers 49 jobs: 43 completed (including a pilot), 5 cancelled, and 1 failed. The pilot is excluded from encoding comparisons. Original counts, QPY circuits, and bootstrap replicas are preserved byte for byte, with SHA-256 for each file. Receipts for failed/cancelled jobs remain in the record; their unexecuted QPY circuits are omitted. No `.env`, credentials, logs, or local process files are included.

The source was commit `156305e8b21d78ba7e729070d86e97e7d21e734e` plus the decoder correction for an inactive AME participant with monomial supports. That correction and its tests are included in the repository, along with the required Qiskit/QPY bit-index normalization and IBM configuration helpers.

Historical provenance files are preserved records, not executable configuration. After publication the path fields in `reports/rerun5000/protocol.json` (`circuit_sources`) and `reports/rerun5000/provenance.json` (`raw_data_root`, `analysis_root`) were normalised from absolute Windows paths to repository-relative form. No hash in `manifest.json` or in `provenance.json` covers those two files, so the distribution integrity boundary is unaffected. The same normalisation was applied to the single absolute path recorded in `validation-original.txt`; no other byte of that log was changed, and it is likewise not covered by any recorded hash. Report links are portable. The binding distribution manifest is `manifest.json`; historical report hashes recorded before link adjustments do not describe the published copies.

`study.py`, `hardware.py`, `target_screen.py`, `execute.py`, and `rerun5000.py` retain the design and execution code for the historical campaign. `report.py` is the original first-stage report generator. The supported entry point for reproducing the publication is **`reproduce.py`**. Historical hardware scripts expect the former `weighted/` layout and are not ready-to-use launchers for a new campaign.

## New QPU execution

Replaying saved counts can reproduce the published numbers. A new IBM/IQM experiment will have new random outcomes, different calibrations, and potentially different quality. The bundle does not submit new jobs automatically.

After extraction, the exact submitted circuits are in `data/jobs/<provider>/<label>/submitted.qpy`; settings, backend, shots, and options are in `receipt.json`. Circuits can be loaded with `qiskit.qpy.load`. Before any new measurement, configure your own credentials outside the repository and check backend availability/topology, ISA compatibility, pricing, and a separate budget. Do not reuse historical job IDs or treat the historical campaign limit as your account balance.

The 5000-shot repeat used Fez `rep_delay=75e-6`, reset enabled, XY4 DD, and twirling 8 × 625; IQM Garnet used one complete variant and two calibrations per job. Each setting has 5000 shots **before** postselection. Postselected/corrected results do not constitute a loophole-free Bell test. The particularly weak IQM repeat results remain in the data; their cause has not been established.
