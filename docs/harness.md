# State preparation harness

The harness searches physical qubit circuits that prepare a specified qudit state.
It combines structure proposals, numerical angle fits, independent state checks,
and records that later searches can reuse.

Use a repository checkout with Python 3.11–3.13. Install the optional search dependency:

```bash
python -m pip install -e ".[evolution]"
qoq-evolution menu
```

Run commands from the repository root. The menu selects the target, encoding mode,
search method, time limits, and optional processor model.

## A short offline search

```bash
qoq-evolution discover --state two_qutrit --encoding-mode free \
  --offline-model mutations --families 1 --iterations 3 --polish-seconds 0 \
  --output artifacts/evolution/first_search
```

This command uses random CZ edits and local numerical fits. It sends no model
requests and submits no QPU jobs. The search starts from a compatible published
record. A short run can preserve the starting circuit without an improvement.
Use a new output directory for each search.

`--encoding-mode fixed` keeps the starting encoding.
`--encoding-mode free` fits an encoding for each party and reports its decoder cost separately.
The final state error and leakage norms must each be at most `1e-10`.
Leakage probability is the square of the leakage norm.

## Targets and saved circuits

| Targets | State identifiers |
| --- | --- |
| Qutrit graph states | `two_qutrit`, `ghz3`, `ame43` |
| Other qutrit states | `aharonov3`, `dicke4`, `cluster4`, `aklt4` |
| Other AME states | `ame44`, `ame45`, `ame47`, `ame48`, `ame64` |
| AME(4,6) | `Bruzda60`, `Pandya60prime`, `HZ96`, `Casas36` |

Published circuits live in [`quantum_circuits/`](../quantum_circuits/README.md).
Each record contains its output encoding and a QPY preparation circuit.
Records with a decoder contain a separate `decoder.qpy` file.

Without a compatible record, graph-state searches use the published baselines
when the original local baselines are absent. Select the canonical baseline explicitly:

```bash
qoq-evolution discover --state ghz3 --method canonical_direct \
  --baseline experiment_inputs/state_preparation --encoding-mode free \
  --offline-model mutations --families 1 --iterations 3 --polish-seconds 0 \
  --output artifacts/evolution/from_canonical
```

Generated targets such as `cluster4` create their baseline on first use.
Large states can require substantially more fitting time than the short example.

## AME(4,6)

Import the published tensors and circuit baselines once:

```bash
qoq-evolution ame46-import --targets experiment_inputs/ame46/targets.npz \
  --circuits experiment_inputs/ame46/circuits
```

This command checks all four states and writes `artifacts/evolution/ame46/`.
It refuses to replace existing imported data.
Then select an AME(4,6) state in the menu or use `--state Bruzda60`.
Fits from random angles often fail for these states. The known circuits provide useful starting points.

To check a saved record without an import, set `QOQ_AME46_TARGETS` to
`experiment_inputs/ame46/targets.npz`. See the [input manifest](../experiment_inputs/ame46/README.md).

## Agents and processor models

The menu can select a construction agent through `--offline-model agent`.
The option name is historical. This mode starts Codex sessions, needs a configured
Codex account, and uses network access and account quota.
`--model-config` selects an LLM through an OpenAI-compatible endpoint instead.
Keep credentials outside tracked files.

Add `--architecture iqm_garnet` to rank circuits after offline layout and routing.
Other choices are `ibm_heron_r2`, `ibm_heron_r3`, `ibm_nighthawk`,
`iqm_emerald`, and `iqm_sirius`.
These models use captured or synthetic device data. Their success estimates are
model outputs, not measured fidelities. The search submits no hardware jobs.

The default objective ranks preparation costs. `--objective total` includes the
decoder before routing. `--objective cost` requires a processor model and ranks
its estimated success first. Compare results only under the same objective and settings.

## Results and resume

The output directory contains `discovery.json` and one directory per family and round.
Completed campaigns contain `summary.json`, `manifest.json`, `best.qpy`,
`best_encodings.npy`, and the saved candidate program. Decoder files are separate.
The harness checks the saved QPY again before it accepts a winner.

Discovery compares each valid winner with compatible records under
`quantum_circuits/<state>/<processor>/`. A better winner can replace the record.
The archive of replaced records is under `artifacts/evolution/records/`.
Cost-objective records use separate `<processor>_cost` directories.

Resume an interrupted operation with its output directory:

```bash
qoq-evolution resume artifacts/evolution/first_search
```

Use `qoq-evolution discover --help` for all search limits and options.
The [published results](../benchmarks/state_preparation/README.md) include a command
that checks the saved evidence without a new search.
The [research log in Polish](pipeline-ewolucji-stanow.md) contains the development history.
