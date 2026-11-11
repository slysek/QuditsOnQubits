# QuditsOnQubits

**A research harness that finds smaller qubit circuits for qudit states.**

The harness searches circuit structures and local qudit encodings, fits gate
angles, and checks the complete output state. It saves valid circuits, their
encodings, decoder costs, and results for further searches.

[Harness guide](docs/harness.md) ·
[Results and checks](benchmarks/state_preparation/README.md) ·
[Saved circuits](quantum_circuits/README.md)

## Results

Saved results through **11 November 2026**. The comparison uses canonical direct
preparation as its baseline. All counts include preparation from physical zero.

| State | Qubits | Canonical CZ | Harness CZ | Fewer CZ | Decoder CZ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Two-qutrit graph state | 4 | 8 | **3** | **62.5%** | 3 |
| GHZ₃ graph state | 6 | 15 | **5** | **66.7%** | 4 |
| AME(4,3) | 8 | 28 | **17** | **39.3%** | 6 |

These are statevector-checked preparation costs before routing and measurement.
The decoder maps the output to the canonical encoding and adds the cost shown.
The saved fidelities exceed `1 - 1e-12`, with final leakage probability below `1e-20`.
These results do not establish global minima or gains on a QPU.

**AME(4,6) is also available:** 12-qubit circuits with **46 CZ** for `Bruzda60`
and **37 CZ** for `Casas36`, plus source tensors and four importable baselines.
These counts already occur in the imported baselines. The harness preserves them
and finds some smaller depths. See the [result details](benchmarks/state_preparation/README.md).

## How the harness works

A qutrit occupies three dimensions of a two-qubit block. The unused dimension
gives the search more freedom. Intermediate states can leave the code space.
The final state must return to the selected encoding and match the target.
This controlled use of **temporary leakage** can reduce the preparation cost.

1. Select a target state, encoding mode, and optional processor model.
2. Propose structures with random mutations, an LLM, or a construction agent.
3. Fit angles and check state error and leakage norms against `1e-10`.
4. Rank valid circuits and save QPY files, encodings, decoders, and search history.

The harness supports searches with fixed or free encodings, shared results between
families, saved sessions, and offline routing for IBM and IQM processor models.
The [guide](docs/harness.md) describes the commands and their limits.

## Start locally

Use Python **3.11–3.13**. Run these commands on Linux or macOS:

```bash
git clone https://github.com/slysek/QuditsOnQubits.git
cd QuditsOnQubits
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[evolution]"
qoq-evolution menu
```

The menu selects the state, search method, time limit, and processor model.
For a short search without a network connection or provider account:

```bash
qoq-evolution discover --state two_qutrit --encoding-mode free \
  --offline-model mutations --families 1 --iterations 3 --polish-seconds 0 \
  --output artifacts/evolution/first_search
```

Run this from the repository root. Use a new output directory for each search.
The example starts from the published record. It checks the search pipeline
and does not guarantee an improvement. Agent searches need a configured Codex
account and use its quota. See [installation](docs/installation.md) for Windows.

## More

- [Harness guide](docs/harness.md): offline runs, agents, routing, AME(4,6), and resume.
- [Results](benchmarks/state_preparation/README.md): baselines, costs, files, and validation.
- [Encoding benchmark](docs/encoding_benchmark.md) and [notebook](notebooks/encoding_benchmark.ipynb): the earlier F₃/CZ₃ comparison.
- [Documentation](docs/README.md) and [advanced usage](docs/usage_guide.md): Bell measurements and provider tools.
- [Historical Bell measurements](benchmarks/bell_20260907/README.md): a separate hardware dataset.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and [CITATION.cff](CITATION.cff)
for citation metadata. Include the commit and result manifest when you report a result.
Licensed under [Apache-2.0](LICENSE).
