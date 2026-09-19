## Summary

Describe the problem, resulting behavior, and scientific limitations. Link the
issue discussed before a substantial change; otherwise explain why none applies.

Related issue:

## Test evidence

Record the exact pytest paths or commands run and their results, including skips
or checks you could not run. Use local simulation or provider test doubles.

```text
Command:
Result:
```

For unified benchmark changes, include the relevant `tests/test_benchmark_*.py`
tests and `tests/test_encoding_benchmark_notebook.py`. For packaging changes,
record the distribution checks from the
[contribution guide](https://github.com/slysek/QuditsOnQubits/blob/main/CONTRIBUTING.md#validate-a-distribution).

## Reproducibility impact

Does this change saved artifacts, manifests, hashes, deterministic inputs, or
replay behavior? Describe affected formats and existing runs, or state that there
is no impact. For benchmark results, record versions, target, seeds, tolerances,
and the manifest; explain how the baseline and compiler budget remain comparable.

## Checklist

Check applicable items and explain any that do not apply.

- [ ] I followed the [contribution guide](https://github.com/slysek/QuditsOnQubits/blob/main/CONTRIBUTING.md).
- [ ] Bug fixes include a small reproducible input and relevant regression coverage.
- [ ] Normal tests require no hardware calls or credentials.
- [ ] Benchmark comparisons preserve comparable baselines, budgets, and thresholds;
      distinct backend or synthesis protocols are not pooled into one Pareto front.
- [ ] Deterministic inputs use `experiment_inputs/` or `examples/data/` with
      provenance and hashes; generated runs belong in `artifacts/`.
- [ ] No credentials, personal environment files, or large generated campaigns are included.
- [ ] Examples are short and in English; limitations and validation results are documented.
