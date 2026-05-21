# Bell Functionals for Encoded Qutrit Graph States

This package computes the Bell functional `I` for three qutrit graph-state
candidates:

- `two_qutrit`
- `ghz3`
- `ame43`

Each qutrit is encoded into two qubits by an isometry `E: C^3 -> C^4` with
`E^dagger E = I_3`. The default encoding is `E_Z`, mapping qutrit basis states
to `|00>`, `|01>`, `|10>`. The implementation does not assume this special
support: `embed_operator_E(E, O)` builds `E O E^dagger + I_perp`, and
`embed_projector_E(E, P)` builds the physical qutrit outcome projector
`E P E^dagger`.

## Theory Summary

For qutrits, `omega = exp(2*pi*i/3)`, with shift and clock operators

```text
X|j> = |j+1 mod 3>
Z|j> = omega^j |j>.
```

The Bell expressions use the audited qutrit graph-state convention from
`docs/bell_qutrit_graph_states_measurements.tex` and the notebooks. The
realizational `A_t(t,n)` matrices are built so that the replacement
`A_tilde_k^(n)` recovers the stabilizer factor `(X Z^k)^n`.

The Estimator path never submits a raw non-Hermitian observable. It splits every
operator as

```text
Re(O) = (O + O^dagger)/2
Im(O) = (O - O^dagger)/(2i)
```

and computes `<O> = <Re(O)> + i <Im(O)>`.

## Estimator

```powershell
python -m bell_functionals_qutrit.cli --candidate two_qutrit --backend estimator
python -m bell_functionals_qutrit.cli --candidate ame43 --backend estimator
python -m bell_functionals_qutrit.cli --candidate ghz3 --backend estimator
```

The default CLI uses exact `Statevector` evaluation through the Estimator-style
Hermitian split. `bell_value_estimator(...)` also accepts an explicit Qiskit
Estimator-compatible object and a circuit.

## Sampler

```powershell
python -m bell_functionals_qutrit.cli --candidate two_qutrit --backend sampler --shots 20000
```

The sampler backend diagonalizes the actual qutrit measurement operators,
embeds their projectors with `E P_a E^dagger`, and samples from those projective
probabilities. This avoids the invalid shortcut `00,01,10 -> 0,1,2` for a
general encoding `E` with possible support on `|11>`.

With `shots=None`, `bell_value_sampler(...)` evaluates the exact projective
probability sums. With finite shots, it samples each term independently and
reports the same leakage diagnostic.

## Leakage

For `N` qutrits, the code-space projector is

```text
P_code = (E E^dagger)^{tensor N}
```

and leakage is reported as

```text
p_leak = 1 - <psi|P_code|psi>.
```

If `p_leak` is not close to zero, the returned Bell value should be interpreted
with care. The sampler defaults to postselection on non-leakage outcomes while
still reporting `p_leak`.

## Expected Bounds

| candidate | beta_Q | beta_L | source |
| --- | ---: | ---: | --- |
| `two_qutrit` | 6 | `6 cos(pi/9) ~= 5.63816` | PDF |
| `ame43` | 8 | `7.63816` | PDF |
| `ghz3` | 6 | computed by deterministic brute force | numeric |

For `ghz3`, the quantum value comes from the PDF formula
`(d-1)(N-N1+d-1)`, giving `6` for `d=3`, `N=3`, `N1=2`. The PDF does not state a
confirmed classical bound for this specific GHZ expression, so the CLI labels
its classical value as `numeric_bruteforce`.
