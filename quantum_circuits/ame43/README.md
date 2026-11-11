# AME(4,3): obwody na procesory

Jeden podfolder na procesor. W każdym:

- `results.json`: wyniki i parametry. Źródło (run, rodzina), ustawienia ewolucji, metryki logiczne (CZ, głębokości,
  dekoder, F), metryki po routingu na procesor (bramki 2q, głębokość, szacowany sukces), kodowanie qutrytów
  (`encoding.real` / `encoding.imag`, 4 × 4 × 3) i layout (`initial_physical_qubits`, `final_physical_qubits`).
- `circuit.qpy`: obwód taki, jaki go znaleziono (logiczny, `u` + `cz`, 8 kubitów, qutryt k na kubitach 2k i 2k+1).
- `circuit_native.json`: opis obwodu w bramkach natywnych procesora (lista bramek: `name`, `qubits` fizyczne, `params`).
- `circuit_native.qpy`: ten sam obwód natywny, gotowy do wczytania (`qiskit.qpy.load`), z layoutem.
- `decoder.qpy`: dekoder z kodowania obwodu do kanonicznego (poziom j qutrytu → stan bazowy j pary kubitów).

Obwody nie mają pomiaru. Kubit logiczny k kończy na kubicie fizycznym `final_physical_qubits[k]`.
Layout pochodzi z kalibracji fake backendu Qiskit (2026-11-05). Na prawdziwym urządzeniu warto zroutować
`circuit.qpy` (+ `decoder.qpy` + pomiar) od nowa.

Pełne archiwum runów (wszystkie rodziny, logi, `verify.py`):
`notebooks/private/ame43_evolution_20261105/`.
