# d6_diagonal_optimizer — notatka z sesji (2026-04-30)

## Co zostało zrobione

Plik: `QuditsOnQubits/QuditsOnQubits/d6_diagonal_optimizer.py` — samowystarczalny moduł
do optymalizacji 6-kubitowej realizacji diagonalnej bramki D[Lambda] dla d=6
(kontekst AME(4,6)).

Public API (wszystko eksportowane przez `__all__`):

- mapowanie quditów ↔ kubitów: `encode_qudit_level`, `pair_to_qubit_index`,
  `qubit_index_to_pair`, `legal_indices`, `illegal_indices`,
- embedding diag36 → theta64: `diag36_to_theta64`, `normalize_phase`,
- analiza fazowa: `walsh_coefficients`, `phase_polynomial_cost`,
- synteza: `synthesize_phase_gadgets`,
- walidacja: `verify_on_code_space`, `extract_diagonal_from_circuit`,
- optymalizatory: `random_search`, `simulated_annealing`,
  `multi_restart_simulated_annealing`, `greedy_coordinate_descent`,
  `multi_restart_greedy`, `two_opt_descent`, `lp_relax_walsh_l1`,
  `transpile_metric`,
- główny wpis: `optimize_d6_diagonal(...)` — zwraca słownik z `best_theta64`,
  `best_diag64`, `walsh_coeffs`, `phase_gadget_circuit`, `transpiled_circuit`,
  `baseline_circuit`, `metrics`, itd.

## Konwencje (jawne, trzymać konsekwentnie)

- Każdy poziom kuditu `a in {0..5}` kodowany jako 3 bity binarnie:
  `0->000, 1->001, 2->010, 3->011, 4->100, 5->101`.
- Indeks pary `(a,b)` na 6 kubitach: `index = (a << 3) | b`.
  Qubity 5,4,3 niosą bity `a` (qubit 5 = MSB), qubity 2,1,0 niosą bity `b`.
- Stany `|110>` i `|111>` na każdym kudicie — illegalne (don't-care).
  Łącznie 36 legalnych + 28 illegalnych = 64 stany.
- `order="row-major"` (domyślnie): `diag36[6*a + b]` = faza `|a,b>`.
  `order="col-major"`: `diag36[a + 6*b]` = faza `|a,b>`.
- Konwencja Qiskita dla RZ: `RZ(phi) = exp(-i phi Z / 2)`, więc
  `exp(i c Z) = RZ(-2 c)` (sprawdzone numerycznie self-testem w `_self_tests`).
- Phase polynomial: `theta(x) = sum_S c_S * (-1)^{popcount(S & x)}`,
  globalna faza `c_0` ląduje w `qc.global_phase`, nie w bramkach.

## Aktualne wyniki

### Vector D[Lambda_{2,3}] z pierwszego promptu (`_example_d_lambda_2_3()`)

```
naive (dc=1):                             —          —    —
method="best_of":  Walsh terms = 41,  cx_est = 150,  max_supp = 5
  phase_gadget_circuit         cx=150 rz=41 depth=173
  phase_gadget post-transpile  cx=102 rz=41 depth=125
  DiagonalGate baseline        cx= 50 rz=41 depth= 85
```

### Vector D_Lambda_23 z drugiego promptu (różny od pierwszego!)

```
naive (dc=1):                             cx=60  rz=53  depth=107
method="both"  (RS + multi-restart SA):
  DiagonalGate baseline        cx=56  rz=45  depth= 94
method="best_of"  (RS + SA + LP + greedy + 2-opt) — DOMYŚLNY:
  phase_gadget_circuit         cx=130 rz=33 depth=139
  phase_gadget post-transpile  cx= 92 rz=33 depth= 98
  DiagonalGate baseline        cx= 52 rz=33 depth= 80
```

Dla `D_Lambda_23` `best_of` zbija depth ze 107 → 80 i cx z 60 → 52, oraz
liczbę Walsh terms z 53 → 33. `verify_on_code_space < 1e-14` na wszystkich
trzech wariantach (phase-gadget, transpiled, baseline).

## Co zmieniło się w ostatniej iteracji (kluczowe poprawki optymalizatora)

1. **Każdy stage dostaje pełny `max_iters`.** Wcześniej `max_iters` był
   dzielony przez liczbę stage'ów, więc dodanie kolejnego optymalizatora
   *psuło* wynik (mniej iteracji RS → gorsze lokalne minimum). Teraz RS,
   SA i greedy mają każdy swój budżet `max_iters`.
2. **Multi-restart SA.** `multi_restart_simulated_annealing` uruchamia
   `n_restarts` razy `simulated_annealing` z różnymi seedami
   (pierwszy restart warm-start z RS-best). Domyślnie `n_restarts=6`.
3. **`greedy_coordinate_descent` + `multi_restart_greedy`.** Tani
   1-opt hill-climber po 28 don't-care fazach × |alphabet| wartości,
   cyklicznie do braku poprawy. Zazwyczaj zbiega w 3-5 przejściach.
4. **`two_opt_descent`.** Dla każdej pary `(i,j)` sprawdza wszystkie
   `|alphabet|^2` przypisania jednoczesne. To ten ruch znalazł
   przełomowe rozwiązanie 33-termowe na `D_Lambda_23` (1-opt utykał
   na 41-43).
5. **`lp_relax_walsh_l1`** (scipy linprog HiGHS). Konceptualnie
   najmocniejszy: ciągła relaksacja L1 zadania
   `min ||c||_1` po fazach don't-care. Daje globalnie optymalne fazy
   ciągłe; zaokrąglenie do alfabetu nie zawsze zachowuje sparsity, ale
   dostarcza dobrego punktu startowego dla greedy. (Na `D_Lambda_23`
   sam zaokrąglony LP daje 44 terms; finalne 33 robi 2-opt.)
6. **`transpile_metric` + `polish_target`.** Opcjonalny finalny polish:
   1-opt greedy z funkcją kosztu `transpile(...).depth()` lub `cx`.
   Domyślnie `polish_target="cx_then_depth"` (lex order). W obu
   testowanych przypadkach polish nie znalazł poprawy poza Walsh,
   ale zostawiony bo bywa przydatny.

Domyślne `method="best_of"` uruchamia kolejność: RS → multi-restart SA →
LP → LP+greedy → multi-restart greedy → 2-opt, każdy etap aktualizuje
`best_phases` jeśli daje niższy koszt.

## Najważniejsze parametry `optimize_d6_diagonal`

- `phase_alphabet`: `"default"` = `[0, 2π/3, -2π/3]` (dla wektorów
  ω₃-strukturalnych); `None` = unikatowe fazy z legalnego inputu;
  iterowalny float = własny alfabet.
- `method`: `"best_of"` (default), `"both"`, `"random"`, `"sa"`,
  `"greedy"`, `"lp"`.
- `n_restarts=6` — dla SA i greedy.
- `polish_target` ∈ `{None, "depth", "cx", "weighted", "cx_then_depth"}`,
  `polish_via` ∈ `{"diagonal_gate", "phase_gadget"}`.
- `qiskit_basis=["rz","sx","x","cx"]`, `optimization_level=3`.
- `order="row-major"` / `"col-major"` — kolejność wpisów w `diag36`.
- `cost_weights={"cx":1, "terms":0.1, "support":0.05, "weighted":0}`
  (domyślne).

## Self-testy / walidacja

`_self_tests()` w pliku (uruchamiane w `main()`):

- mapowanie indeksów (legal/illegal rozłączne, suma = `range(64)`),
- round-trip Walsh-Hadamard,
- zgodność `synthesize_phase_gadgets` z `qiskit.DiagonalGate` na losowym
  wektorze (do globalnej fazy),
- `verify_on_code_space` < 1e-7 na losowym `diag36`.

`verify_on_code_space(qc, diag36, order)` zwraca
`max_error_on_code_space` na 36 legalnych indeksach z poprawką globalnej
fazy. `extract_diagonal_from_circuit(qc)` rzuca `ValueError`, gdy obwód
nie jest diagonalny do tolerancji.

## Otwarte sprawy / pomysły na dalej

1. **2-opt z metryką transpile** — obecnie polish robi tylko 1-opt po
   `transpile`. 2-opt × |alphabet|^2 × 28×27/2 ≈ 3.4k transpile calls,
   ~3 minuty — czasem zejdzie z `cx`/`depth` jeszcze trochę. Możliwe
   ograniczenie: 2-opt tylko na parach z najwyższym wkładem do kosztu.
2. **Szerszy alfabet.** Dla `D_Lambda_23` testowałem tylko
   `[0, ±2π/3]`. Rozszerzenie do `[0, ±π/3, ±2π/3, π]` może otworzyć
   nowe minima, ale zwykle pogarsza Walsh sparsity dla wektorów ω₃.
   Wartość — sprawdzić eksperymentalnie.
3. **Dolna granica.** Dla 28 zmiennych ciągłych i 63 niezerowych c_S
   teoretyczna dolna granica nonzero termów ≈ 35 (dim = 64-28-1).
   Dla pierwszego wektora dostaliśmy 41, dla `D_Lambda_23` aż 33 —
   poniżej naïve granicy, bo struktura ω₃ wymusza zależności w macierzy
   B (rangę < 28). Warto policzyć dokładną dolną granicę przez ranko-
   liczenie albo solver MIP (mała instancja, 28 binarek na wartości
   alfabetu — rozwiązywalne CPLEX/Gurobi/SCIP w sekundach).
4. **Polish phase-gadget + komutacje.** `synthesize_phase_gadgets`
   robi naïwne CNOT laddery (2(k-1) CNOT-ów per term). Lepsza synteza
   (ParityNetwork z Amy/Maslov, Gray-code-merge) zbiłaby `cx` ≈ 130
   dla phase_gadget circuit. To akurat w pliku jeszcze nie ma — dziś
   `DiagonalGate` baseline (Qiskit Shende et al.) bije nasz phase
   gadget właśnie dlatego, że robi Gray-code merging.
5. **Kombinacja LP→MIP.** LP daje ciągły punkt; pomysł: użyć go jako
   warm-start MIP-a z binarnymi zmiennymi alfabetu. Nie zaimplemen-
   towane.

## Minimalny snippet do reprodukcji

```python
import numpy as np
from d6_diagonal_optimizer import optimize_d6_diagonal, verify_on_code_space

omega3 = np.exp(2j * np.pi / 3)
omega3_bar = np.conjugate(omega3)
D_Lambda_23 = np.array([
    1,
    omega3_bar, omega3_bar, omega3_bar, omega3_bar,
    1, 1, 1,
    omega3, 1, omega3, 1, 1, omega3, 1, omega3,
    1, 1, 1, omega3, 1,
    omega3_bar, omega3, omega3_bar, omega3_bar,
    omega3, omega3, omega3_bar,
    1, 1,
    omega3, omega3, omega3_bar,
    1, 1, omega3_bar,
])

result = optimize_d6_diagonal(
    D_Lambda_23,
    phase_alphabet="default",
    max_iters=4000,
    seed=2024,
    order="row-major",
    method="best_of",
    verbose=True,
)

print(result["metrics"])                     # cx/rz/depth dla 3 obwodów
err = verify_on_code_space(result["baseline_circuit"], D_Lambda_23)
print(f"err on code space = {err:.2e}")      # ~1e-15
```

Oczekiwane: baseline depth=80, cx=52, rz=33, terms=33, err < 1e-14.

## Stan plików

- `QuditsOnQubits/QuditsOnQubits/d6_diagonal_optimizer.py` — moduł
  (przeszedł wszystkie self-testy + przykład w `__main__`).
- `QuditsOnQubits/QuditsOnQubits/docs/d6_diagonal_optimizer_notes.md` —
  ta notatka.
- Plik tymczasowy `_tmp_bench_d_lambda_23.py` skasowany.
- Linter (ReadLints) bez błędów.

Wcześniejszy moduł `find_best_diagonal_decomposition.py` (też w tym
folderze) zostaje nietknięty — robi powiązaną, ale ogólniejszą rzecz
(diagonalne na n kubitach, sparse phase-poly + DiagonalGate + permutacje
kubitów, brak optymalizacji don't-care). Można go traktować jako baseline
porównawczy.
