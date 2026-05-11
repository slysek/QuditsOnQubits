# Milestones

Folder zawiera "save pointy" dla wiekszych zmian w projekcie. Kazdy plik
opisuje stan repo na konkretny moment, dodane / zmienione pliki, komendy
do odtworzenia kontekstu i krotkie instrukcje na wypadek awarii.

Konwencja nazewnictwa:

```
docs/milestones/<YYYY-MM-DD>-<short-slug>.md
```

Czytaj **od najnowszego do najstarszego** zeby zrozumiec, co aktualnie
jest pod reka.

## Indeks

- [`2026-04-30-graph-states-extended-suite.md`](./2026-04-30-graph-states-extended-suite.md)
  — dodanie jednoetapowego pipeline'u `graph_states_extended` do nocnych
  benchmarkow nowych qutrytowych stanow grafowych.

## Szybko: benchmark tylko dla wybranych klas

Nowy runner suite (`encoding_search_v2.suite_cli`) domyslnie bierze wszystkie
klasy z `QuditsOnQubits.benchmark_encoding_bases.ALL_CLASS_NAMES`. Jezeli
chcesz ograniczyc benchmark do kilku klas, dodaj `--class-filter` z lista
oddzielona przecinkami:

```powershell
python -m encoding_search_v2.suite_cli `
    --states path4,cycle4 `
    --candidate-mode full `
    --class-filter baseline,product,local_ry_only `
    --jobs 4 `
    --n-transpile-runs 2 `
    --output-root data/benchmarks/suites_smoke_selected
```

To odpali tylko kandydatow z klas `baseline`, `product` i `local_ry_only`
dla stanow `path4` oraz `cycle4`.

Dla calego zestawu `graph_states_extended`, ale tylko wybranych klas:

```powershell
python -m encoding_search_v2.suite_cli `
    --suite graph_states_extended `
    --class-filter baseline,product,local_general_su2 `
    --jobs 32 `
    --output-root data/benchmarks/suites_selected_classes
```

Wazne: przy takim eksperymencie najlepiej ustawic osobny `--output-root`.
Opcja `--skip-existing` patrzy tylko, czy istnieje CSV dla danego stanu,
nie sprawdza, czy poprzedni run byl robiony z tym samym `--class-filter`.

Nazwy klas, ktore mozna podac w `--class-filter`:

```text
baseline
monomial_old_codespace
monomial_full
fourier_like
householder_random
clifford_wh
haar_random_isometry
perturbed_isometry
entangling_isometry
structured_entangling
product
local_ry_only
local_general_su2
real_orthogonal
near_identity
finer_structured
two_cz_ansatz
```

Starsze entrypointy maja podobny filtr, ale pod nazwa `--class`, np.:

```powershell
python -m QuditsOnQubits.benchmark_encoding_bases_parallel `
    full path4 `
    --class baseline,product `
    --max-workers 4 `
    --no-circuit-export
```
