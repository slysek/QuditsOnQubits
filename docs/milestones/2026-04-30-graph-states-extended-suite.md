# Milestone — 2026-04-30: graph_states_extended benchmark suite

> Snapshot stanu pipeline'u po dodaniu nocnego, jednoetapowego benchmarku
> dla rozszerzonego zestawu qutrytowych stanow grafowych. Dokument sluzy
> jako "save point" — gdyby srodowisko padlo, ten plik zawiera komplet
> niezbednych informacji do odtworzenia kontekstu pracy.

**Baza:** commit `485e76e Refactor code structure for improved readability
and maintainability` (gałąź `main`).

**Status na koniec milestonu:**
- 19 / 19 nowych testow `tests/test_graph_states.py` przechodzi.
- 8 / 8 nowych testow `tests/test_suite_runner.py` przechodzi.
- 25 / 25 dotychczasowych testow `tests/test_encoding_search_v2.py` nadal
  przechodzi.
- 39 / 40 testow w `tests/test_benchmark_encoding_bases*.py` przechodzi;
  jedyny czerwony test
  (`TestProductGenerators::test_single_state_benchmark_can_filter_only_product_class`)
  jest *czerwony juz na czystym `main`* — to nie regresja ode mnie i
  wymaga osobnej naprawy, niezwiazanej z tym milestonem.
- 13 / 13 testow `test_create_ame_circuit.py` i 2 / 2 testow
  `test_repo_layout.py` przechodzi.

---

## 1. Co bylo dodane (zakres milestonu)

Nowy *jednoetapowy* pipeline benchmarkowy `graph_states_extended` —
jedna komenda CLI, ktora:

1. *generuje* qutrytowe stany grafowe ze zdefiniowanego zestawu nazw,
2. dla *kazdego* z tych stanow odpala benchmark dla *wszystkich* klas baz
   kodowania zaimplementowanych w
   `QuditsOnQubits.benchmark_encoding_bases`,
3. zapisuje wyniki w osobnym folderze suite z czytelna struktura,
4. loguje progres + ETA do stdout *i* do pliku `suite_run.log`,
5. dziala rownolegle (`--jobs N` -> `ProcessPoolExecutor`).

Pipeline *swiadomie* nie uzywa schematu stage1/stage2 ani preselekcji
top-k. To osobny, prostszy tor uruchomieniowy: state -> wszystkie
klasy -> benchmark -> wynik.

### Dodane stany grafowe

W rejestrze `EXTENDED_GRAPH_STATES` (28 stanow):

| rodzina   | stany                                      |
|-----------|--------------------------------------------|
| GHZ/star  | `ghz4 ghz5 ghz6 ghz7 ghz8 ghz9`            |
| path      | `path4 path5 path6 path7 path8 path9`      |
| cycle     | `cycle4 cycle5 cycle6 cycle7 cycle8 cycle9`|
| wheel     | `wheel5 wheel6 wheel7 wheel8 wheel9`       |
| complete  | `complete4 complete5 complete6`            |
| cluster2D | `cluster2x2 cluster2x3`                    |

Wykluczone z zestawu (juz zbenchmarkowane wczesniej):
`two_qutrit`, `ghz3`, `ame43`.

Dodatkowo nowy parser nazw potrafi rozpoznac (i tworzyc graf) dla:
- `cluster3x3` itp. — *parsowalne, ale celowo nie wlaczone* do nocnego
  zestawu, zeby nie powiekszac kosztu nocnego runa bez akceptacji
  uzytkownika.
- `ghz_star_<n>`, `ghz_n_<n>` — backward compat ze starym v2 CLI.

### Pokryte klasy baz kodowania

Jeden run pokrywa wszystkie 17 klas eksponowanych przez
`QuditsOnQubits.benchmark_encoding_bases.ALL_CLASS_NAMES` —
`baseline`, `monomial_old_codespace`, `monomial_full`, `fourier_like`,
`householder_random`, `clifford_wh`, `haar_random_isometry`,
`perturbed_isometry`, `entangling_isometry`, `structured_entangling`,
`product`, `local_ry_only`, `local_general_su2`, `real_orthogonal`,
`near_identity`, `finer_structured`, `two_cz_ansatz`. *Nie* wprowadzilem
zadnej nowej klasy baz kodowania — wszystkie postulowane potencjalne
dodatki (gestszy product grid, perturbacje) sa juz w repo i obejmie je
benchmark `--candidate-mode full`.

---

## 2. Lista plikow

### Nowe pliki

- `QuditsOnQubits/graph_states.py` — generatory krawedzi
  (`star/path/cycle/wheel/complete/cluster_edges`),
  `GraphStateSpec`, `resolve_graph_state(...)`,
  rejestr `BENCHMARK_SUITES` z `EXTENDED_GRAPH_STATES`.
  Wspoldzielone zrodlo prawdy dla obu warstw (`benchmark_encoding_bases`
  i `encoding_search_v2`).
- `encoding_search_v2/suite.py` — jednoetapowy runner
  (`SuiteConfig`, `run_benchmark_suite(...)`), tee-log
  (`stdout` + `suite_run.log`), per-state `_BenchmarkProgressReporter`
  (z dotychczasowego `runner.py`) plus suite-level ETA.
- `encoding_search_v2/suite_cli.py` — CLI: `--suite`, `--states`,
  `--jobs`, `--n-transpile-runs`, `--candidate-mode`,
  `--encoding-strategy`, `--export-circuits`, `--skip-existing`,
  `--dry-run`.
- `tests/test_graph_states.py` — 19 testow generatorow + parsera +
  rejestru.
- `tests/test_suite_runner.py` — 8 testow: layout wynikow, budowa zadan,
  end-to-end runner z mockiem `benchmark_basis`, `--skip-existing`,
  CLI `--dry-run`.
- `docs/milestones/2026-04-30-graph-states-extended-suite.md` (ten plik).

### Pliki zmienione

- `QuditsOnQubits/benchmark_encoding_bases.py`
  - `_resolve_state_spec(...)` deleguje do `graph_states.resolve_graph_state`.
  - `_normalize_state_name`, `_state_family`, `_state_num_qutrits`
    czytaja `GraphStateSpec`.
  - `_get_state_graph` cache'uje po kanonicznym `state_id` (poprzez
    nowy `_build_cached_graph`), dzieki czemu
    `_get_ame43_graph()` i `_build_state_circuit("ame43", ...)` zwracaja
    *te sama* instancje `igraph.Graph` (test
    `test_ame43_reuses_cached_graph_instance` znow zielony).
  - `_build_state_circuit` jest teraz *generyczny* — dziala dla kazdego
    stanu znanego rejestrowi (a nie tylko `star`/`ame43`).
  - Pozostawione zostaly `_resolve_star_graph_n`,
    `_parse_ghz_star_n_from_name`, `_validate_star_n`,
    `_get_ame43_graph` jako legacy helpery (zachowanie back-compat).
- `QuditsOnQubits/project_paths.py`
  - `benchmark_state_slug(...)` rozpoznaje rowniez nowe slug-i
    (`ghz<n>`, `path<n>`, `cycle<n>`, `wheel<n>`, `complete<n>`,
    `cluster<r>x<c>`).
- `encoding_search_v2/states.py`
  - `resolve_benchmark_state` deleguje do `graph_states`. Zachowuje
    pole `state_name="ghz_star"` dla starego API (`--state ghz_star
    --n-qutrits N`).
- `encoding_search_v2/__init__.py`
  - eksportuje `SuiteConfig`, `run_benchmark_suite`.

### Plików, których celowo NIE ruszalem

- `encoding_search_v2/runner.py`, `cli.py` — nadal obsluguja stage1/stage2
  i `ghz_star`/`ame43`/`ghz3`. To inny pipeline.
- `QuditsOnQubits/create_ame_circuit.py` — juz akceptuje dowolny
  `igraph.Graph`, wiec wystarczy podac mu graph z `_get_state_graph`.

---

## 3. Komendy do uruchomienia (smoke + nightly)

> Smoke testy juz uruchomione lokalnie podczas pracy nad milestonem.
> Pozostale ciezkie runy *zostaja do uruchomienia recznie* przez
> uzytkownika.

### Smoke testy (lekkie, juz odpalone)

```powershell
python -m pytest tests/test_graph_states.py            # 19 passed
python -m pytest tests/test_suite_runner.py            # 8 passed
python -m pytest tests/test_encoding_search_v2.py      # 25 passed
python -m pytest tests/test_benchmark_encoding_bases.py::TestBuildStateCircuit
# 5 passed (regresja w cache `_get_ame43_graph` naprawiona)
```

### Dry-run nocnego pipeline'u

```powershell
python -m encoding_search_v2.suite_cli --suite graph_states_extended --jobs 32 --dry-run
```

### Maly real-run sanity check (1 stan, mocno ograniczony)

```powershell
python -m encoding_search_v2.suite_cli `
    --states path4 `
    --candidate-mode original `
    --jobs 4 `
    --n-transpile-runs 2 `
    --output-root data/benchmarks/suites_smoke
```

### Nocny benchmark (wlasciwy run)

```powershell
$env:OMP_NUM_THREADS=1
$env:OPENBLAS_NUM_THREADS=1
$env:MKL_NUM_THREADS=1
python -m encoding_search_v2.suite_cli `
    --suite graph_states_extended `
    --jobs 32 `
    --skip-existing
```

Domyslna lokalizacja wynikow:
`encoding_search_v2/results/graph_states_extended/<state_id>/...`
oraz zbiorczy `encoding_search_v2/results/graph_states_extended/suite_combined_results.csv`
(plus `suite_run.log` z calym progresem).

---

## 4. Kontrakt logow / progresu

Pojedynczy run produkuje na stdout *i* do `suite_run.log` linijki w stylu:

```
[suite graph_states_extended] starting; states=28, candidates_per_state=1503, jobs=32, mode=full, strategy=append_w
[suite graph_states_extended] [1/28] === state ghz4 (family=ghz_star, qutrits=4, edges=3) ===
[suite graph_states_extended :: ghz4] starting benchmarks: 0/1487
[suite graph_states_extended :: ghz4] 1/1487 (  0.1%) done baseline/E_old elapsed=0s avg=0.40s/cand eta=9m48s remaining=1486
...
[suite graph_states_extended :: ghz4] finished 1487/1487 in 12m04s (avg 0.49s/cand)
[suite graph_states_extended] [1/28] finished ghz4 in 12m04s; suite elapsed=12m04s eta=5h37m12s remaining_states=27
...
[suite graph_states_extended] DONE in 6h12m43s; combined CSV: .../suite_combined_results.csv
```

Czyli dostepne sa:
- liczba ukonczonych / pozostalych zadan,
- aktualnie przetwarzany stan,
- `elapsed` ostatniego ukonczonego zadania (per cand i per state),
- per-state `avg/cand` i ETA do konca tego stanu,
- suite-level `elapsed` i ETA do konca calego runa.

---

## 5. Co bylo decyzja architektoniczna (krotka rekapitulacja)

- **Wspoldzielony rejestr `graph_states.py`** — pojedyncze zrodlo
  prawdy. Bez niego musialbym duplikowac logike rozpoznawania nazw w
  `benchmark_encoding_bases` i `encoding_search_v2`.
- **Cache po kanonicznym `state_id`** — naprawia regresje, w ktorej
  `_get_state_graph("ame43")` i `_get_state_graph("ame43", 4)`
  trafialy do innych pozycji `lru_cache`.
- **Petla po stanach + parallel po kandydatach** — prosty model
  obciazenia. Gdyby cos wybuchlo w trakcie nocnego runa, dokladnie
  widac na ktorym stanie, i `--skip-existing` pozwala wznowic.
- **Niezmieniony `runner.py`** — nie chcialem powiekszac scopu i
  ryzykowac zlamania stage1/stage2 dla istniejacych benchmarkow
  `two_qutrit`/`ghz3`/`ame43`.

---

## 6. Wskazowki na wypadek awarii / wskrzeszenia

1. **Sprawdz git diff** wzgledem `485e76e`:

   ```powershell
   git diff 485e76e -- QuditsOnQubits/graph_states.py `
       QuditsOnQubits/benchmark_encoding_bases.py `
       QuditsOnQubits/project_paths.py `
       encoding_search_v2/states.py `
       encoding_search_v2/suite.py `
       encoding_search_v2/suite_cli.py `
       encoding_search_v2/__init__.py `
       tests/test_graph_states.py `
       tests/test_suite_runner.py
   ```

2. **Zainstaluj brakujace deps:**

   ```powershell
   python -m pip install igraph
   ```

   (`qiskit`, `pandas`, `numpy`, `scipy` byly juz zainstalowane.)

3. **Odpal smoke testy** (cala lista z sekcji 3 powyzej).
4. **Jezeli odpalales nocny run** i go przerwales:
   - sprawdz `encoding_search_v2/results/graph_states_extended/suite_run.log`
     — ostatnia linia `[suite ...] [k/28] finished <state_id> ...`
     mowi co ostatnio domknieto;
   - uruchom ponownie z `--skip-existing` — pominie ukonczone stany.

5. **Jezeli `igraph` znow zginie**, calym ekosystemem rzadzi
   `from igraph import Graph` w `create_ame_circuit.py`. Mozna albo
   `python -m pip install igraph`, albo zastapic to `rustworkx`
   (uwaga: bedzie wymagac drobnej zmiany w `create_ame_circuit.py`
   bo nie kasuje API).

---

## 7. Otwarte zadania / TODO po milestonie

- [ ] Pelny nocny run `--suite graph_states_extended --jobs 32` na
      maszynie produkcyjnej.
- [ ] Naprawa zastanego (nie wprowadzonego ode mnie) testu
      `TestProductGenerators::test_single_state_benchmark_can_filter_only_product_class`
      — bug w `_save_top3_per_class_csvs` zaklada kolumny
      (`best_size`), ktorych test fake'em nie podaje. Wymaga osobnego
      patchu poza scopem milestonu.
- [ ] Opcjonalnie: dodac `cluster3x3` do `EXTENDED_GRAPH_STATES`,
      jezeli nocny run pokaze, ze masz na to budzet czasu.
- [ ] Opcjonalnie: dorzucic w `suite.py` zapis krotkiego
      `suite_summary.md` (oprocz `suite_combined_results.csv`).

---

## 8. Szybki "co odpalic w nocy" cheat sheet

```powershell
# 1. terminal 1: srodowisko
$env:OMP_NUM_THREADS=1
$env:OPENBLAS_NUM_THREADS=1
$env:MKL_NUM_THREADS=1

# 2. dry-run sanity
python -m encoding_search_v2.suite_cli --suite graph_states_extended --jobs 32 --dry-run

# 3. lekki single-state smoke (5-10 minut)
python -m encoding_search_v2.suite_cli --states path4 --candidate-mode original --jobs 4 --n-transpile-runs 2 --output-root data/benchmarks/suites_smoke

# 4. wlasciwy nocny run
python -m encoding_search_v2.suite_cli --suite graph_states_extended --jobs 32 --skip-existing

# 5. obserwacja progresu w innym terminalu (Windows / PowerShell)
Get-Content encoding_search_v2/results/graph_states_extended/suite_run.log -Wait -Tail 20
```

> Wyniki:
> `encoding_search_v2/results/graph_states_extended/<state_id>/encoding_search_v2_suite_<state_id>_*.csv`
> + zbiorczy `suite_combined_results.csv`
> + `suite_run.log`.
