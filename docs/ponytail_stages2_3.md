# Ponytail: etapy 2d–3 — core i Bell

Data: 2026-10-25. Stan odniesienia: `8e3c0e6dd8975e1a31c7160fd2ce6278d1876601`.

Przeczytano 25 przypisanych plików źródłowych i 13 istniejących modułów testowych.
Dodano trzy moduły regresji. Wszystkie 41 plików mają decyzję w tabeli poniżej.
Wcześniejsze zmiany 2a–2c pozostają zachowane i opisane w głównym planie.
`core/benchmark_encoding_bases*.py` należy do etapu 6.
Nie zmieniono AME46, danych pomiarowych, QPY ani zapisanych modeli naukowych.

## Poprawki i uproszczenia

- `is_known_graph_state` odrzuca nazwy z niedopuszczalnym rozmiarem grafu.
  Parametryczne nazwy `ghz_star` i `ghz_n` nadal są rozpoznawane bez podania liczby stron.
  Generatory krawędzi, poprawne aliasy, identyfikatory i kolejność krawędzi pozostają zgodne.
  Skrócono opis modułu i usunięto stare nazwy pakietów.
- `generate_b_ame` wybiera drugiego uczestnika korzenia spośród sąsiadów pierwszego.
  Liczy różnych sąsiadów, a nie powtórzone krawędzie.
  Naprawiono użycie `.index` na liczbie całkowitej, indeksowanie niekolejnych numerów wierzchołków,
  oraz przypisanie stabilizatorów i współczynników do grup wag.
  Niepełna albo nadmiarowa lista wyników pomiarowych wywołuje błąd.
  Nieobsługiwany wymiar lub znak operatora także wywołuje błąd.
- `th_values` liczy iloczyn wewnętrzny stanu i jego ewolucji. Nie tworzy gęstych macierzy całych stabilizatorów.
  Zachowano limit 18 kubitów, `MemoryLimitExceeded`, pustą listę wyników i zaokrąglenie do czterech miejsc.
  Przed każdą ewolucją powstaje kopia stanu: Qiskit 2.1 stosuje fazę obwodu na płytkiej kopii.
  Bez jawnej kopii kolejne wyniki zmieniałyby stan odniesienia.
- Walidacja izometrii, hermitowskości i unitarności używa `rtol=0`.
  Domyślna tolerancja względna wcześniej przepuszczała błąd rzędu `1e-6`, mimo `atol=1e-10`.
  Baza pomiarowa odrzuca nieunitarny obserwabl, zamiast traktować zmianę modułu wartości własnych jak fazę globalną.
- Generatory obwodów odrzucają `inplace=True` dla więcej niż jednego ustawienia, zanim zmienią wejście.
  Dotychczas każde ustawienie dopisywało bramki do tego samego obwodu, a lista zawierała wielokrotnie ten sam obiekt.
  Pojedyncze jawne ustawienie zachowuje działanie `inplace=True`.
- Metadane obwodu zawierają globalne indeksy bitów każdego rejestru klasycznego.
  Dekoder SamplerV2 scala skorelowane wyniki przez `join_data`, a następnie używa `slice_bits` do globalnej kolejności.
  Obsługuje także wcześniejszy rejestr o permutowanych bitach.
  Bez proweniencji wielu rejestrów wywołuje błąd zamiast wybierać pierwszy rejestr.
  Ścieżka backendowego `get_counts` zachowuje dotychczasowy kontrakt.

Celowe zaostrzenia kontraktu dotyczą niepoprawnych danych, niepełnych wyników i wieloustawieniowego `inplace=True`.
Nie dodano zależności ani nowego interfejsu CLI.
Nie usuwano publicznych klas, eksportów ani potrzebnych helperów wyłącznie z powodu ich rozmiaru.
Nie powtarzano oszczędności pakietu 2a przy opisie bieżącego uproszczenia.

## Ocena testów i dowody

Nowy `tests/test_graph_states.py` ma 38 przypadków. Oczekiwane krawędzie zapisano jawnie.
Testy chronią minimalne rozmiary, poprawne aliasy, konflikty liczby stron i rejestr zestawów.
Przed poprawką pięć nazw błędnych grafów uznawano za poprawne.

Nowy `tests/test_generate_b_ame.py` ma 17 przypadków.
Pierwsze 13 regresji wykazało błędy przed zmianą kodu.
Dodatkowo sprawdzono niezależne współczynniki ważonego grafu, stabilizatory ścieżki dla wymiarów 3 i 4,
pełność wyników oraz limit kubitów.
Test 12 kubitów odrzuca budowanie gęstej macierzy całego obwodu, ale dopuszcza małe macierze lokalnych bramek.
Sprawdza także fazę globalną i powtórną ocenę tego samego stabilizatora.

Nowy `tests/test_bell_measurement_regressions.py` ma 38 przypadków.
Oczekiwane wartości Bella wynoszą 6, 6 i 8 dla `two_qutrit`, `ghz3` i `ame43`.
Stany do porównania wynikają bezpośrednio z faz grafu `exp(2πi sum(w*a_u*a_v)/3)`.
Sprawdzono kodowanie kanoniczne i gęste zespolone kodowanie, leakage 0 oraz 0.125, i oba mianowniki warunkowania.
Osobne przypadki sprawdzają odwrotną transformację do operatorów Weyla dla wszystkich sześciu par ustawienia i potęgi.
Rzeczywiste StatevectorSampler i Aer sprawdzają wyniki bitowe, dwa rejestry i ich permutację.

Istniejące testy pozostają. Ich role opisano oddzielnie w tabeli decyzji.
Testy dokumentacji i wywołań adapterów nie stanowią dowodu poprawności naukowej.
Uzupełniają je porównania analityczne, pełna enumeracja granic klasycznych oraz rzeczywiste lokalne samplery.
Nie uruchamiano agentów review.

## Walidacja

Końcowy zestaw zależny: **359 zaliczonych, 1 pominięty**, 12 zastanych ostrzeżeń IBM.
Czas pytest z pomiarem pokrycia: **27.14 s**.
Pominięcie dotyczy rzeczywistego klienta `cft-piastq`, którego nie ma w środowisku.
Przegląd diffu i `git diff --check` nie wykazały błędów białych znaków.

Powtórzenie, z katalogu repo:

```bash
export PYTHONPATH=src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/qoq-ponytail-matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QISKIT_PARALLEL=FALSE RAYON_NUM_THREADS=1
export COVERAGE_FILE=/tmp/qoq-core-bell.coverage
unset QOQ_RUN_IQM_HARDWARE QOQ_RUN_IQM_SELECTOR_SMOKE QOQ_RUN_BQSKIT_TESTS
.venv-ponytail/bin/python -m pytest -q \
  tests/test_graph_states.py tests/test_generate_b_ame.py \
  tests/test_bell_measurement_regressions.py tests/test_create_ame_circuit.py \
  tests/test_encoding_change_unitary.py tests/test_package_data.py \
  tests/test_reference_experiments.py tests/test_reference_measurement_integration.py \
  tests/test_reference_regressions.py tests/test_bell_identity_spectator.py \
  tests/test_bell_postprocessing_weights.py tests/test_bell_measurements_iqm_runner.py \
  tests/test_bell_measurements_piastq_runner.py tests/test_clean_repo_smoke.py \
  tests/test_public_api.py tests/test_randomized_bell_preparation.py \
  tests/test_experiment_preparation.py tests/test_experiment_aer_integration.py \
  tests/test_bell_circuit_serialization.py tests/test_bell_replay_runtime_options.py \
  --cov=src/qudits_on_qubits \
  --cov-report=json:/tmp/qoq-core-bell-coverage.json --cov-report= -ra
```

Pierwszy pomiar z wieloma nazwami modułów w `--cov` przerwał import NumPy.
Pomiar ze źródłowym katalogiem w `--cov` przeszedł. Nie zmieniano zależności środowiska.
Dane pomiaru i mapa decyzji są w `artifacts/ponytail/stages2_3_coverage.json`
oraz `artifacts/ponytail/stages2_3_decisions.json`.

Pokrycie instrukcji zmienionych modułów w tym zestawie:

| Moduł | Wykonane instrukcje | Pokrycie |
| --- | ---: | ---: |
| `src/qudits_on_qubits/core/graph_states.py` | 131/131 | 100.00% |
| `src/qudits_on_qubits/core/generate_b_ame.py` | 335/383 | 87.47% |
| `src/qudits_on_qubits/bell_functionals/encoding.py` | 69/81 | 85.19% |
| `src/qudits_on_qubits/bell_functionals/operators.py` | 41/45 | 91.11% |
| `src/qudits_on_qubits/bell_measurements/basis.py` | 135/163 | 82.82% |
| `src/qudits_on_qubits/bell_measurements/qiskit_measurements.py` | 86/100 | 86.00% |
| `src/qudits_on_qubits/bell_measurements/sampler_circuits.py` | 231/286 | 80.77% |

To pomiar instrukcji, nie wszystkich gałęzi ani całego repo.
Niepokryte ścieżki obejmują błędy braku pamięci i importu opcjonalnych bibliotek oraz część błędnych wejść.

## Decyzje dla wszystkich przypisanych plików

| Plik | Decyzja i uzasadnienie |
| --- | --- |
| `src/qudits_on_qubits/bell_functionals/__init__.py` | sprawdzony bez zmian: eksporty konstruktorów i kodowania pozostają zgodne. |
| `src/qudits_on_qubits/bell_functionals/bell_builders.py` | sprawdzony bez zmian: adapter rejestru zachowuje współczynniki, potęgi i kolejność tensorów. |
| `src/qudits_on_qubits/bell_functionals/classical_bounds.py` | sprawdzony bez zmian: pełna enumeracja klasyczna potwierdza zapisane granice. Cache ma istniejące uzasadnienie kosztowe. |
| `src/qudits_on_qubits/bell_functionals/cli.py` | sprawdzony bez zmian: istniejące argumenty i format wyniku pozostają zgodne. |
| `src/qudits_on_qubits/bell_functionals/encoding.py` | uproszczony: sprawdzanie izometrii używa wyłącznie zadanej tolerancji absolutnej. |
| `src/qudits_on_qubits/bell_functionals/estimator_backend.py` | sprawdzony bez zmian: zachowano rozdzielenie części hermitowskiej i antyhermitowskiej oraz dokładny fallback Statevectora. |
| `src/qudits_on_qubits/bell_functionals/operators.py` | uproszczony: testy hermitowskości i unitarności używają tolerancji absolutnej. Niezależnie sprawdzono odwrotną transformację do operatorów Weyla. |
| `src/qudits_on_qubits/bell_functionals/sampler_backend.py` | sprawdzony bez zmian: projektory obsługują gęste kodowania, pełny leakage, warunkowanie i kontrolowane próbkowanie. |
| `src/qudits_on_qubits/bell_measurements/__init__.py` | sprawdzony bez zmian: publiczne eksporty zachowano. |
| `src/qudits_on_qubits/bell_measurements/basis.py` | uproszczony: tolerancje izometrii i unitarności są absolutne. Nieunitarny obserwabl jest odrzucany przed rozkładem spektralnym. |
| `src/qudits_on_qubits/bell_measurements/graph_settings.py` | sprawdzony bez zmian: zachowano ustawienia, potęgi, wagi oraz specjalny przypadek rejestru two_qutrit. |
| `src/qudits_on_qubits/bell_measurements/piastq_runner.py` | sprawdzony bez zmian: kontrakty jednego zadania, kolejności ustawień, liczby wyników i opcjonalnej zależności mają osobne testy. |
| `src/qudits_on_qubits/bell_measurements/postprocessing.py` | sprawdzony bez zmian: podpisane wagi, mianowniki, mapowanie bitów, zerowe zbiory i obie polityki leakage są potrzebne. |
| `src/qudits_on_qubits/bell_measurements/qiskit_measurements.py` | uproszczony: walidacja izometrii jest absolutna. Metadane zawierają globalne indeksy bitów każdego rejestru do poprawnego wspólnego dekodowania. |
| `src/qudits_on_qubits/bell_measurements/sampler_circuits.py` | uproszczony: wieloustawieniowe inplace=True jest odrzucane przed mutacją. Sampler scala skorelowane wyniki rejestrów i przywraca globalną kolejność bitów. Bez proweniencji nie zgaduje wielu rejestrów. |
| `src/qudits_on_qubits/core/__init__.py` | sprawdzony bez zmian: leniwe eksporty zachowują publiczny interfejs i nie wymagają nowych wrapperów. |
| `src/qudits_on_qubits/core/create_ame_circuit.py` | uproszczony w 2a i 2c: zachowano wcześniejsze poprawki i niezależne testy faz stanów, baz oraz kodowań. |
| `src/qudits_on_qubits/core/draw_graph.py` | sprawdzony bez zmian: mały helper rysowania z dotychczasową numeracją wierzchołków. |
| `src/qudits_on_qubits/core/encoding_change_unitary.py` | uproszczony w 2b: zachowano walidację absolutnej tolerancji, dopełnienie SVD i publiczny raport. |
| `src/qudits_on_qubits/core/generate_b_ame.py` | uproszczony: poprawiono wybór krawędzi korzenia, indeksy stabilizatorów, grupy współczynników i unikalność sąsiadów. Ewolucja stanu zastępuje gęste macierze. Wyniki wymagają pełnej liczby pomiarów. |
| `src/qudits_on_qubits/core/graph_states.py` | uproszczony: błędny rozmiar grafu nie jest już rozpoznanym stanem. Zachowano aliasy, generatory i kolejność krawędzi. Usunięto nieaktualny opis ścieżek. |
| `src/qudits_on_qubits/core/prepare_op_to_ibm.py` | sprawdzony bez zmian: hermitowska część stabilizatora i apply_layout realizują istniejącą ścieżkę Estimatora. |
| `src/qudits_on_qubits/core/project_paths.py` | sprawdzony bez zmian: zasoby działają przez importlib.resources. Odbiorcy helperów starego benchmarku należą także do etapu 6. |
| `src/qudits_on_qubits/core/quditsonqubits.py` | sprawdzony bez zmian: mała klasa zgodności pozostaje publicznym eksportem. |
| `src/qudits_on_qubits/reference_experiments.py` | sprawdzony bez zmian: niezmienny rejestr, walidacja modeli, kanoniczne hashe i wartości Bella pozostają wspólnym kontraktem. |
| `tests/test_bell_circuit_serialization.py` | sprawdzony bez zmian: chroni mapowanie przewodów, sterowanie warunkowe, layout i zapis QPY rzeczywistego obwodu IQM offline. |
| `tests/test_bell_identity_spectator.py` | sprawdzony bez zmian: chroni odczyt leakage dla wszystkich nieużywanych poziomów i gęstego kodowania bez zbędnych bramek w kodowaniu kanonicznym. |
| `tests/test_bell_measurement_regressions.py` | nowy: 38 przypadków chroni absolutne tolerancje, operatory Weyla, inplace, gęste kodowania, leakage, realne wyniki samplerów i permutowane rejestry. |
| `tests/test_bell_measurements_iqm_runner.py` | sprawdzony bez zmian: atrapy chronią brak niezamówionej transpilacji i przekazanie opcji dostawcy. Lokalne rzeczywiste wyniki sprawdzają nowe regresje. |
| `tests/test_bell_measurements_piastq_runner.py` | sprawdzony bez zmian: zachowano testy kontraktów zadania i opcjonalnego klienta. Rzeczywisty klient wymaga osobnej zależności. |
| `tests/test_bell_postprocessing_weights.py` | sprawdzony bez zmian: każdy wariant chroni inną regułę wag, mianownika, mutowalności lub błędnych danych. |
| `tests/test_bell_replay_runtime_options.py` | sprawdzony bez zmian: chroni lokalną konfigurację konta i brak ujawniania sekretów w błędach. |
| `tests/test_clean_repo_smoke.py` | sprawdzony bez zmian: sprawdza importy, ścieżki artefaktów, dokumentację wejściową i brak prywatnych adresów. Nie służy jako dowód poprawności naukowej. |
| `tests/test_create_ame_circuit.py` | sprawdzony bez zmian: niezależne fazy grafu, tensorowe kodowania i bazy oraz błędne wejścia chronią wcześniejszy pakiet 2c. |
| `tests/test_encoding_change_unitary.py` | sprawdzony bez zmian: geometria izometrii, absolutne tolerancje, kopie wejść i błędne dane chronią pakiet 2b. |
| `tests/test_generate_b_ame.py` | nowy: 17 przypadków chroni znalezione awarie korzenia, indeksowania, grup wag, liczby wyników i pamięci. Fazy globalne pozostają istotne. |
| `tests/test_graph_states.py` | nowy: 38 przypadków sprawdza konkretne krawędzie, minimalne rozmiary, nazwy, aliasy, konflikty i rejestr zestawów. |
| `tests/test_package_data.py` | sprawdzony bez zmian: odczyt QPY poza checkoutem i pełne macierze X/Z/adjointów chronią pakiet 2a. |
| `tests/test_reference_experiments.py` | sprawdzony bez zmian: wartości analityczne 6/6/8, sumy prawdopodobieństw, enumeracja klasyczna i modele rejestru mają różne zadania. Testy adapterów zachowują zgodność publiczną. |
| `tests/test_reference_measurement_integration.py` | sprawdzony bez zmian: konwersja rejestru, mapy wyników i conditional/unconditional leakage pozostają kontraktem między modułami. |
| `tests/test_reference_regressions.py` | sprawdzony bez zmian: wartości 6/6/8, sprzężone składniki, próbkowanie i znane leakage mają niezależne oczekiwane wyniki. |

## Ograniczenia

Nie wykonywano QPU, usług IQM/IBM/PiastQ ani kosztownej syntezy BQSKit.
Nie dowodzono nierówności Bella dla dowolnego grafu generowanego przez ogólną funkcję metadanych.
Sprawdzenia naukowe dotyczą trzech wspieranych stanów referencyjnych i opisanych stabilizatorów.
Dokładny fallback Estimatora i projektorowy sampler zachowują istniejące zasady API.
Przekazanie obiektu samplera do projektorowej ścieżki nadal nie zleca zadania sprzętowego, zgodnie z jej opisem.
Walidacja pozostałych etapów, całego zestawu offline i dystrybucji należy do integracji głównego agenta.
