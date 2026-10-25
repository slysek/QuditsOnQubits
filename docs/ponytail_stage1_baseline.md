# Ponytail: etap 1 — stan bazowy

Data: 2026-10-24. Commit: `8e3c0e6dd8975e1a31c7160fd2ce6278d1876601`.

Zebrano 3065 testów. W pełnym przebiegu 3051 przeszło, 13 pominięto,
a jeden zakończył się błędem uprawnień sandboxa. Ten sam test przeszedł
poza sandboxem. Łącznie zaliczono 3052 różne przypadki.
Nie stwierdzono błędu kodu wymagającego naprawy w tym etapie.
Wynik testów nie oznacza przeglądu poprawności całego kodu.

## Zakres i zachowanie plików

[Spis plików](ponytail_file_inventory.csv) zawiera 819 plików: wszystkie
śledzone oraz nieignorowane pliki w zakresie na początku etapu.
Każdy ma przypisany etap, stan `czeka` i początkowy SHA-256.
Stan `czeka` oznacza, że przegląd kodu lub danych jeszcze się nie odbył.
Spis obejmuje 133 pliki Python w `src`, 25 notebooków i 135 modułów testowych.

Pełna lista początkowa ma 10 846 plików. Wyłączono 10 027 plików AME46:
trzy katalogi wskazane w planie, testy `test_ame46_*.py`, `tools/CEL.md`,
obraz `ai_scientist/runtime/Dockerfile.codex` i blokadę jego budowania.
`log.txt` pozostaje w zakresie: zawiera błąd uruchomienia procesu,
ale jego związku z AME46 nie potwierdzono.
Spis nie obejmuje ignorowanych środowisk, cache ani nowych wyników lokalnych.
Nie odczytywano plików z poświadczeniami.

| Etap | Pliki |
| --- | ---: |
| 1 | 2 |
| 2 | 11 |
| 3 | 25 |
| 4 | 38 |
| 5 | 39 |
| 6 | 62 |
| 7 | 38 |
| 8 | 29 |
| 9 | 72 |
| 10 | 460 |
| 11 | 43 |

Nowe pliki tego etapu to ten raport, spis CSV i zapis wersji zależności.
Nie należą do początkowego spisu. Uzupełniono również plan wykonania.
Zastane zmiany `.gitignore`, `MANIFEST.in`, narzędzia i eksperymenty użytkownika
pozostały bez zmian. Nie tworzono commitów ani PR.

## Środowisko

Utworzono osobne `.venv-ponytail`. Początkowe `.venv` pozostało bez zmian.
Jego pytest 9.1.1 i pytest-cov 7.1.0 wykraczały poza deklarowane zakresy.
Brakowało również części zależności projektu oraz pakietu `build`.

| Składnik pomiaru | Wersja |
| --- | --- |
| Python | 3.12.15 |
| pytest / pytest-cov | 8.4.2 / 6.3.0 |
| build | 1.6.1 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| Qiskit / Aer | 2.1.2 / 0.17.2 |
| BQSKit | 1.2.1 |
| IBM Runtime | 0.44.0 |
| IQM Client / Qubit Selector | 35.0.3 / 1.1.3 |

Wszystkie bezpośrednie zależności podstawowe i `dev` mieszczą się w zakresach
`pyproject.toml`. `uv pip check` potwierdził zgodność 171 zainstalowanych pakietów.
[Zapis wersji](ponytail_stage1_requirements.txt) zawiera także zależności pośrednie.
Nie zmieniono deklaracji zależności projektu.

Zastane problemy przygotowania:

- Pobieranie pakietów wymagało dostępu do sieci poza sandboxem.
- Budowanie `ibm-platform-services==0.55.2` zgłosiło brak `pkg_resources`.
  Instalacja zależności przeszła z ograniczeniem wersji narzędzia budowania:
  `setuptools>=77,<81`.
- Budowanie metadanych instalacji editable przerwano po kilku minutach.
  Nie ustalono przyczyny. W `artifacts/` jest 336 801 plików, także ignorowanych.
  Zatrzymano wyłącznie dwa procesy budowania tego zadania, w tym proces
  osierocony po restarcie serwera.

Zainstalowano zależności z `pyproject.toml`, bez instalacji samego projektu.
Testy korzystały z bieżącego źródła przez `PYTHONPATH=src`.
Nie jest to walidacja instalacji paczki. Editable, wheel i sdist pozostają
do sprawdzenia w etapie 11.

## Testy i czasy

| Kontrola | Wynik | Czas pytest / czas całego procesu |
| --- | --- | --- |
| Zbieranie | 3065 testów, 135 modułów, bez błędów importu | 5,76 s / 7,15 s |
| Zestaw kontrolny CI | 26 zaliczonych | 7,13 s / 9,84 s |
| Pełny zestaw offline w sandboxie | 3051 zaliczonych, 13 pominiętych, 1 błąd środowiska | 138,26 s / 142,72 s |
| Powtórka testu BQSKit poza sandboxem | 1 zaliczony | 2,48 s / 2,72 s |

Błąd `test_two_real_bqskit_runtimes_can_coexist` wynikał z odmowy utworzenia
gniazda: `PermissionError: [Errno 1] Operation not permitted`.
Powtórka tego samego testu poza sandboxem potwierdziła współpracę dwóch
rzeczywistych procesów BQSKit. Nie powtarzano pozostałych zaliczonych testów.

Pełny przebieg obejmował 66 testów historycznego benchmarku oraz 55 testów
nowego, nieśledzonego `agent_relay`. Wszystkie przeszły.
Testy relay korzystają z lokalnych atrap CLI, bez wywołań modeli.
Odtworzenie archiwum potwierdziło 43 zakończone zadania i 240 wierszy wyników,
zgodność opublikowanych wartości oraz ochronę istniejącego katalogu wyników.
Ten test nie przelicza bootstrapu.

Pominięcia:

- 10 przypadków wymaga opcjonalnego, prywatnego `cft_piastq`.
- 1 przypadek wymaga `QOQ_RUN_BQSKIT_TESTS=1`: rzeczywista kosztowna synteza CZ3.
- 1 przypadek wymaga `QOQ_RUN_IQM_HARDWARE=1`: wykonanie na QPU.
- 1 przypadek wymaga `QOQ_RUN_IQM_SELECTOR_SMOKE=1`: selektor i kompilacja z usługą IQM.

Pełny przebieg zgłosił 1737 ostrzeżeń deprecacji `IBMFractionalTranslationPlugin`.
Nie wyciszano ostrzeżeń ani nie zmieniano kodu dostawców.
Najwolniejsze przypadki to odtworzenie archiwum (6,41 s), równoległy harness
IQM (6,17 s) oraz eksport 33 kątów szablonu theta (3,84 s).

## Odtworzenie

Polecenia uruchamiaj z katalogu repo. Użyj osobnego środowiska Python 3.12.
Zapis wersji odpowiada temu pomiarowi na Linux, nie zastępuje zakresów projektu.

```bash
uv --cache-dir /tmp/qoq-ponytail-uv-cache venv --python 3.12 .venv-ponytail
printf '%s\n' 'setuptools>=77,<81' > /tmp/qoq-ponytail-build-constraints.txt
uv --cache-dir /tmp/qoq-ponytail-uv-cache pip install \
  --python .venv-ponytail/bin/python \
  --build-constraints /tmp/qoq-ponytail-build-constraints.txt \
  -r docs/ponytail_stage1_requirements.txt
uv --cache-dir /tmp/qoq-ponytail-uv-cache pip check --python .venv-ponytail/bin/python

unset QOQ_RUN_IQM_HARDWARE QOQ_RUN_IQM_SELECTOR_SMOKE QOQ_RUN_BQSKIT_TESTS
export PYTHONPATH=src MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export QISKIT_PARALLEL=FALSE RAYON_NUM_THREADS=1

.venv-ponytail/bin/python -m pytest --collect-only -q tests benchmarks/bell_20260907 \
  --ignore-glob='tests/test_ame46_*.py'
.venv-ponytail/bin/python -m pytest -q tests/test_clean_repo_smoke.py \
  tests/test_experiment_aer_integration.py tests/test_reference_regressions.py
.venv-ponytail/bin/python -m pytest -q tests benchmarks/bell_20260907 \
  --ignore-glob='tests/test_ame46_*.py' --durations=25 -ra
```

Pełne polecenie wymaga zgody środowiska na lokalne gniazda BQSKit.
Jeżeli sandbox je blokuje, uruchom sam test `tests/test_bqskit_parallel_runtime.py`
w środowisku z dostępem do localhost. Nie włączaj trzech flag opt-in.

Surowe logi, JUnit XML, pomiary `/usr/bin/time`, wersje, początkowy stan Git,
diff użytkownika i pełny spis wyłączeń zapisano lokalnie w
`artifacts/ponytail/stage1-20261024/`. Ten katalog jest ignorowany przez Git.

## Review Ponytail

Niezależny reviewer sprawdził pełny diff planu, raport, wszystkie wiersze spisu,
wyłączenia, wersje zależności oraz zgodność dowodów z testami i konfiguracją.
Przyjęte użycie: jeden badacz uruchamia lokalny zestaw testów.
Wynik: `CLEAN`, bez usterek do poprawy. Nie powtarzano testów ani instalacji.
Ten przegląd dotyczył wyniku etapu 1, nie całego kodu repo.

## Następny mały pakiet: 2a

Przegląd dostępu do zasobów QPY: `quantum_circuits_resource`,
`quantum_circuits_path` i ich użycie w `create_ame_circuit` oraz `generate_b_ame`.
Graphify wskazał te zależności, ale nie wskazał testów zasobów.
Lokalnie istnieją cztery przypadki w `tests/test_package_data.py` i testy
`tests/test_clean_repo_smoke.py`. Przeszły w przebiegu bazowym.

Najpierw sprawdzimy ładowanie zasobów poza katalogiem repo, zgodność publicznych
wywołań oraz niezależne własności stanów. Funkcje ścieżek starego benchmarku
pozostają przypisane do etapu 6. Kod czytelny i potrzebny może pozostać bez zmian.
Nie rozpoczynano jeszcze refaktoryzacji 2a.

Nie sprawdzono QPU, usług dostawców, prywatnego PiastQ, pełnej syntezy CZ3,
bootstrapu archiwum, macierzy systemów/Pythonów, instalacji paczki ani pokrycia kodu.
