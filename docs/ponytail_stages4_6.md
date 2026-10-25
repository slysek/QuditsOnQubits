# Ponytail: etapy 4–6

Data: 2026-10-25. Stan odniesienia: `8e3c0e6dd8975e1a31c7160fd2ce6278d1876601`.

Sprawdzono przypisane źródła, skrypty, eksporty, grupy testów i kontrakty artefaktów.
Mapa decyzji obejmuje 139 plików początkowego spisu oraz cztery nowe moduły testowe:
`artifacts/ponytail/stages4_6_decisions.json`. Każda ścieżka ma decyzję i uzasadnienie.
Lista nieprzeczytanych plików źródłowych tego zakresu jest pusta.
Testy oceniono jako grupy chronionych zachowań. Nie usuwano istniejących testów.

## Zmiany

| Obszar | Zmiana | Dowód |
| --- | --- | --- |
| Walidacja macierzy | `direct_basis/math_utils.py` ustawia `rtol=0.0` dla ortogonalności i tożsamości leakage. Domyślne `rtol` NumPy przepuszczało błąd `1e-7` przy zadanym `tol=1e-10`. | Przed poprawką: 7 błędów, 2 zaliczone. Po poprawce: 9/9. Są przypadki dodatniego i ujemnego błędu normy oraz błąd fazy leakage. |
| Wznowienie suite | Nie zastępujemy uszkodzonego CSV pustą ramką. Błąd odczytu zatrzymuje przebieg przed nadpisaniem wyników łącznych. Usunięto funkcję kopiującą wyłącznie listę kandydatów. | Regresja ochrony zapisanego pliku nie przechodziła przed zmianą. Po zmianie przechodzi, podobnie jak rzeczywisty mały benchmark i jego wznowienie bez kompilacji. |
| Baseline | Usunięto martwą kontrolę W ze starym importem `from encoding_change_unitary`. Bezpośrednie porównanie E oraz porównanie z fazą globalną wystarczają do decyzji. | Porównanie z kodem HEAD: identyczne metadane wszystkich 713 domyślnych kandydatów. Osobne testy izometrii oraz fazy globalnej i względnej. |
| Nazwy plików | Główny benchmark i oba harnessy importują istniejące `selection.safe_path_part` zamiast trzech identycznych kopii. Prywatna nazwa importu pozostaje dostępna. | Testy eksportów, nazw QPY, obu harnessów i procesu potomnego przechodzą. |
| Raport legacy | Usunięto pierwszą definicję `_save_top3_per_class_csvs`, którą druga definicja zawsze przesłaniała. Aktywna funkcja pozostała bez zmian. | Nowy test rzeczywistych CSV przeszedł przed i po usunięciu. Sprawdza kolejność depth/2q i kolumn fid095, odrzucenie błędnych wierszy oraz identyczny alias exact. Nie sprawdza odrzucenia fidelity poniżej progu. |
| Konwerter QPY | Konwersja zapisuje plik tymczasowy na tym samym dysku. Dopiero poprawna serializacja zastępuje cel. Błąd wersji nie usuwa istniejącego QPY. | Przed poprawką nieobsługiwana wersja pozostawiała pusty plik źródłowy przy konwersji w miejscu. Po poprawce przechodzą trzy testy: ochrona oryginału, wybór katalogów i rzeczywisty CLI z QPY 13. |

Łączna zmiana ośmiu plików produkcyjnych: **135 linii mniej**.
Nie dodano zależności ani nowych opcji API lub CLI.
Nie zmieniono danych źródłowych, QPY ani NPY.

## Decyzje dla pozostałego kodu

- Etap 4: zachowano rozdział kandydatów, rodzin, logicznego workloadu, syntezy i targetu.
  Główny benchmark korzysta już ze wspólnego Pareto oraz store theta.
  Kanoniczny kandydat ma ten sam budżet kompilacji w obrębie protokołu.
  Zapisane bramki wymagają weryfikacji hashy i jawnego powiązania pełnego przygotowania z workloadem.
  Zachowano historyczne interfejsy dwóch etapów i suite.
- Etap 5: pozostawiono algorytmy theta bez zmian. Kontynuacja sprawdza dziewięć kolumn przestrzeni kodowej z jedną fazą.
  Ograniczona rekursja i odzyskiwanie surowych fallbacków zachowują historyczną gałąź.
  Kompilacja kontroluje cały operator i ma dokładną konwersję U3 do U, gdy transpiler usuwa małe rotacje.
  Warstwa offline zachowuje permutacje, topologię i wszystkie histogramy Bella.
- Etap 6: zachowano odrębne scenariusze direct-basis i legacy, formaty CSV oraz adaptery dostawców.
  Zachowano fazy stanów, kolejność par kubitów, śledzenie dodatkowych kubitów i kontrolę ISA.
  Zachowano procesy IQM, blokady BQSKit, atomową publikację cache i krótkie nazwy QPY Piast.
  Nie łączono przepływów o różnych kryteriach wyboru i kontraktach błędów.

Przejrzano grupy testów macierzy i stanów, seedów i granic Pareto, targetów dostawców,
cache i serializacji, raportów, kontynuacji i wznowień, wyboru kandydatów oraz argumentów CLI.
Zachowano testy rzeczywistego Qiskit i artefaktów obok atrap usług.
Kontrole prywatnych funkcji pozostają potrzebne tam, gdzie sprawdzają osobne kryterium matematyczne,
ranking albo ochronę danych. Brak zmiany czytelnego kodu jest celową decyzją.

## Walidacja

Użyto `.venv-ponytail/bin/python` i środowiska etapu 1:
`PYTHONPATH=src`, `MPLBACKEND=Agg`, `MPLCONFIGDIR=/tmp/qoq-ponytail-matplotlib`,
po jednym wątku BLAS/OMP/Rayon oraz `QISKIT_PARALLEL=FALSE`.
Usunięto z otoczenia flagi opt-in sprzętu, selektora IQM i pełnej syntezy BQSKit.

```bash
.venv-ponytail/bin/python -m pytest -q \
  tests/test_benchmark*.py tests/test_theta*.py tests/test_direct_basis*.py \
  tests/test_harness_progress.py tests/test_iqm_harness_parallel.py \
  tests/test_convert_selected_qpy.py tests/test_encoding_benchmark_notebook.py \
  tests/test_encoding_search_candidates.py tests/test_encoding_search_suite.py \
  tests/test_legacy_encoding_reports.py \
  --cov=src/qudits_on_qubits --cov=scripts --durations=10 -ra
```

Końcowy szeroki przebieg: **1252 zaliczone, 1 pominięty**, 127,55 s.
Zgłoszono 1524 zastane ostrzeżenia deprecacji pluginu IBM.
Po dodaniu testu CLI konwertera jego zestaw: **3 zaliczone**, 0,52 s.
Nowy przypadek CLI podnosi liczbę różnych zaliczonych przypadków do 1253.
`git diff --check` nie zgłasza błędów.
Jedna wcześniejsza próba coverage z nazwą modułu zgłosiła `ImportError: cannot load module more than once per process` przy imporcie NumPy.
Końcowy pomiar używał ścieżek źródłowych i przeszedł bez tego błędu.

| Zmieniony obszar | Pokrycie instrukcji |
| --- | ---: |
| `direct_basis/math_utils.py` | 132/165 — 80% |
| `encoding_search/suite.py` | 155/176 — 88% |
| `encoding_search/triviality.py` | 65/70 — 93% |
| `direct_basis/benchmark.py` | 517/556 — 93% |
| Harness IQM | 325/349 — 93% |
| Harness Piast | 281/309 — 91% |
| Konwerter QPY, po teście CLI | 47/50 — 94% |
| Aktywna funkcja raportu top-3 legacy | 48/53 — 91% |

Cały stary `benchmark_encoding_bases.py` ma 532/953 instrukcji pokrytych, czyli 56%.
Zmiana tego pliku wyłącznie usuwała nieosiągalną definicję.
Nie dodawano pełnych kampanii legacy tylko dla podniesienia procentu modułu.
Brakuje przede wszystkim wykonania wszystkich historycznych trybów i kombinacji eksportu.

Testy obejmują mały główny benchmark od logicznego wejścia do raportu i ponownego odczytu,
zapisane bramki theta, całe operatory F3/CZ3, niezależne histogramy Bella oraz procesy potomne IQM.
Hashe zasobów `theta_continuation/data/CZ3_W.qpy` i `E.npy` zgadzają się ze spisem początkowym.
Nieprzerwana pełna synteza CZ3 pozostaje osobnym testem opt-in.

Logi i pełne dane coverage zapisano w `artifacts/ponytail/stages4-6-20261025/`.
Osobny `COVERAGE_FILE` zachował wcześniejszy plik `.coverage` użytkownika.

## Ograniczenia

Nie uruchamiano QPU, usług IQM/IBM/PCSS, prywatnego PiastQ ani nowych kampanii sprzętowych.
Nie wykonano kosztownej rzeczywistej syntezy CZ3. Nie oceniano optimum globalnego syntezy ani routingu.
Test współistnienia dwóch rzeczywistych runtime BQSKit wymaga lokalnych gniazd i należy do odrębnej walidacji całości.
Nie uruchamiano automatycznego review. Nie tworzono commitów ani PR.
Pakowanie, instalację poza checkoutem, pełny zestaw offline i archiwum Bella domyka etap integracji.
