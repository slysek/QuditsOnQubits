# Benchmark E_theta: F3 i kontynuacja CZ3

Benchmark bada rodzinę izometrii

```text
E_theta = [[ cos(theta), 0, 0],
           [          0, 1, 0],
           [          0, 0, 1],
           [-sin(theta), 0, 0]]
l_theta = [sin(theta), 0, 0, cos(theta)]
```

w zakresie `0 <= theta <= pi/4`. Wszystkie kutryty danego punktu mają to samo kodowanie. Obliczenia są lokalne; CLI nie uruchamia zadań na QPU.

F3 otrzymuje swobodną fazę leakage w rozszerzeniu `E_theta F3 E_theta† + exp(i alpha)|l_theta><l_theta|`. Istniejący algorytm dobiera `alpha`; wynik musi używać najwyżej dwóch kubitowych CZ oraz spełniać tolerancje działania i leakage. Osobno zapisywany jest koszt kontrolny przy `alpha = 0`.

CZ3 zaczyna od przypiętego obwodu `experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy`. Jego SHA-256 wynosi `a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791`; plik sąsiedni `E.npy` musi zawierać `eye(4, 3)`. Oryginał ma sześć CZ i jedenaście U3, czyli 33 parametry kątowe. Nie dodajemy bramek ani połączeń. Zachowujemy kolejność kubitów oraz fazę globalną.

## Uruchamianie

Polecenia PowerShell wykonuj z katalogu repozytorium lub worktree, w którym znajduje się istniejące `.venv`. Skrypt sam dodaje lokalny `src` względem własnej ścieżki; jawny `PYTHONPATH` przydaje się także przy korzystaniu z API i testów.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --help
```

Pilotaż trzech pierwszych punktów docelowej siatki 41 punktów, przy pełnych ustawieniach numerycznych:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --mode gates --theta-points 41 --limit-points 3 --output-dir artifacts/theta_pilot
```

`--limit-points 3` wybiera prefiks siatki 41 punktów; nie zmienia jej na trzy punkty rozciągnięte do `pi/4`. Jest częścią niezmiennej konfiguracji. Pilotaż i pełny przebieg zapisujemy w oddzielnych katalogach.

Pełny przebieg bramek i stanów `two_qutrit`, `ghz3`, `ame43`:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --mode all --output-dir artifacts/theta_full
```

Wznowienie po przerwaniu procesu, z konfiguracją odczytaną z manifestu:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --resume artifacts/theta_full --mode all
```

Oddzielny etap pełnych obwodów dla istniejących, zapisanych wyników bramek:

```powershell
.venv/Scripts/python.exe scripts/run_theta_benchmark.py --mode circuits --resume artifacts/theta_pilot
```

Równoważna postać ostatniego polecenia używa `--mode circuits --output-dir artifacts/theta_pilot`. Tryb `circuits` wymaga istniejącego manifestu i nie prowadzi ponownej syntezy bramek. Tryb `all` może rozszerzyć zakończony etap `gates` o pełne obwody w tym samym przebiegu. Nie wolno łączyć `--resume` z `--output-dir`. Jawnie podane, sprzeczne parametry konfiguracji są odrzucane; nie są ignorowane.

Jeśli środowisko jest już aktywowane, zamiast `.venv/Scripts/python.exe` można użyć `python`. W POSIX odpowiednikiem jest:

```bash
PYTHONPATH=src python scripts/run_theta_benchmark.py --mode all --output-dir artifacts/theta_full
```

Nowy przebieg wymaga nowego lub pustego katalogu. Bez `--output-dir` powstaje osobny katalog pod `artifacts/theta_continuation/`, identyfikowany czasem UTC i losowym sufiksem. Windows wymaga osłony `if __name__ == '__main__':` przy własnych programach używających runnera; dostarczony CLI ją zawiera dla runtime'u BQSKit.

## Parametry protokołu

| Argument | Domyślnie | Znaczenie |
| --- | --- | --- |
| `--mode` | `all` | `gates`, `circuits` albo oba etapy |
| `--theta-points` | `41` | Równomierna siatka od 0 do pi/4, co najmniej 2 punkty |
| `--limit-points` | cały zakres | Prefiks siatki, od 1 do `theta-points` |
| `--max-nfev` | `3000` | Budżet dopasowania kątów w pojedynczej próbie |
| `--max-subdivisions` | `4` | Maksymalna głębokość połówkowania kroku, od 0 do 4 |
| `--f3-tolerance` | `1e-10` | Próg dla obu norm F3 |
| `--cz3-tolerance` | `1e-5` | Próg dla obu norm CZ3 |
| `--transpiler-seeds` | `0 1 2` | Jawny zbiór różnych seedów transpilacji |
| `--optimization-level` | `3` | Poziom transpilacji, od 0 do 3 |
| `--baseline-qpy` | przypięty plik powyżej | Lokalny QPY z jednym obwodem czterokubitowym |
| `--baseline-sha256` | przypięty hash powyżej | Dokładny SHA-256 źródłowego QPY |
| `--states` | `two_qutrit ghz3 ame43` | Stany dla etapu pełnych obwodów |

Baseline i kandydaci są transpilowani identycznie: `basis_gates=['u','cz']`, pełna łączność, `approximation_degree=1.0`. Wśród poprawnych obwodów wygrywa kolejno najmniejsza liczba CZ, głębokość, liczba bramek 1q i seed. Koszt oryginalnego schematu i koszt po transpilacji są oddzielnymi wielkościami. Routing na konkretnym procesorze nie jest częścią tego porównania.

Dla CZ3 `B_theta = E_theta tensor E_theta`. Dopasowanie obejmuje działanie `C(p) B_theta = exp(i gamma) B_theta CZ3`, przy jednej wspólnej fazie globalnej dla wszystkich dziewięciu kolumn. Końcowe `E_norm` i `L_norm` są obliczane niezależnie od statusu optymalizatora. Osobna faza dla każdej kolumny nie jest dozwolona: mogłaby ukrywać błędne fazy logiczne.

Po nieudanej próbie bezpośredniej runner połówkuje krok do zadanej głębokości. Zapisuje także próby pośrednie jako `requested_grid_point=false`; nie dopisuje ich do podstawowej siatki wyników. Fallback korzysta z istniejącego BQSKit StateSystem i jego `synthesis_epsilon=1e-8`. Ten parametr nie zastępuje końcowej tolerancji norm CZ3.

Fallback następuje po niepoprawnym dopasowaniu, niepoprawnej transpilacji albo przekroczeniu porównywalnego kosztu baseline. Wybierany jest najtańszy poprawny obwód spośród dostępnych prób. Poprawny wynik droższy niż baseline otrzymuje `above_baseline`; brak poprawnego wyniku otrzymuje `failed`.

Wynik BQSKit nie zmienia schematu kontynuacji. Gdy schemat przed transpilacją był poprawny, jego parametry mogą pozostać startem następnego punktu nawet przy wyborze fallbacku. W przeciwnym razie runner wraca do rzeczywistego ostatniego poprawnego stanu gałęzi, także punktu pośredniego. Sąsiednie theta wykonywane są sekwencyjnie.

## Artefakty i raport

Manifest utrwala konfigurację, źródło baseline, fingerprint, wersje bibliotek i hashe źródeł. Każdy ukończony pakiet ma `complete.json` z hashami JSON, NPY i QPY. Wznowienie odczytuje te same parametry gałęzi; nie odtwarza ich z zaokrąglonych tabel. Uszkodzony ukończony pakiet jest błędem artefaktu, a nie niepowodzeniem optymalizacji.

| Plik lub katalog | Zawartość |
| --- | --- |
| `manifest.json` | Niezmienny protokół i pochodzenie |
| `baseline/` | Kopia baseline, dane jego transpilacji, opis schematu i parametrów |
| `points/00000/` itd. | Wyniki docelowego theta, obwody, macierze, wszystkie próby oraz stan gałęzi |
| `circuits/{state}/{index}/` | Pełne obwody z zapisanej biblioteki F3/CZ3 i zweryfikowane metryki |
| `checkpoints/latest.json` | Ostatni utrwalony stan wznowienia |
| `points.csv` | Wszystkie ukończone punkty, również `failed` i `above_baseline` |
| `attempts.csv` | Próby, punkty pośrednie, rzeczywiści rodzice i metryki dopasowania |
| `parameters.csv` | Kąt, instrukcja, kubit, schemat, surowe i osobno unwrapped parametry |
| `operator_distances.csv` | Odległości lokalnych U3 między sąsiednimi poprawnymi punktami |
| `full_circuits.csv` | Koszty, statusy i fidelity pełnych obwodów |
| `figures/` | Samodzielne rysunki PNG i PDF, do dalszej analizy lub publikacji |
| `report.md` | Polski raport z protokołem, wynikami, ograniczeniami i wykresami |

CLI wypisuje bieżący etap oraz końcową ścieżkę raportu. Kod wyjścia `0` oznacza ukończenie protokołu, także gdy wystąpiły naukowo nieudane punkty. Kod `2` oznacza błąd konfiguracji lub artefaktów, `1` nieoczekiwany błąd wykonania, a `130` przerwanie przez użytkownika.

Ponowne wygenerowanie raportu bez uruchamiania syntezy:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -c "from qudits_on_qubits.benchmarks.theta_continuation.report import generate_report; print(generate_report('artifacts/theta_full'))"
```

Raport odczytuje wszystkie ukończone pakiety przez walidujący magazyn artefaktów. CSV i wykresy są danymi pochodnymi i można je odtworzyć z JSON/NPY/QPY.

## Koszt przygotowania i bloków

Raport zestawia koszt przygotowania zakodowanego zera, wszystkich bloków F3 oraz wszystkich bloków CZ3 w danym pełnym obwodzie. Są to sumaryczne koszty osobno kompilowanych składników przed łączeniem bloków, przy tej samej dokładnej transpilacji U/CZ. Obok sumy widnieje rzeczywisty koszt całego przetranspilowanego obwodu. Suma może różnić się od tego wyniku, ponieważ optymalizacja całego obwodu upraszcza bramki także na granicach bloków.

W `full_circuits.csv` składniki mają pola `preparation_n_cz`, `f3_blocks_n_cz`, `cz3_blocks_n_cz`, a ich suma pole `unfused_n_cz`. Odpowiednie koszty bramek jednokubitowych zapisano jako `preparation_n_1q`, `f3_blocks_n_1q`, `cz3_blocks_n_1q` i `unfused_n_1q`. Liczba kutrytów i bloków CZ3 znajduje się w `component_num_qutrits` i `component_num_cz3_blocks`. Końcowe koszty całego obwodu to `two_qubit_gate_count` i `one_qubit_gate_count`.

Starsze pakiety mogą nie zawierać tego podziału. Raport pokazuje wtedy —, a CSV pozostawia brak danych; brak nie oznacza kosztu zero. Nie wyprowadzamy kosztów składników z końcowej liczby bramek i nie zmieniamy istniejących artefaktów podczas raportowania.

## Interpretacja parametrów

Surowe 33 kąty stałego schematu są grupowane po trzy według rzeczywistych instrukcji U3 i kubitów. Każdy `template_id` ma osobne wykresy. Brak poprawnych parametrów bieżącego theta oznacza przerwę, nawet gdy zapisano parametry poprzedniego poprawnego rodzica albo fallback znalazł inny poprawny obwód.

`unwrap` usuwa skoki reprezentacji o wielokrotność `2*pi` wyłącznie wewnątrz spójnego segmentu poprawnych punktów. Nie przechodzi przez nieudany punkt ani zmianę schematu. Sam wykres po `unwrap` nie dowodzi gładkości: kąty Eulera mają niejednoznaczną reprezentację i osobliwości współrzędnych.

Dlatego raport liczy także `min_gamma ||U_j(theta_i) - exp(i gamma) U_j(theta_(i-1))||_F` dla lokalnych U3 zapisanych w artefaktach. Ta miara respektuje okresowość i równoważne reprezentacje tej samej U3 z dokładnością do fazy globalnej. Dotyczy sąsiednich poprawnych punktów tego samego schematu; nie usuwa całej swobody przenoszenia gauge między różnymi bramkami i nie dowodzi gładkości operatora pełnego obwodu.

Znany baseline z sześcioma CZ nie jest dowodem globalnego minimum. Znalezienie F3 z co najwyżej dwoma CZ nie dowodzi minimum po wszystkich `alpha`. Niepowodzenie kontynuacji oznacza jedynie, że wskazany schemat, start i skończony budżet numeryczny nie dały wyniku spełniającego progi. Raport zachowuje wyniki negatywne oraz koszty większe od baseline.

## Testy lokalne

Testy CLI używają atrap kosztownego runnera. Testy raportu zapisują rzeczywiste, małe pakiety artefaktów i weryfikują luki parametrów, okresowość U3, odrębność schematów oraz odrzucanie uszkodzonych danych.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv/Scripts/python.exe -m pytest tests/test_theta_benchmark_cli.py tests/test_theta_benchmark_report.py -q
```

F3 wymaga `0 < --f3-tolerance <= 1e-10`. CZ3 przyjmuje jawny, dodatni i skończony `--cz3-tolerance`; domyślnie nadal `1e-5`. Próg jest przekazywany do syntezy, walidacji bramek i niezależnej walidacji biblioteki pełnych obwodów. Nie zmienia epsilon, seedu ani algorytmu BQSKit. Zmiana progu wymaga nowego przebiegu i nowego manifestu.


## Ponowna ocena przy progu 5e-4

Osobny skrypt `scripts/reassess_theta_benchmark.py` odczytuje istniejącą kampanię i tworzy nowy raport. Nie modyfikuje źródłowego katalogu. Dla CZ3 sprawdza oba warunki: `E_norm <= 5e-4` i `L_norm <= 5e-4`; dla F3 zachowuje `1e-10`.

```powershell
.venv/Scripts/python.exe scripts/reassess_theta_benchmark.py --source-run artifacts/theta_benchmark/full-41points --output-dir artifacts/theta_benchmark/full-41points-tol5e-4 --cz3-tolerance 5e-4 --workers 3
```

Wznowienie używa tego samego polecenia z dopisanym `--resume`. Konfiguracja, wersje bibliotek, hashe kodu i źródłowych pakietów muszą zgadzać się z nowym manifestem.

Zapisane F3, baseline i surowe CZ3 są używane ponownie po walidacji hashy oraz działania operatorów. Jeśli stara synteza zapisała wyłącznie metryki odrzuconego wyniku, obwód jest odtwarzany z tymi samymi ustawieniami BQSKit. Nowy raport oznacza takie rekonstrukcje i porównuje ich normy oraz liczbę CZ z poprzednimi wartościami. Wznowienie odczytuje ukończone pakiety odzyskania, zamiast ponawiać ich syntezę.

To ponowna ocena zapisanych prób, nie nowa kontynuacja: historyczne parametry startowe i rodzice pozostają niezmienione. Niezależne rekonstrukcje BQSKit mogą działać równolegle. Nie wolno interpretować ich jako równoległej kontynuacji kolejnych theta.

Jeżeli wszystkie standardowe transpilacje CZ3 zmieniają pełny operator o więcej niż `1e-10`, kompilacja próbuje dokładnego zastąpienia każdej U3 przez U z identycznymi parametrami. Zachowuje przewody, kolejność CZ i fazę globalną. Ten wariant jest jawnie oznaczony; próg zachowania pełnego operatora nadal wynosi `1e-10`.

Raport zawiera rzeczywiste liczby CZ i bramek jednokubitowych, głębokość całkowitą, głębokość warstw CZ, normy błędu i leakage, fidelity pełnych stanów oraz podział kosztów na przygotowanie zera, F3 i CZ3. Rysunki powstają z zapisanych QPY. Akceptacja przy `5e-4` oznacza spełnienie tego progu, a nie zerowy błąd.
