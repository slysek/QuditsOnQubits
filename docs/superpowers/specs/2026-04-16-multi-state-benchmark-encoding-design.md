# Multi-State Encoding Benchmark Design

## Goal

Rozszerzyc `QuditsOnQubits/benchmark_encoding_bases.py`, tak aby ten sam benchmark obslugiwal trzy typy stanow:

- stan 2-qutrytowy,
- stan GHZ dla 3 qutrytow budowany przez `create_ame_circuit(n=3, dim=3, graph_type="star")`,
- stan `ame43`, dla ktorego trzeba najpierw utworzyc graf `Graph(n=4, edges=[[0, 1], [0, 1], [1, 2], [2, 3], [3, 0]])`, a dopiero potem wywolac `create_ame_circuit(dim=3, graph=game43)`.

Efektem maja byc osobne wyniki CSV i osobne katalogi obwodow dla `two_qutrit` i `ame43`, przy zachowaniu juz istniejacych wynikow dla `ghz3`, oraz jeden wspolny raport markdown z tabelami benchmarkow i progow fidelity dla wszystkich trzech stanow.

## Architecture

Plik `benchmark_encoding_bases.py` pozostaje glownym miejscem orkiestracji. Zmiana polega na wydzieleniu helpera budujacego obwod dla zadanego typu stanu oraz na rozszerzeniu `run_benchmark`, aby mogl:

- uruchomic benchmark dla jednego stanu,
- albo uruchomic komplet eksperymentow `two_qutrit`, `ghz3`, `ame43`,
- a nastepnie wygenerowac jeden wspolny raport markdown na podstawie trzech plikow CSV.

Logika generowania kandydatow baz pozostaje wspolna. Zmienia sie tylko sposob przygotowania obwodu bazowego i wybor sciezek wyjsciowych.

## Components

### 1. State selection

Do benchmarku zostanie dodany jawny wybor typu stanu, np. `state_kind`, obslugujacy:

- `two_qutrit`
- `ghz3`
- `ame43`

Helper budujacy obwod zwroci:

- gotowy `QuantumCircuit`,
- nazwe stanu do raportowania i sciezek,
- ewentualnie pomocniczo metadane potrzebne do opisu raportu.

Mapowanie:

- `two_qutrit` -> `create_ame_circuit(n=2, dim=3, graph_type="star", E_new=...)`
- `ghz3` -> `create_ame_circuit(n=3, dim=3, graph_type="star", E_new=...)`
- `ame43` -> utworzenie `Graph(...)`, potem `create_ame_circuit(dim=3, graph=game43, E_new=...)`

### 2. Benchmark execution

`benchmark_basis(...)` przestanie zakladac na sztywno `n_qutrits=3` i `graph_type="star"`. Zamiast tego bedzie korzystac z helpera stanu.

`run_benchmark(...)` dostanie dwa style pracy:

- pojedynczy stan: liczy tylko wskazany `state_kind`,
- zestaw eksperymentow: uruchamia trzy stany po kolei.

W trybie zestawu:

- `two_qutrit` liczy sie od nowa i zapisuje osobny CSV,
- `ghz3` moze skorzystac z juz istniejacego CSV, jesli uzytkownik nie zada przeliczenia,
- `ame43` liczy sie od nowa i zapisuje osobny CSV,
- po zebraniu danych tworzony jest jeden wspolny markdown.

### 3. Output layout

Zostana dodane osobne sciezki wynikowe dla stanow, tak aby nie mieszac danych:

- CSV:
  - `data/benchmarks/benchmark_encoding_bases_two_qutrit_<mode>_results.csv`
  - `data/benchmarks/benchmark_encoding_bases_ghz3_<mode>_results.csv`
  - `data/benchmarks/benchmark_encoding_bases_ame43_<mode>_results.csv`
- obwody:
  - `data/benchmarks/circuits/two_qutrit/...`
  - `data/benchmarks/circuits/ghz3/...`
  - `data/benchmarks/circuits/ame43/...`

Wspolny raport:

- `docs/benchmarks/benchmark_encoding_bases_multi_state_analysis.md`

### 4. Markdown report

Generator raportu markdown bedzie czytal trzy DataFrame i budowal jeden dokument z trzema sekcjami:

- `two_qutrit`
- `ghz3`
- `ame43`

Kazda sekcja dostanie tabele:

- podstawowe podsumowanie datasetu,
- top kandydaci wedlug `best_depth`,
- najlepszy kandydat w kazdej klasie,
- tabele fidelity dla progow `fid085`, `fid090`, `fid095`.

Tabele fidelity maja pokazywac nie tylko sam prog i osiagnieta fidelity, ale tez koszt obwodu przy danym progu. Dla kazdego progu w raporcie maja pojawic sie co najmniej:

- `best_approx_degree`,
- `best_fidelity`,
- `best_depth`,
- `best_two_qubit_gate_count`.

Dzieki temu bedzie od razu widac, jaka glebokosc obwodu odpowiada kazdemu poziomowi fidelity, zamiast tylko informacji o tym, czy prog zostal osiagniety.

Na koncu raportu pojawi sie tabela porownawcza miedzy trzema stanami, pokazujaca dla kazdej klasy najlepszy kandydat i jego metryki, zeby latwo zobaczyc roznice miedzy `two_qutrit`, `ghz3` i `ame43`.

## Data flow

Przeplyw dla pojedynczego stanu:

1. Wygeneruj kandydatow baz.
2. Dla kazdego kandydata zbuduj obwod odpowiedni dla `state_kind`.
3. Zapisz surowy obwod do katalogu przypisanego do stanu.
4. Policz metryki transpilacji i sweep aproksymacji.
5. Zapisz CSV dla tego stanu.

Przeplyw dla trybu laczonego:

1. Zbierz lub policz CSV dla `two_qutrit`, `ghz3`, `ame43`.
2. Wczytaj trzy zbiory do DataFrame.
3. Wygeneruj jeden markdown z tabelami per stan i tabela porownawcza.

## Error handling

- Nieznany `state_kind` powinien zwracac czytelny `ValueError`.
- Budowa `ame43` ma failowac z jasnym komunikatem, jesli nie uda sie zbudowac grafu lub obwodu.
- Generator raportu ma pomijac brakujacy zbior z czytelnym komunikatem tylko wtedy, gdy uzytkownik jawnie pozwoli na korzystanie z istniejacych danych; w standardowym przebiegu brak wymaganego CSV ma byc bledem.
- Istniejace zachowanie `status`, `error_message`, `approx_status`, `approx_error_message` zostaje zachowane.

## Testing

Testy maja objac:

- poprawna budowe obwodu dla `two_qutrit`,
- poprawna budowe obwodu dla `ame43` przez `Graph`,
- zapis obwodow do katalogow roznych stanow,
- zapis CSV do sciezek zaleznych od stanu,
- generowanie wspolnego markdowna zawierajacego sekcje `two_qutrit`, `ghz3`, `ame43` oraz tabele fidelity z kolumnami `best_fidelity`, `best_depth` i `best_two_qubit_gate_count` dla kazdego progu.

## Non-goals

- Bez zmiany generatorow baz kodowania.
- Bez zmiany logiki metryk kodowania i sweepu aproksymacji.
- Bez przepisywania calego benchmarku na wiele modulow.
