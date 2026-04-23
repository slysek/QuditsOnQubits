# Benchmark `benchmark_encoding_bases_full_results.csv`

## Gdzie są dane

Źródłowy plik z pełnymi wynikami znajduje się teraz tutaj:

- `data/benchmarks/benchmark_encoding_bases_full_results.csv`

Ten benchmark zawiera **768 kandydatów** i w aktualnym eksporcie wszystkie wiersze mają
status **`ok`**. Zbiór obejmuje zarówno bazy pozostające w starej przestrzeni
kodowej, jak i bardziej ogólne izometrie działające w pełnym `C^4`.

## Jak czytać kolumny

### Identyfikacja kandydata

| Kolumna | Znaczenie |
| --- | --- |
| `class_name` | rodzina / klasa kodowania |
| `candidate_name` | konkretna instancja wewnątrz klasy |
| `is_valid` | czy kandydat przeszedł walidację mapy kodowania |
| `status`, `error_message` | status wykonania benchmarku |

### Geometria i własności kodowania

| Kolumna | Znaczenie |
| --- | --- |
| `uses_old_codespace_only` | czy kodowanie pozostaje całkowicie w starej przestrzeni kodowej |
| `avg_codeword_entanglement` | średnie splątanie słów kodowych |
| `overlap_with_old_codespace` | nakładanie z poprzednią przestrzenią kodową w skali `0..1` |

### Koszt po transpilacji

| Kolumna | Znaczenie |
| --- | --- |
| `best_depth`, `mean_depth`, `std_depth` | najlepsza i średnia głębokość obwodu po wielokrotnej transpilacji |
| `best_size`, `mean_size` | najlepszy i średni rozmiar obwodu |
| `best_two_qubit_gate_count`, `mean_two_qubit_gate_count` | liczba bramek 2-qubitowych |
| `best_count_ops` | rozkład operacji dla najlepszego przebiegu |
| `n_transpile_runs`, `successful_trials`, `failed_trials` | ile prób wykonano i ile zakończyło się sukcesem |

### Kolumny aproksymacji

Pełny CSV zawiera też osobny benchmark dla `approximation_degree`.

| Kolumna | Znaczenie |
| --- | --- |
| `approx_ref_depth`, `approx_ref_two_qubit_gate_count` | punkt odniesienia dla transpilacji bez dodatkowej aproksymacji |
| `approx_status`, `approx_error_message` | status benchmarku aproksymacji |
| `fid085_*`, `fid090_*`, `fid095_*` | najlepsze ustawienie aproksymacji, które utrzymuje zadaną wierność (`0.85`, `0.90`, `0.95`) |

## Najważniejsze liczby

- Wszystkie **768 / 768** rekordów zakończyły się poprawnie.
- Kandydaci `old-codespace-only`: **233**
- Kandydaci wychodzący poza starą przestrzeń: **535**
- Średni `mean_depth` dla kandydatów bez splątania słów kodowych: **99.39**
- Średni `mean_depth` dla kandydatów ze splątaniem: **119.78**

To już na pierwszy rzut oka sugeruje, że w tym benchmarku bardziej „uporządkowane”
i mniej splątane kodowania zwykle są tańsze po transpilacji.

## Globalnie najlepsze wyniki

### Top 10 według `best_depth`

| Miejsce | Klasa | Kandydat | `best_depth` | `best_two_qubit_gate_count` | `mean_depth` |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | `baseline` | `E_old` | 47 | 32 | 53.90 |
| 2 | `monomial` | `P012_ph000` | 74 | 44 | 81.15 |
| 3 | `clifford_wh` | `X0Z0F0` | 74 | 44 | 81.15 |
| 4 | `structured_entangling` | `t0.00_p0.00_a0.00` | 74 | 44 | 81.15 |
| 5 | `monomial` | `P012_ph110` | 74 | 44 | 81.75 |
| 6 | `monomial` | `P012_ph220` | 74 | 44 | 81.75 |
| 7 | `local_ry_only` | `ry_3.142_0.000` | 74 | 44 | 82.05 |
| 8 | `structured_entangling` | `t0.52_p0.00_a0.00` | 74 | 44 | 82.65 |
| 9 | `structured_entangling` | `t0.79_p0.00_a0.00` | 74 | 44 | 82.65 |
| 10 | `structured_entangling` | `t1.05_p0.00_a0.00` | 74 | 44 | 82.65 |

Najmocniejszy wniosek jest prosty: **`baseline / E_old` nadal wygrywa wyraźnie**.
Najlepszy wynik spoza baseline ma `best_depth = 74`, czyli jest wyraźnie gorszy od
`47` dla `E_old`.

### Fidelity dla globalnego Top 10

W tabeli niżej zapis `approximation_degree -> fidelity` pokazuje, jaki poziom
aproksymacji dawał najlepszy wynik spełniający dany próg fidelity.

| Miejsce | Kandydat | `fid085` | `fid090` | `fid095` |
| --- | --- | --- | --- | --- |
| 1 | `baseline / E_old` | `0.91 -> 0.8786` | `0.95 -> 0.9098` | `0.99 -> 0.9909` |
| 2 | `monomial / P012_ph000` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| 3 | `clifford_wh / X0Z0F0` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| 4 | `structured_entangling / t0.00_p0.00_a0.00` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| 5 | `monomial / P012_ph110` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| 6 | `monomial / P012_ph220` | `0.96 -> 0.9909` | `0.96 -> 0.9909` | `0.96 -> 0.9909` |
| 7 | `local_ry_only / ry_3.142_0.000` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| 8 | `structured_entangling / t0.52_p0.00_a0.00` | `0.96 -> 0.9901` | `0.96 -> 0.9901` | `0.96 -> 0.9901` |
| 9 | `structured_entangling / t0.79_p0.00_a0.00` | `0.91 -> 0.8814` | `0.96 -> 0.9890` | `0.96 -> 0.9890` |
| 10 | `structured_entangling / t1.05_p0.00_a0.00` | `0.96 -> 0.8777` | `0.99 -> 0.9887` | `0.99 -> 0.9887` |

To dobrze pokazuje, że najlepsze kandydaty nie tylko mają niski `best_depth`,
ale też zazwyczaj umieją utrzymać wysoki poziom fidelity przy dość agresywnej
aproksymacji. Szczególnie wyróżniają się tu `P012_ph220` oraz
`t0.52_p0.00_a0.00`, które już przy `approximation_degree = 0.96` osiągają
poziom bliski `0.99`.

## Najlepszy kandydat w każdej klasie

| Klasa | Najlepszy kandydat | `best_depth` | `best_2q` | `mean_depth` | Splątanie | Overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `E_old` | 47 | 32 | 53.90 | 0.000 | 1.000 |
| `monomial` | `P012_ph000` | 74 | 44 | 81.15 | 0.000 | 1.000 |
| `clifford_wh` | `X0Z0F0` | 74 | 44 | 81.15 | 0.000 | 1.000 |
| `structured_entangling` | `t0.00_p0.00_a0.00` | 74 | 44 | 81.15 | 0.000 | 1.000 |
| `local_ry_only` | `ry_3.142_0.000` | 74 | 44 | 82.05 | 0.000 | 0.667 |
| `finer_structured` | `fine_t0.00_p1.57_a0.50` | 91 | 50 | 101.50 | 0.000 | 0.822 |
| `local_general_su2` | `lsu2_008` | 91 | 53 | 101.65 | 0.000 | 0.721 |
| `real_orthogonal` | `real_009` | 96 | 56 | 107.75 | 0.270 | 0.825 |
| `fourier_like` | `D010_F3_D110` | 102 | 56 | 111.30 | 0.550 | 1.000 |
| `two_cz_ansatz` | `2cz_000` | 101 | 59 | 114.80 | 0.998 | 0.801 |
| `householder_random` | `rand_000` | 108 | 62 | 122.55 | 0.456 | 1.000 |
| `haar_random_isometry` | `haar_000` | 108 | 62 | 122.55 | 0.497 | 0.729 |
| `perturbed_isometry` | `pert_eps0.01_00` | 108 | 62 | 122.55 | 0.003 | 1.000 |
| `entangling_isometry` | `ent_000` | 108 | 62 | 122.55 | 0.388 | 0.704 |
| `near_identity` | `nearid_eps0.01_00` | 108 | 62 | 122.55 | 0.001 | 1.000 |

### Fidelity dla najlepszego kandydata z każdej klasy

`brak` oznacza, że w zapisanym sweepie aproksymacji nie udało się znaleźć wyniku
spełniającego dany próg fidelity.

| Klasa | Kandydat | `fid085` | `fid090` | `fid095` |
| --- | --- | --- | --- | --- |
| `baseline` | `E_old` | `0.91 -> 0.8786` | `0.95 -> 0.9098` | `0.99 -> 0.9909` |
| `monomial` | `P012_ph000` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| `clifford_wh` | `X0Z0F0` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| `structured_entangling` | `t0.00_p0.00_a0.00` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| `local_ry_only` | `ry_3.142_0.000` | `0.91 -> 0.9098` | `0.91 -> 0.9098` | `0.99 -> 0.9909` |
| `finer_structured` | `fine_t0.00_p1.57_a0.50` | `0.98 -> 0.8784` | `0.99 -> 0.9927` | `0.99 -> 0.9927` |
| `local_general_su2` | `lsu2_008` | `0.97 -> 0.9164` | `0.97 -> 0.9164` | `0.98 -> 0.9796` |
| `real_orthogonal` | `real_009` | `0.95 -> 0.8854` | `0.99 -> 0.9888` | `0.99 -> 0.9888` |
| `two_cz_ansatz` | `2cz_000` | `0.99 -> 0.9850` | `0.99 -> 0.9850` | `0.99 -> 0.9850` |
| `fourier_like` | `D010_F3_D110` | `0.97 -> 0.8531` | `0.98 -> 0.9909` | `0.98 -> 0.9909` |
| `householder_random` | `rand_000` | `0.98 -> 0.9075` | `0.98 -> 0.9075` | `brak` |
| `haar_random_isometry` | `haar_000` | `0.98 -> 0.8543` | `0.99 -> 0.9427` | `brak` |
| `perturbed_isometry` | `pert_eps0.01_00` | `0.97 -> 0.9858` | `0.97 -> 0.9858` | `0.97 -> 0.9858` |
| `entangling_isometry` | `ent_000` | `0.97 -> 0.8597` | `0.99 -> 0.9205` | `brak` |
| `near_identity` | `nearid_eps0.01_00` | `0.93 -> 0.9138` | `0.93 -> 0.9138` | `0.99 -> 0.9869` |

## Średni obraz klas

| Klasa | Liczba kandydatów | Średni `mean_depth` klasy | Najlepszy `mean_depth` | Średni `mean_2q` |
| --- | ---: | ---: | ---: | ---: |
| `baseline` | 1 | 53.90 | 53.90 | 33.05 |
| `structured_entangling` | 125 | 99.83 | 81.15 | 49.05 |
| `monomial` | 120 | 101.19 | 81.15 | 53.30 |
| `clifford_wh` | 27 | 104.95 | 81.15 | 53.28 |
| `local_ry_only` | 99 | 90.61 | 82.00 | 46.11 |
| `finer_structured` | 100 | 104.89 | 101.50 | 51.23 |
| `local_general_su2` | 30 | 102.50 | 101.65 | 51.50 |
| `real_orthogonal` | 20 | 111.34 | 107.75 | 55.70 |
| `fourier_like` | 64 | 121.08 | 111.30 | 59.38 |
| `two_cz_ansatz` | 50 | 114.80 | 114.80 | 57.50 |
| `householder_random` | 20 | 122.55 | 122.55 | 60.50 |
| `haar_random_isometry` | 20 | 122.55 | 122.55 | 60.50 |
| `perturbed_isometry` | 32 | 122.55 | 122.55 | 60.50 |
| `entangling_isometry` | 20 | 122.55 | 122.55 | 60.50 |
| `near_identity` | 40 | 122.55 | 122.55 | 60.50 |

## Wnioski praktyczne

1. Jeśli celem jest **jak najmniejsza głębokość po transpilacji**, nadal najlepiej wypada `baseline`.
2. Wśród alternatyw najbardziej obiecujące są klasy **mocno uporządkowane i mało splątujące**:
   `monomial`, `clifford_wh`, `structured_entangling`, `local_ry_only`.
3. Klasy bardziej ogólne i „losowe” często wpadają w ten sam słaby plateau:
   `best_depth = 108`, `best_two_qubit_gate_count = 62`, `mean_depth = 122.55`.
4. Sam fakt bycia blisko starej przestrzeni kodowej nie wystarcza. Dużo ważniejsze jest to,
   czy dana baza daje się zrealizować jako prosty, mocno ustrukturyzowany obwód.

## Co warto rozwijać dalej

- Dalsze strojenie klas `monomial`, `clifford_wh` i `structured_entangling`.
- Analizę progów `fid085`, `fid090`, `fid095`, bo pełny CSV ma już dane do osobnego
  porównania kompromisu między kosztem obwodu a fidelity.
- Porównanie najlepszych kandydatów nie tylko po `best_depth`, ale też po
  `mean_depth`, `mean_two_qubit_gate_count` i stabilności (`std_depth`).
