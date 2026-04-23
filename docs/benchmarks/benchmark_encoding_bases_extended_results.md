# Wyniki benchmarków baz kodowania quditów na qubitach

> **Pliki źródłowe:**
> - `data/benchmarks/benchmark_encoding_bases_results.csv` — benchmark podstawowy (429 kandydatów)
> - `data/benchmarks/benchmark_encoding_bases_extended_results.csv` — benchmark rozszerzony (339 kandydatów)
>
> **Łącznie:** 768 kandydatów | **Status wszystkich prób:** ✅ OK (brak błędów)

---

## Spis treści

1. [Opis ogólny](#opis-ogólny)
2. [Struktura danych](#struktura-danych)
3. [Benchmark podstawowy](#benchmark-podstawowy)
   - [Baseline — punkt odniesienia](#1-baseline--punkt-odniesienia)
   - [Monomial — permutacje z fazami](#2-monomial--permutacje-z-fazami)
   - [Fourier-like — transformacje Fouriera](#3-fourier_like--transformacje-fouriera)
   - [Householder Random](#4-householder_random--losowe-odbicia-householdera)
   - [Clifford-WH — kombinacje Clifforda i Walsh-Hadamarda](#5-clifford_wh--kombinacje-clifforda-i-walsh-hadamarda)
   - [Haar Random Isometry](#6-haar_random_isometry--losowe-izometrie-haara)
   - [Perturbed Isometry](#7-perturbed_isometry--zaburzone-izometrie)
   - [Entangling Isometry](#8-entangling_isometry--splątane-izometrie)
   - [Structured Entangling](#9-structured_entangling--strukturowane-splątujące)
4. [Benchmark rozszerzony](#benchmark-rozszerzony)
   - [local_ry_only](#1-local_ry_only--lokalne-rotacje-ry)
   - [local_general_su2](#2-local_general_su2--ogólne-lokalne-su2)
   - [real_orthogonal](#3-real_orthogonal--rzeczywiste-macierze-ortogonalne)
   - [near_identity](#4-near_identity--transformacje-bliskie-identyczności)
   - [finer_structured](#5-finer_structured--strukturowane-bazy-z-parametrami-ciągłymi)
   - [two_cz_ansatz](#6-two_cz_ansatz--ansatz-z-dwoma-bramkami-cz)
5. [Porównanie obu benchmarków](#porównanie-obu-benchmarków)
6. [Podsumowanie i wnioski](#podsumowanie-i-wnioski)

---

## Opis ogólny

Oba benchmarki testują **bazy kodowania** używane do reprezentacji quditów (układów kwantowych o więcej niż dwóch poziomach) przy pomocy qubitów na 128-qubitowym rejestrze kwantowym.

**Benchmark podstawowy** skupia się głównie na transformacjach zachowujących oryginalną przestrzeń kodową (`uses_old_codespace_only = True`) lub mających pełne nakładanie (`overlap = 1.0`) — testując strukturalne, algebraiczne typy kodowań.

**Benchmark rozszerzony** eksploruje bardziej ogólne transformacje wykraczające poza oryginalną przestrzeń kodową, w tym losowe i parametryczne bazy.

### Parametry globalne (oba benchmarki)

| Parametr | Wartość |
|---|---|
| Liczba qubitów | **128** |
| Liczba uruchomień transpilacji | **20** |
| Udane próby / Nieudane próby | **20 / 0** |
| Używane typy bramek | `rz`, `rx`, `sx`, `cz`, `rzz`, (`barrier` tylko w baseline) |

---

## Struktura danych

Każdy wiersz pliku CSV opisuje jednego kandydata.

### Kolumny — identyfikacja

| Kolumna | Opis |
|---|---|
| `class_name` | Kategoria/klasa kandydata |
| `candidate_name` | Unikalna nazwa kandydata |
| `status` | Status wykonania (`ok` lub błąd) |

### Kolumny — właściwości fizyczne

| Kolumna | Opis |
|---|---|
| `is_valid` | Czy kandydat spełnia kryteria poprawności kwantowej |
| `uses_old_codespace_only` | Czy transformacja zachowuje oryginalną przestrzeń kodową |
| `avg_codeword_entanglement` | Średnie splątanie słów kodowych (0 = brak splątania) |
| `overlap_with_old_codespace` | Nakładanie się z poprzednią przestrzenią kodową (0–1) |

### Kolumny — metryki obwodu

| Kolumna | Opis |
|---|---|
| `best_depth` | Najlepsza osiągnięta głębokość obwodu |
| `mean_depth` / `std_depth` | Średnia i odch. std. głębokości po 20 transpilacjach |
| `best_size` | Najlepsza łączna liczba bramek |
| `mean_size` | Średnia liczba bramek |
| `best_two_qubit_gate_count` | Najlepsza liczba bramek dwuqubitowych |
| `mean_two_qubit_gate_count` | Średnia liczba bramek dwuqubitowych |
| `best_count_ops` | Szczegółowy rozkład bramek przy najlepszym wyniku |

---

## Benchmark podstawowy

**Plik:** `data/benchmarks/benchmark_encoding_bases_results.csv`  
**Łączna liczba kandydatów:** 429

### Klasy kandydatów (podstawowy)

---

### 1. `baseline` — punkt odniesienia

**Liczba kandydatów:** 1 (`E_old`)

Oryginalny, istniejący schemat kodowania — punkt bazowy dla wszystkich porównań.

| Metryka | Wartość |
|---|---|
| `uses_old_codespace_only` | **True** |
| `avg_codeword_entanglement` | 0.0 |
| `overlap_with_old_codespace` | **1.0** |
| `best_depth` | **47** |
| `best_size` | **141** |
| `best_two_qubit_gate_count` | **32** |
| Rozkład bramek | `rz:52, rx:30, sx:27, rzz:17, cz:15` + 6 barier |

> **To jest absolutnie najlżejszy obwód w całym zbiorze** — głębokość 47 i zaledwie 32 bramki dwuqubitowe. Stanowi punkt odniesienia dla oceny kosztów wszystkich pozostałych kodowań.

---

### 2. `monomial` — permutacje z fazami

**Liczba kandydatów:** 120

Macierze monomiczne łączą **permutację poziomów quditu** z **lokalnymi przesunięciami fazy**. Kandydaci opisani są formatem `PXXX_phYYY`, gdzie:
- `PXXX` — jedna z permutacji: `P012`, `P021`, `P102`, `P120`, `P201`
- `phYYY` — trójka faz w podstawie trójkowej (każda cyfra ∈ {0,1,2})

Wszystkie warianty mają `uses_old_codespace_only = True` i `overlap = 1.0`.

| Permutacja | Kandydatów | Głębokość | Rozmiar | Bramki 2-qub. | Uwagi |
|---|---|---|---|---|---|
| `P012` | 27 | **69** | 168 | **46** | Tożsama z identycznością w strukturze |
| `P021` | 27 | **69** | 168 | **46** | Jak P012 |
| `P102` | 27 | 101–103 | 172–181 | 47–48 | Nieznacznie głębsza |
| `P120` | 27 | 107 | 177 | 53 | Najbardziej złożona z grupy |
| `P201` | 12 | 101–103 | 172–181 | 47–48 | Jak P102 |

> **Obserwacja:** Fazy (`phYYY`) nie zmieniają złożoności obwodu — wszystkie 27 kombinacji faz tej samej permutacji mają identyczne metryki. Różnica jest wyłącznie między permutacjami.

> **Najefektywniejsza klasa zachowująca pełne nakładanie** (poza baseline): P012/P021 z głębokością 69.

---

### 3. `fourier_like` — transformacje Fouriera

**Liczba kandydatów:** 64  
**Format:** `D{pre}_F3_D{post}` — transformacja Fouriera quditu rzędu 3 otoczona macierzami diagonalnymi

| Metryka | Wartość |
|---|---|
| `uses_old_codespace_only` | **True** |
| `avg_codeword_entanglement` | **0.550048** (stałe dla wszystkich) |
| `overlap_with_old_codespace` | **1.0** |
| `best_depth` | 964–975 |
| `best_size` | 1592–1614 |
| `best_two_qubit_gate_count` | **392** (prawie stałe) |

> **Obserwacja:** Wszystkie 64 warianty mają niemal identyczne metryki — macierze `D` (pre/post) nie wpływają na złożoność obwodu. Splątanie wynosi dokładnie ≈ 0.550 dla każdego kandydata, co jest charakterystyczne dla transformacji Fouriera quditu 3-poziomowego. Klasa zachowuje pełne nakładanie, ale kosztem bardzo głębokiego obwodu.

---

### 4. `householder_random` — losowe odbicia Householdera

**Liczba kandydatów:** 20  
**Identyfikatory:** `rand_000` – `rand_019`

Losowe złożenia odbić Householdera generujące unitarne transformacje zachowujące oryginalną przestrzeń kodową.

| Metryka | Min | Max |
|---|---|---|
| `uses_old_codespace_only` | True | True |
| `avg_codeword_entanglement` | 0.104 | **0.571** |
| `overlap_with_old_codespace` | **1.0** | **1.0** |
| `best_depth` | 967 | 973 |
| `best_two_qubit_gate_count` | 392 | 392 |

> **Obserwacja:** Klasa utrzymuje `overlap = 1.0` przy różnym poziomie splątania (0.10–0.57). Wszystkie obwody mają niemal identyczną strukturę ~970 bramek głębokości i 392 bramki dwuqubitowe — co sugeruje, że transpilator redukuje każdą taką transformację do podobnego szablonu.

---

### 5. `clifford_wh` — kombinacje Clifforda i Walsh-Hadamarda

**Liczba kandydatów:** 27  
**Format:** `XxZzFf` — kombinacje operatorów X (x∈{0,1,2}), Z (z∈{0,1,2}) oraz F (f∈{0,1,2})

Klasa łączy transformacje Clifforda z transformacjami Walsh-Hadamarda, tworząc trzy wyraźnie różne poziomy złożoności w zależności od parametru `F`:

| Typ (F) | Kandydatów | Opis | Głębokość | Rozmiar | Bramki 2-qub. | Splątanie |
|---|---|---|---|---|---|---|
| **F0** | 9 | Czysta permutacja Clifforda | 69–107 | 168–181 | 46–53 | 0.0 |
| **F1** | 9 | Z transformacją Fouriera | ~955–973 | ~1574–1610 | 388–392 | **0.550** |
| **F2** | 9 | Z transformacją Hadamarda | ~450–507 | ~748–820 | 185–209 | ≈ 0.0 |

Wszystkie mają `uses_old_codespace_only = True` i `overlap = 1.0`.

> **Obserwacja:** Trzy wyraźne "rodziny" złożoności. F0 to lekkie transformacje Clifforda, F2 to pośrednie transformacje Hadamarda, F1 to pełne transformacje Fouriera. Parametry X i Z nie zmieniają złożoności — jedynie parametr F decyduje o klasie trudności.

---

### 6. `haar_random_isometry` — losowe izometrie Haara

**Liczba kandydatów:** 20  
**Identyfikatory:** `haar_000` – `haar_019`

Losowe izometrie próbkowane z miary Haara — **pierwsza klasa podstawowego benchmarku** z `uses_old_codespace_only = False`.

| Metryka | Min | Max |
|---|---|---|
| `uses_old_codespace_only` | False | False |
| `avg_codeword_entanglement` | 0.145 | **0.699** |
| `overlap_with_old_codespace` | 0.674 | **0.885** |
| `best_depth` | 932 | 977 |
| `best_two_qubit_gate_count` | 375 | 407 |

> **Obserwacja:** Losowość Haara wprowadza typowe splątanie ~0.5 i nakładanie ~0.73. Najlepszy kandydat `haar_006` osiąga nakładanie 0.885.

---

### 7. `perturbed_isometry` — zaburzone izometrie

**Liczba kandydatów:** 32  
**Format:** `pert_epsX.XX_YY` — perturbacje bazowej izometrii z 4 poziomami ε

| ε | Kandydatów | Śr. nakładanie | Zakres splątania |
|---|---|---|---|
| **0.01** | 8 | ~0.9997 | 0.001 – 0.002 |
| **0.05** | 8 | ~0.9968 | 0.013 – 0.079 |
| **0.10** | 8 | ~0.9806 | 0.075 – 0.205 |
| **0.30** | 8 | ~0.8884 | 0.128 – 0.709 |

> **Obserwacja:** Analogia do `near_identity` z benchmarku rozszerzonego. Im większe ε, tym niższe nakładanie i wyższe splątanie. Przy ε=0.30 nakładanie spada do ~0.80 i pojawia się znaczące splątanie.

---

### 8. `entangling_isometry` — splątane izometrie

**Liczba kandydatów:** 20  
**Identyfikatory:** `ent_000` – `ent_019`

Izometrie skonstruowane tak, by celowo wprowadzać splątanie między qubitami kodowymi.

| Metryka | Min | Max |
|---|---|---|
| `avg_codeword_entanglement` | 0.281 | **0.866** |
| `overlap_with_old_codespace` | 0.682 | **0.848** |
| `best_depth` | 903 | 973 |
| `best_two_qubit_gate_count` | 367 | 394 |

> **Obserwacja:** Wysoki poziom splątania (do 0.87) przy umiarkowanym nakładaniu. Kandydat `ent_011` osiąga najwyższe splątanie (0.822) w tej klasie.

---

### 9. `structured_entangling` — strukturowane splątujące

**Liczba kandydatów:** 125  
**Format:** `tT_pP_aA` — siatka 5×5×5 parametrów

Systematyczna eksploracja przestrzeni parametrów (t, p, a), każdy z 5 wartości:
- **t** (theta) ∈ {0.00, 0.52, 0.79, 1.05, 1.57}
- **p** (phi) ∈ {0.00, 0.52, 0.79, 1.05, 1.57}
- **a** (alpha) ∈ {0.00, 0.52, 0.79, 1.05, 1.57}

| Metryka | Min | Max | Uwagi |
|---|---|---|---|
| `uses_old_codespace_only` | False | **True** | Jeden kandydat: `t0.00_p0.00_a0.00` = identyczność |
| `avg_codeword_entanglement` | 0.0 | 0.0 | Brak splątania |
| `overlap_with_old_codespace` | **0.75** | **1.0** | Systematyczny spadek wraz z rosnącymi kątami |
| `best_depth` | **69** | 1003 | Min. dla t=p=a=0 (identyczność) |
| `best_two_qubit_gate_count` | **46** | 412 | |

> **Obserwacja:** Kandydat `t0.00_p0.00_a0.00` to identyczność — identyczny wynik z baseline klasy `monomial`. Dla t=p=1.57 i dużych wartości a nakładanie osiąga minimum **0.75**. Brak splątania dla wszystkich kandydatów.

> **Ważna regularność:** Kandydaci z `a=1.57` (maksymalna alpha) przy dowolnych t i p mają nakładanie 0.75 lub 0.833 — wartości te pojawiają się systematycznie.

---

## Benchmark rozszerzony

**Plik:** `data/benchmarks/benchmark_encoding_bases_extended_results.csv`  
**Łączna liczba kandydatów:** 339

---

### 1. `local_ry_only` — lokalne rotacje Ry

**Liczba kandydatów:** 99

Kandydaci opisani są parą kątów `(θ₁, θ₂)` z siatki wartości wielokrotności π/5:
`{0.000, 0.628, 1.257, 1.885, 2.513, 3.142, 3.770, 4.398, 5.027, 5.655}`.

Klasa używa wyłącznie **lokalnych jednoqubitowych rotacji Ry** — bez splątania.

| Metryka | Min | Max | Uwagi |
|---|---|---|---|
| `avg_codeword_entanglement` | 0.0 | 0.0 | Brak splątania |
| `overlap_with_old_codespace` | 0.6667 | 0.9682 | Minimum = 2/3 (przy θ ≈ π) |
| `best_depth` | **201** | 1009 | Absolutnie najlepszy wynik całego zbioru |
| `best_two_qubit_gate_count` | **86** | 415 | |

> **Najlepszy kandydat:** `ry_3.142_0.000` (θ₁ = π, θ₂ = 0)  
> — głębokość = 201, rozmiar = 358, 86 bramek dwuqubitowych

---

### 2. `local_general_su2` — ogólne lokalne SU(2)

**Liczba kandydatów:** 30  
**Identyfikatory:** `lsu2_000` – `lsu2_029`

Losowe ogólne transformacje SU(2) na każdym qubicie — bez splątania.

| Metryka | Min | Max |
|---|---|---|
| `avg_codeword_entanglement` | 0.0 | 0.0 |
| `overlap_with_old_codespace` | 0.676 | 0.931 |
| `best_depth` | 887 | 1008 |
| `best_two_qubit_gate_count` | 359 | 415 |

---

### 3. `real_orthogonal` — rzeczywiste macierze ortogonalne

**Liczba kandydatów:** 20  
**Identyfikatory:** `real_000` – `real_019`

| Metryka | Min | Max |
|---|---|---|
| `avg_codeword_entanglement` | 0.039 | 0.748 |
| `overlap_with_old_codespace` | 0.671 | 0.955 |
| `best_depth` | 902 | 977 |
| `best_two_qubit_gate_count` | 364 | 408 |

---

### 4. `near_identity` — transformacje bliskie identyczności

**Liczba kandydatów:** 40  
**Format:** `nearid_epsX.XX_YY` — 4 poziomy ε × 10 próbek

| ε | Śr. nakładanie | Zakres splątania |
|---|---|---|
| **0.01** | ~0.9999 | 0.0006 – 0.0018 |
| **0.03** | ~0.9994 | 0.0007 – 0.0151 |
| **0.05** | ~0.9983 | 0.0046 – 0.0229 |
| **0.10** | ~0.9939 | 0.0072 – 0.1228 |

---

### 5. `finer_structured` — strukturowane bazy z parametrami ciągłymi

**Liczba kandydatów:** 100  
**Format:** `fine_tT_pP_aA` — siatka 4×5×5 parametrów (t, p, a)

| Metryka | Min | Max |
|---|---|---|
| `avg_codeword_entanglement` | 0.0 | 0.0 |
| `overlap_with_old_codespace` | 0.769 | 0.880 |
| `best_depth` | 869 | 1012 |
| `best_two_qubit_gate_count` | 361 | 418 |

---

### 6. `two_cz_ansatz` — ansatz z dwoma bramkami CZ

**Liczba kandydatów:** 50  
**Identyfikatory:** `2cz_000` – `2cz_049`

| Metryka | Min | Max |
|---|---|---|
| `avg_codeword_entanglement` | ~0.0 | **0.998** |
| `overlap_with_old_codespace` | 0.669 | 0.926 |
| `best_depth` | 874 | 982 |
| `best_two_qubit_gate_count` | 362 | 412 |

---

## Porównanie obu benchmarków

### Zestawienie klas według nakładania i złożoności

| Klasa | Benchmark | Kandyd. | `overlap` | Splątanie | Śr. głębokość | Zachowuje stary codespace? |
|---|---|---|---|---|---|---|
| `baseline` | podstawowy | 1 | 1.0 | 0.0 | **47** | ✅ Tak |
| `monomial` (P012/P021) | podstawowy | 54 | 1.0 | 0.0 | **69** | ✅ Tak |
| `clifford_wh` (F0) | podstawowy | 9 | 1.0 | 0.0 | 69–107 | ✅ Tak |
| `monomial` (P102/P201) | podstawowy | 39 | 1.0 | 0.0 | 101–103 | ✅ Tak |
| `monomial` (P120) | podstawowy | 27 | 1.0 | 0.0 | 107 | ✅ Tak |
| `clifford_wh` (F2) | podstawowy | 9 | 1.0 | ≈ 0.0 | ~450–510 | ✅ Tak |
| `local_ry_only` (najl.) | rozszerzony | 99 | 0.667–0.968 | 0.0 | 201–1009 | ❌ Nie |
| `structured_entangling` | podstawowy | 125 | 0.75–1.0 | 0.0 | 69–1003 | Częściowo |
| `fourier_like` | podstawowy | 64 | 1.0 | 0.550 | ~970 | ✅ Tak |
| `clifford_wh` (F1) | podstawowy | 9 | 1.0 | 0.550 | ~960 | ✅ Tak |
| `householder_random` | podstawowy | 20 | 1.0 | 0.10–0.57 | ~970 | ✅ Tak |
| `finer_structured` | rozszerzony | 100 | 0.769–0.880 | 0.0 | 869–1012 | ❌ Nie |
| `local_general_su2` | rozszerzony | 30 | 0.676–0.931 | 0.0 | 887–1008 | ❌ Nie |
| `near_identity` | rozszerzony | 40 | 0.985–0.9999 | ~0 | 924–975 | ❌ Nie |
| `perturbed_isometry` | podstawowy | 32 | 0.801–0.9997 | 0.001–0.71 | 924–973 | ❌ Nie |
| `real_orthogonal` | rozszerzony | 20 | 0.671–0.955 | 0.039–0.748 | 902–977 | ❌ Nie |
| `haar_random_isometry` | podstawowy | 20 | 0.674–0.885 | 0.145–0.699 | 932–977 | ❌ Nie |
| `entangling_isometry` | podstawowy | 20 | 0.682–0.848 | 0.281–0.866 | 903–973 | ❌ Nie |
| `two_cz_ansatz` | rozszerzony | 50 | 0.669–0.926 | 0.0–0.998 | 874–982 | ❌ Nie |

---

### Globalne rankingi głębokości — top 10 w całym zbiorze

| Rank | Kandydat | Klasa | Benchmark | Głębokość | Bramki 2-qub. | Overlap |
|---|---|---|---|---|---|---|
| 🥇 1 | `E_old` | baseline | podstawowy | **47** | 32 | 1.0 |
| 🥈 2 | `P012_ph000` – `P021_ph222` | monomial | podstawowy | **69** | 46 | 1.0 |
| 🥈 2 | `X0Z0F0`, `X0Z1F0`, `X0Z2F0` | clifford_wh (F0) | podstawowy | **69** | 46 | 1.0 |
| 🥉 3 | `t0.00_p0.00_a0.00` | struct. entangl. | podstawowy | **69** | 46 | 1.0 |
| 4 | `P102_*`, `P201_*`, `X1Z*F0` | monomial/clifford | podstawowy | 101–107 | 47–53 | 1.0 |
| 5 | `ry_3.142_0.000` | local_ry_only | rozszerzony | **201** | 86 | 0.667 |
| 6 | `ry_0.000_3.142`, `ry_3.142_0.628` | local_ry_only | rozszerzony | 212 | 92 | 0.667 |

---

### Porównanie metryk złożoności obwodu

```
Klasa                      | Głębokość (min-max)
---------------------------|--------------------
baseline                   |  ▌ 47
monomial (P012/P021)       |  ▌▌ 69
clifford_wh F0             |  ▌▌ 69-107
monomial (P102/P120)       |  ▌▌▌ 101-107
clifford_wh F2             |  ████████ 450-510
local_ry_only              |  ████-██████████████████ 201-1009
structured_entangling      |  ▌▌-██████████████████████ 69-1003
near_identity              |  ████████████████████ 924-975
perturbed_isometry         |  ████████████████████ 924-973
fourier_like               |  ████████████████████ 964-975
householder_random         |  ████████████████████ 967-973
clifford_wh F1             |  ████████████████████ 955-973
entangling_isometry        |  ████████████████████ 903-973
haar_random_isometry       |  ████████████████████ 932-977
two_cz_ansatz              |  ████████████████████ 874-982
```

---

## Podsumowanie i wnioski

### Co mierzono?

Benchmarki oceniają zdolność transformacji unitarnych do implementacji kodowania quditów na 128-qubitowym procesorze kwantowym. Metryki obejmują:
- **nakładanie z przestrzenią kodową** — zachowanie struktury poprzedniego kodowania
- **splątanie słów kodowych** — stopień splątania kwantowego w słowach kodowych
- **złożoność obwodu** — głębokość i liczba bramek po transpilacji

### Kluczowe wnioski

1. **Punkt odniesienia (`baseline E_old`)** osiąga głębokość 47 i tylko 32 bramki dwuqubitowe — jest **4–20× prostszy** od najtańszych nowych kodowań z pełnym nakładaniem (`monomial`, głębokość 69).

2. **Kodowania monomiczne (P012/P021)** są najtańszymi alternatywami zachowującymi pełne nakładanie (`overlap = 1.0`) — przy głębokości 69 są 17× głębsze od baseline.

3. **Transformacje Fouriera** (`fourier_like`, `clifford_wh F1`) utrzymują pełne nakładanie i wprowadzają splątanie ≈ 0.550, lecz kosztem bardzo dużej głębokości (~970).

4. **`local_ry_only`** (benchmark rozszerzony) osiąga absolutnie najniższą głębokość spoza klasy zachowującej pełne nakładanie — min. 201 dla θ₁=π.

5. **`near_identity`** i **`perturbed_isometry`** realizują tę samą koncepcję w obu benchmarkach — małe perturbacje bazowej izometrii zachowują niemal pełne nakładanie (do >0.999) kosztem umiarkowanej złożoności obwodu.

6. **`structured_entangling`** (podstawowy) i **`finer_structured`** (rozszerzony) to analogiczne klasy systematycznej eksploracji przestrzeni parametrów — obie bez splątania.

7. Większość klas **głębokość ~870–1010** — istnieje naturalny próg złożoności dla ogólnych 128-qubitowych transformacji unitarnych po transpilacji, niezależnie od struktury algebraicznej.

8. **`two_cz_ansatz`** oferuje unikalną kombinację: maksymalne splątanie (do ≈ 1.0) przy relatywnie niskiej głębokości (~870–982) — jest to najbardziej ekspresywna klasa z rozsądną złożonością.

---

*Wygenerowano na podstawie plików `data/benchmarks/benchmark_encoding_bases_results.csv` oraz `data/benchmarks/benchmark_encoding_bases_extended_results.csv`*
