# Plan uproszczenia repo: Ponytail + Graphify

Data: 2026-10-21. Stan odniesienia: `8e3c0e6dd8975e1a31c7160fd2ce6278d1876601`.

Cel: przejrzeć całe repo poza AME46, uprościć zbędny kod i ocenić poprawność oraz przydatność testów.
Pracujemy kolejnymi, małymi zmianami. Kod, który jest potrzebny i czytelny, może pozostać bez zmian.
Zmniejszenie liczby linii nie jest warunkiem sukcesu.

## Zakres

W zakresie:

- `src/qudits_on_qubits/`: wszystkie moduły, eksporty i publiczne API.
- `tests/` oraz testy w `benchmarks/bell_20260907/`.
- `scripts/`, `examples/`, notebooki, kod historycznych benchmarków i narzędzia niezwiązane z AME46.
- Konfiguracja, zależności, pakowanie, CI i dokumentacja.
- Dane wejściowe, QPY, zapisane wyniki i archiwa: ich użycie, integralność i odtwarzalność.
  Nie zmieniamy danych pomiarowych w ramach porządkowania kodu.

Poza zakresem:

- `ame46/`, `ai_scientist/ame46/`, `tools/ame46_scientist/`.
- `tests/test_ame46_*.py` oraz dane i konfiguracja używane wyłącznie przez ten eksperyment.
- Nowe funkcje, zmiana modelu naukowego i nowe kampanie sprzętowe.

Wyłączenie AME46 nie wyłącza AME43 ani ogólnych funkcji konstrukcji stanów AME.
Wspólny kod pozostaje w zakresie. Jeśli korzysta z niego AME46, zachowujemy dotychczasowy kontrakt.
Współdzielonych plików nie wykluczamy tylko dlatego, że zawierają odwołanie do AME46.

Na początku istnieją już zmiany w `.gitignore` i `MANIFEST.in` oraz nieśledzone pliki eksperymentów,
narzędzi i testów. Zachowujemy je. Nowe pliki poza indeksem Graphify wymagają osobnego sprawdzenia.
Nie tworzymy commitów, gałęzi zdalnych ani PR bez osobnego polecenia użytkownika.

## Co sprawdzono podczas planowania

- Graphify działa. Indeks zawiera 6206 węzłów i 16849 relacji.
- Indeks pochodzi z 2026-10-09, ale jego commit odpowiada lokalnemu `HEAD`.
  Nie uwzględnia nowych plików ani późniejszych zmian roboczych.
- Spis plików poza wymienionymi ścieżkami AME46: 133 pliki Python w `src`, 41285 linii.
- Python w `tests`: 39916 linii. W samym katalogu `tests/` jest 129 plików `test_*.py`.
  Osobne testy historycznego benchmarku również należą do zakresu.
- Spis zawiera 25 notebooków. Liczby obejmują pliki śledzone oraz nieignorowane pliki nieśledzone.
- Duże obszary: `experiments` — 14531 linii, `benchmarks` — 13685 linii.
- `experiments/runner.py` ma 3908 linii, a `tests/test_experiment_runner.py` — 4498.
  Rozmiar wskazuje miejsce do analizy, ale nie dowodzi zbędnej złożoności.
- Są testy rzeczywistego Aer, regresje wartości Bella i odtwarzanie historycznych wyników.
  Są również testy treści dokumentacji, struktury AST notebooków i prywatnych funkcji.
  Ich poprawności jeszcze nie oceniono w całości.
- Domyślny `python` ma wersję 3.14.7, poza deklarowanym zakresem 3.11–3.13.
  `.venv/bin/python` ma wersję 3.12.15.
- W `.venv` są pytest 9.1.1 i pytest-cov 7.1.0, poza deklarowanymi zakresami odpowiednio `<9` i `<7`.
  Brakuje pakietu `build`. Przed pomiarem bazowym trzeba uzgodnić środowisko z deklaracją projektu.
- Przeczytano konfigurację pytest, oba workflow CI, instrukcję współtworzenia i wybrane testy.
  Podczas planowania nie uruchomiono testów i nie zmieniono kodu.

## Kolejność pracy

Każdy wiersz obejmuje również testy, wywołania, dane i dokumentację danego obszaru.
Duży etap dzielimy według jednej odpowiedzialności, zamiast zmieniać cały katalog naraz.

| Etap | Zakres i zadanie | Dowód zakończenia | Stan |
| --- | --- | --- | --- |
| 0 | Ustalić zakres, sprawdzić Graphify, stan Git, strukturę repo i istniejące walidacje. | Ten plan i ustalony stan odniesienia. | Gotowe |
| 1 | Uporządkować środowisko testowe. Zebrać testy, wyniki bazowe, czasy i przyczyny pominięć. Sporządzić spis wszystkich plików w zakresie. | Powtarzalne polecenia, raport błędów zastanych i przypisanie plików do dalszych etapów. | Gotowe — [raport](ponytail_stage1_baseline.md) |
| 2 | `core`, ścieżki zasobów, podstawowe bramki, kodowania i konstrukcja stanów. Stary benchmark z `core` przechodzi do etapu 6. | Sprawdzone własności matematyczne i zgodność dotychczasowych wywołań. | Gotowe offline — [raport](ponytail_stages2_3.md) |
| 3 | `bell_functionals`, `bell_measurements`, `reference_experiments.py`. Sprawdzić kolejność bitów, ustawienia pomiarowe i przetwarzanie wyników. | Zgodne niezależne obliczenia logiczne, estymator i sampler dla wspieranych stanów, w tym AME43. | Gotowe offline — [raport](ponytail_stages2_3.md) |
| 4 | `encoding_search` oraz główny interfejs `benchmarks`: kandydaci, synteza, kompilacja, walidacja, koszty, Pareto, zapis i CLI. | Mały benchmark od wejścia do zapisanego wyniku oraz poprawna kontrola kanoniczna. | Gotowe offline — [raport](ponytail_stages4_6.md) |
| 5 | `benchmarks/theta_continuation`: kontynuacja, zapisane bramki, ponowna ocena i warstwa sprzętowa offline. | Sprawdzone działanie bramek, leakage, tolerancje i odtworzenie zapisanych kandydatów. | Gotowe offline — [raport](ponytail_stages4_6.md) |
| 6 | `benchmarks/direct_basis`, `core/benchmark_encoding_bases.py` i powiązane skrypty. Szukać powtórzeń względem głównego benchmarku. | Zachowane używane scenariusze i formaty wejścia. Każde usunięcie ma uzasadnienie oraz sprawdzonych odbiorców. | Gotowe offline — [raport](ponytail_stages4_6.md) |
| 7 | `experiments`: modele, walidacja, store, artefakty, niepewność, bloki losowane, ZNE i readout. | Poprawne dane, statystyka, zapis i odczyt. Przerwanie nie niszczy wyników ani informacji o zadaniu. | Gotowe offline — [raport](ponytail_stages7_8.md) |
| 8 | `experiments/runner.py`, wznowienia i adaptery Aer, IBM, IQM, PiastQ. Dzielić zmiany według przepływu wykonania. | Lokalny przebieg Aer oraz testy kontraktów dostawców, błędów i wznowień bez ponownego zlecenia zadania. | Gotowe offline — [raport](ponytail_stages7_8.md) |
| 9 | `vertical_slice`, publiczne eksporty, przykłady, notebooki, pozostałe skrypty i narzędzia. | Używane scenariusze działają z czystych wejść, bez zależności od prywatnych katalogów. | Gotowe offline — [raport](ponytail_stages9_11.md) |
| 10 | `benchmarks/bell_20260907`, archiwa, `experiment_inputs`, zasoby QPY i zapisane wyniki. | Odtworzenie historycznych wyników w zadanych tolerancjach oraz zgodne hashe danych źródłowych. | Gotowe offline — [raport](ponytail_stages9_11.md) |
| 11 | Zależności, `pyproject.toml`, `requirements.txt`, pakowanie, CI i dokumentacja. Domknąć spis całego repo. | Testy offline, wheel/sdist i przykłady spoza checkoutu. Brak plików bez decyzji oraz jawna lista ograniczeń. | Gotowe offline — [raport](ponytail_stages9_11.md) |

Kolejność można skorygować, jeśli Graphify lub testy pokażą inną zależność.
Testy oceniamy w każdym etapie, przed uproszczeniem chronionego kodu.
Nie odkładamy ich audytu na koniec.


## Pierwszy etap wykonawczy: stan bazowy

1. Zachować spis początkowych zmian użytkownika. Niczego nie resetować ani nie czyścić.
2. Użyć Pythona zgodnego z `pyproject.toml`. Sprawdzić wersje i zależności bez wyświetlania sekretów.
3. Przygotować zgodne środowisko. Nie aktualizować bibliotek bez potrzeby i nie zmieniać zakresów tylko po to, by pasowały do lokalnej instalacji.
4. Sprawdzić warunki uruchamiania testów. Wyłączyć jawnie testy AME46, sprzęt i dostęp do usług.
5. Zebrać testy. Następnie uruchomić mały zestaw kontrolny i pełny zestaw offline w zakresie.
6. Zapisać wynik, czas, pominięcia i błędy. Oddzielić problemy środowiska, błędy zastane oraz problemy nowych plików.
7. Zbudować listę modułów i plików. Każdy otrzymuje stan: czeka, sprawdzony bez zmian, uproszczony albo wyłączony z uzasadnieniem.
8. Wybrać pierwszy mały pakiet zmian z etapu 2 na podstawie kodu, zależności i wyników testów.

Polecenia po przygotowaniu zgodnego środowiska, z katalogu repo:

```bash
env -u QOQ_RUN_IQM_HARDWARE -u QOQ_RUN_IQM_SELECTOR_SMOKE -u QOQ_RUN_BQSKIT_TESTS \
  .venv/bin/python -m pytest --collect-only -q tests benchmarks/bell_20260907 \
  --ignore-glob='tests/test_ame46_*.py'

.venv/bin/python -m pytest -q \
  tests/test_clean_repo_smoke.py \
  tests/test_experiment_aer_integration.py \
  tests/test_reference_regressions.py

env -u QOQ_RUN_IQM_HARDWARE -u QOQ_RUN_IQM_SELECTOR_SMOKE -u QOQ_RUN_BQSKIT_TESTS \
  .venv/bin/python -m pytest -q tests benchmarks/bell_20260907 \
  --ignore-glob='tests/test_ame46_*.py' --durations=25
```

To lista startowa z etapu planowania. Wyniki i polecenia dla przygotowanego środowiska są w [raporcie etapu 1](ponytail_stage1_baseline.md).
Podczas zbierania trzeba sprawdzić również nowe pliki i fixture.
Samo `-k 'not ame46'` nie wystarczy: pytest może wcześniej zaimportować wyłączany moduł.
Kosztowna rzeczywista synteza BQSKit ma osobny test opt-in.
Jeśli zmiana dotyczy syntezy, trzeba go uruchomić osobno albo jawnie wskazać brak tej walidacji.

## Jak oceniamy testy

Dla każdego testu lub grupy parametrycznej zapisujemy chronione zachowanie i źródło oczekiwanego wyniku.
Decyzja: zostawić, uprościć, połączyć, zastąpić albo usunąć. Usunięcie wymaga konkretnego uzasadnienia.

| Pytanie | Kryterium |
| --- | --- |
| Czy przypadek jest rzeczywisty? | Wynika z publicznego użycia, kontraktu biblioteki, danych eksperymentu, błędu regresyjnego albo matematycznej własności. |
| Czy test może wykryć błąd? | Oczekiwany wynik nie powstaje wyłącznie przez ponowne wywołanie tej samej badanej logiki. |
| Czy sprawdza właściwy wynik? | Poprawność fizyczna, statystyczna lub kontrakt danych. Sama skończoność liczby albo brak wyjątku nie wystarcza do potwierdzenia wyniku naukowego. |
| Czy atrapa odpowiada rzeczywistemu API? | Zachowuje kształt wyniku, layout, kolejność rejestrów i cykl życia zadania. Lokalnej integracji nie zastępuje zestaw samych atrap. |
| Czy test utrwala niepotrzebny szczegół? | Nazwy helperów, AST notebooka, dokładne nagłówki i kolejność prywatnych wywołań wymagają uzasadnienia. Kontrole sekretów i wykonania sprzętowego mogą go mieć. |
| Czy warianty są potrzebne? | Każdy wariant chroni osobną regułę, granicę albo znany błąd. Powtórzenia tych samych zachowań można połączyć. |
| Czy tolerancja jest uzasadniona? | Wynika z dokładności numerycznej, aproksymacji lub liczby strzałów. Nie dobieramy jej tylko po to, by test przeszedł. |
| Czy test jest stabilny? | Ma kontrolowane losowanie i nie zależy od zegara, sieci, lokalnych artefaktów ani szczęśliwego wyniku próbkowania. |
| Czy przypadek błędny jest potrzebny? | Jest osiągalny na granicy danych, dostawcy lub zapisu, albo chroni przed istotną utratą danych. Nietypowy przypadek nie oznacza automatycznie zbędnego testu. |

Obowiązkowe obszary poprawności:

- Kodowanie, ortonormalność, normalizacja i leakage, również dla niezerowego leakage.
- Kolejność tensorów, qubitów, bitów, rejestrów i mapowanie layoutu po kompilacji.
- Działanie F3/CZ3 w przestrzeni kodowej. Sprawdzenie jednego stanu nie dowodzi poprawności całej bramki.
- Faza globalna tam, gdzie fizycznie nie zmienia wyniku.
- Wartości Bella dla two-qutrit, GHZ3 i AME43 oraz zgodność obliczeń dokładnych z próbkowaniem.
- Warunkowanie na brak leakage, mianowniki, liczba strzałów i przypadki pustych lub niepoprawnych danych.
- Niepewność i bloki losowane. Zachować strukturę bloków i zależności, które wymaga metoda statystyczna.
- Równe budżety kompilacji, kontrola kanoniczna i niewłączanie niepoprawnych prób do Pareto.
- Uszkodzone artefakty, integralność zapisu, wznowienia, tajne dane w błędach oraz brak podwójnego zlecenia zadania.

Dla brakującego niezależnego sprawdzenia najpierw dodajemy najmniejszy test rzeczywistego zachowania.
Przy podejrzanym teście sprawdzamy, czy kontrolowany błędny wynik powoduje jego błąd.
Nie dodajemy od razu nowego frameworka ani masowych testów mutacyjnych.

Pokrycie mierzymy dla zmienianego obszaru, z celem co najmniej 80%.
Obok wyniku zapisujemy brakujące ważne przypadki. Procent nie zastępuje testu poprawności naukowej.
Nie ustalamy docelowej liczby testów ani obowiązkowego procentu redukcji kodu.

## Cykl małej zmiany

1. Graphify: zlokalizować symbole, wywołania, zależności i powiązane testy.
2. Przeczytać lokalny kod. Sprawdzić także eksporty, CLI, notebooki, fixture i pliki poza indeksem.
   Brak krawędzi w grafie nie jest dowodem braku użycia.
3. Zapisać problem, niezbędne zachowanie i najmniejsze rozwiązanie Ponytail.
4. Sprawdzić istniejące testy. Dla błędu lub brakującej ochrony dodać mały test przed zmianą, gdy to praktyczne.
5. Uprościć kod i wszystkich dotkniętych odbiorców. Zachować walidację, obsługę błędów i kalibrację sprzętu.
6. Uruchomić testy obszaru i zależnych ścieżek. Dobrać dodatkowe sprawdzenie do ryzyka.
7. Obejrzeć pełny diff. Zapisać wynik, zmianę rozmiaru oraz pominięte sprawdzenia w tym planie.

Nowa abstrakcja wymaga istniejącej potrzeby. Podział dużego pliku bez zmniejszenia złożoności nie wystarcza.
Usunięcie kodu wymaga sprawdzenia użycia w publicznym API, notebookach i odtwarzaniu archiwów.
Błąd naukowy lub bezpieczeństwa naprawiamy jawnie i z regresją, zamiast utrwalać go jako zachowanie bazowe.

Po zmianach indeks Graphify może opisywać starszy kod. Wynik grafu zawsze porównujemy z lokalnymi plikami.
Nie uruchamiamy automatycznych agentów recenzujących podczas tego testu Ponytail.
Użytkownik 2026-10-24 zlecił review Ponytail kolejnych zmian. Review wykonujemy w tym zakresie na jego polecenie.

## Warunki zakończenia

Etap jest gotowy, gdy cały przypisany zakres ma decyzję, istotne zachowania mają poprawne testy,
a adekwatna walidacja przechodzi. Niewykonane kontrole i błędy zastane mają jawny opis.
Każdy raport etapu podaje: co usunięto lub uproszczono, dlaczego, wyniki testów oraz pozostałe ryzyko.

Cała praca jest gotowa, gdy:

- Wszystkie pliki w zakresie mają decyzję. Brak zmian może być poprawnym wynikiem przeglądu.
- Pełny zestaw offline oraz odtwarzanie historycznego benchmarku przechodzą.
- Oba CLI i lokalne przykłady działają również po instalacji paczki poza checkoutem.
- Pakowanie obejmuje wymagane QPY, zapisane bramki i dane przykładów.
- Publiczne API i zapisane eksperymenty zachowują zgodność albo mają udokumentowaną, celową zmianę.
- Wyniki lokalne nie są przedstawiane jako potwierdzenie działania na QPU.
- Lista ograniczeń obejmuje niewykonane platformy, opcjonalnych dostawców i testy sprzętowe.

## Rejestr wykonania

| Etap / pakiet | Zmiana i uzasadnienie | Walidacja | Ograniczenia |
| --- | --- | --- | --- |
| 0 | Rozpoznanie repo i zapisanie planu. Bez zmian kodu. | Graphify, konfiguracja CI/pytest, spis plików i wersji środowiska. | Testy jeszcze nieuruchomione. To plan, nie pełny audyt poprawności. |
| 1 | Osobne `.venv-ponytail`, [spis 819 plików](ponytail_file_inventory.csv), [wersje zależności](ponytail_stage1_requirements.txt) i [raport](ponytail_stage1_baseline.md). Bez zmian kodu i danych. | 3065 testów zebranych. 3051 zaliczonych w pełnym przebiegu, 13 pominiętych, 1 błąd lokalnych gniazd sandboxa. Ten sam test BQSKit poza sandboxem: zaliczony. Zestaw kontrolny: 26 zaliczonych. Niezależne review Ponytail: `CLEAN`. | Testy źródła przez `PYTHONPATH=src`. Przerwano budowanie editable. Prywatny PiastQ, QPU, usługi IQM i kosztowna synteza CZ3 niezweryfikowane. |
| 2a | `generate_b_ame` korzysta z istniejącego `_load_qpy_gate`; usunięto osiem powtórzonych bloków. Strategia przygotowania w nowym kodowaniu ładuje F/CZ raz zamiast dwa razy. Kod produkcyjny: 20 linii mniej. | Osiem testów zasobów przeszło przed zmianą kodu. Po zmianie: 92 testy zaliczone, 8 zastanych ostrzeżeń IBM, bez pominięć. Niezależne review Ponytail: `CLEAN`. | Sprawdzono dostęp do QPY i podstawowe działanie bramek. Pełny przegląd obu modułów, pakowanie, QPU i pozostałe platformy nadal czekają. |
| 2b | Poprawiono walidację danych i absolutną tolerancję w `encoding_change_unitary.py`. Usunięto ogólny helper `_nullspace`; dla sprawdzonej macierzy (4, 3) rangi 3 wystarcza ostatni wektor SVD. Kod produkcyjny: 3 linie mniej. | Regresje przed zmianą: 28 błędów, 13 zaliczonych. Po zmianie: 41/41; z zależnymi testami: 133/133. Pokrycie modułu po testach i zachowanym przykładzie: 97%. Niezależne review Ponytail: `CLEAN`. | Celowo odrzucamy kodowania przekraczające `tol`, które wcześniej przepuszczał domyślny `rtol`. Bez pełnego ponownego zestawu offline i bez QPU. |
| 2c | Poprawiono walidację `basis` i usunięto przybliżanie jej małych faz przez Weyla. Ścieżka przygotowania bez `E_new` korzysta z istniejącego konstruktora bazowego. Usunięto budowanie macierzy jednostkowej w `change_basis`. Liczba linii pliku produkcyjnego bez zmian. | Przed poprawką: 14 błędów, 41 zaliczonych. Po zmianie: 55/55; z zależnymi testami: 188/188. Pokrycie instrukcji modułu: 100%. Niezależne review Ponytail: `CLEAN`. | Bardziej rygorystyczna walidacja `basis`; poprawne małe fazy są teraz realizowane. Bez pełnego ponownego zestawu offline, pakowania i QPU. |

### Pakiet 2a — decyzje i dowody

Data: 2026-10-24. `quantum_circuits_resource` i `quantum_circuits_path`
pozostają bez zmian. Używają `importlib.resources`; odczyt nie zależy od katalogu
roboczego ani od `_REPO_ROOT`. Dotychczasowy loader zamyka strumień po odczycie
i przekazuje wyjątki pliku lub QPY. Ponowne użycie zachowuje te własności.
Nazwy ośmiu globalnych obwodów X/Z, ich kolejność ładowania i publiczne eksporty
pozostają zgodne. Nie dodano cache, zależności ani nowego interfejsu.

| Testy | Decyzja i źródło oczekiwanego wyniku |
| --- | --- |
| Zasoby i konstruktor, wymiary 3/4 | Zostawić odczyt wszystkich 12 QPY i sprawdzenie ścieżki. Zastąpić samą normę stanem grafowym o amplitudach `exp(2πiab/d) / d`, osadzonym w dwóch parach kubitów. Test zmienia katalog roboczy i `_REPO_ROOT`. |
| Dwie strategie zmiany kodowania | Zastąpić porównanie dwóch wywołań konstruktora niezależnym stanem logicznym i `kron(E, E)`. Użyć niekanonicznego kodowania z fazami zespolonymi. |
| X/Z, XZ i ich adjointy, wymiary 3/4 | Dodać cztery przypadki parametryczne. Oczekiwane macierze wynikają z przesunięcia cyklicznego i faz `exp(2πia/d)`. Sprawdzać pełne macierze z tolerancją `1e-10`, także kolejność iloczynu i działanie na czwartym poziomie. |

Porównania stanów dopuszczają fazę globalną. Porównania macierzy stabilizatorów
jej nie pomijają. To chroni fazy istotne przy obliczaniu wartości oczekiwanych.

Walidacja używa środowiska i wyłączonych flag opt-in z etapu 1.
Dodatkowo `MPLCONFIGDIR=/tmp/qoq-ponytail-matplotlib` kieruje cache wykresów
do katalogu z prawem zapisu. Zestaw po zmianie:

```bash
.venv-ponytail/bin/python -m pytest -q \
  tests/test_package_data.py tests/test_clean_repo_smoke.py \
  tests/test_reference_regressions.py tests/test_reference_experiments.py \
  tests/test_reference_measurement_integration.py tests/test_public_api.py \
  tests/test_experiment_aer_integration.py -ra --durations=10
```

Wynik: 92 zaliczone w 10,19 s. Osobny przebieg ośmiu testów zasobów z coverage
potwierdził wykonanie 10/10 dodanych lub zmienionych linii produkcyjnych.
Nie jest to pomiar pokrycia całego repo ani całego zestawu testów.
Dowody, kopie plików sprzed zmiany, pokrycie i pomiary czasu:
`artifacts/ponytail/stage2a-20261024/`. Nie powtarzano pełnego zestawu offline.
Porównanie początkowych hashy potwierdziło brak zmian QPY i zapisanych danych.
Przegląd `generate_b_ame` obejmuje tutaj wyłącznie ładowanie QPY i wybór bramek
w stabilizatorach. Moduły z nieprzejrzaną resztą kodu nadal mają stan `czeka` w spisie.
Niezależny reviewer sprawdził pełny diff, lokalne wywołania, eksporty, obsługę
strumieni, wzory matematyczne i dowody walidacji. Wynik: `CLEAN`, bez usterek.
Następny mały pakiet: 2b, walidacja i budowa zmiany kodowania w
`core/encoding_change_unitary.py`.

### Pakiet 2b — decyzje i dowody

Data: 2026-10-24. Sprawdzono cały `core/encoding_change_unitary.py`, jego użycie
w konstruktorze obwodów, starym benchmarku oraz `encoding_search/triviality.py`.
Graphify nie wskazał testów jednostkowych modułu. Dodano
`tests/test_encoding_change_unitary.py`; spis ma teraz 820 wpisów.
Początkowe hashe pozostają bez zmian, a nowy plik ma pusty hash początkowy.

Potwierdzone problemy i poprawki:

- Przy `tol=1e-10` macierz `(1 + 1e-7) * eye(4, 3)` przechodziła walidację.
  Zwrócona W miała błąd unitarności około `2e-7`. Domyślny `rtol=1e-5`
  poszerzał próg. Wszystkie kontrole matematyczne obu funkcji używają teraz `rtol=0`.
- NaN, nieskończoność i dane tekstowe powodowały błędy NumPy zamiast raportu.
  Walidator sprawdza dane przed SVD. Błędne kodowanie daje raport z `is_valid=False`,
  a konstruktor zgłasza dotychczasowy `ValueError` z informacjami walidacji.
- Błędny `tol` zgłasza czytelny `ValueError` w obu publicznych funkcjach.
  Dozwolona jest skończona nieujemna liczba rzeczywista, także zero.
  Bool, tekst, wartości zespolone i tablice nie są tolerancją skalarną.
- Obliczenia używają float64 lub complex128. Usuwa to błędy arytmetyki bool
  i brak obsługi float16 w SVD. Dane wejściowe pozostają bez zmian.
- Zachowano klucze raportu, oba publiczne interfejsy, normalizację wektora
  dopełnienia oraz końcowe kontrole W. Usunięto zbędne kopiowanie `W.astype(complex)`.
  Macierz W już ma typ zespolony. Zachowano przykład `python -m` z trzema scenariuszami.

| Testy / kontrola | Chronione zachowanie i źródło wyniku |
| --- | --- |
| Cztery poprawne kodowania | Kanoniczne, permutacja z fazami, gęste rzeczywiste i gęste zespolone. W musi zawierać E w pierwszych trzech kolumnach i spełniać `W†W=I`. Test nie wywołuje badanego konstruktora do obliczenia oczekiwanego wyniku. |
| Typy liczbowe i zero tolerancji | Kanoniczne E zachowuje pełną macierz jednostkową W. Dane wejściowe nie zmieniają się. |
| Tolerancja absolutna | Ten sam błąd około `2e-7` musi zostać odrzucony przy `1e-10` i zaakceptowany przy `1e-6`. |
| Błędne kodowania i konfiguracja | Niepoprawny typ, kształt, ranga, normalizacja, dane nieskończone i nienumeryczne oraz błędne tolerancje. Sprawdzane są raport walidatora i wyjątek konstruktora. |
| Porównanie ze starym kodem | 40 kodowań QR o ustalonym seedzie, rzeczywistych i zespolonych: największa różnica W wyniosła 0. Wszystkie cztery zapisane `E.npy` nadal przechodzą walidację i dają tę samą W. |
| Odbiorca w starym benchmarku | Trzech wygenerowanych kandydatów oraz obwody W zachowują kodowanie i unitarność. Bez syntezy i sprzętu. |

Walidacja w środowisku i z wyłączonymi flagami opt-in z etapu 1:

```bash
.venv-ponytail/bin/python -m pytest -q \
  tests/test_encoding_change_unitary.py tests/test_package_data.py \
  tests/test_clean_repo_smoke.py tests/test_reference_regressions.py \
  tests/test_reference_experiments.py tests/test_reference_measurement_integration.py \
  tests/test_public_api.py tests/test_experiment_aer_integration.py -ra
.venv-ponytail/bin/python -m qudits_on_qubits.core.encoding_change_unitary
```

133 testy zaliczone, bez pominięć, 8 zastanych ostrzeżeń IBM. Przebieg z coverage:
21,22 s pytest / 24,11 s całego procesu. Sam pytest wykonał 66/109 instrukcji
modułu. Zachowany przykład wykonał kolejne 40; łączne pokrycie to 106/109 (97,25%).
Pozostałych trzech gałęzi diagnostycznych nie wymuszano atrapami dla samego procentu.
Nowe angielskie komunikaty i komentarz przeszły kontrolę `asd-ste100`.
Dowody i kopie plików sprzed zmiany: `artifacts/ponytail/stage2b-20261024/`.

Zastany import `from encoding_change_unitary import ...` w
`encoding_search/triviality.py` nie korzysta ze ścieżki pakietu, a wyjątek jest tam
przechwytywany. Pozostaje do oceny razem z logiką równoważności w etapie 4.
Nie zmieniano tego modułu ani formatów zapisanych danych.
Niezależny reviewer sprawdził cały pakiet, odbiorców i dowody. Dodatkowo
sprawdził 12 zespolonych izometrii, także w tablicach tylko do odczytu.
Potwierdził działanie W bez zmiany danych. Wynik review: `CLEAN`, bez usterek.
Następny mały pakiet: 2c, konstrukcja obwodów w `core/create_ame_circuit.py`.

### Pakiet 2c — decyzje i dowody

Data: 2026-10-24. Przejrzano cały `core/create_ame_circuit.py`, eksporty,
wywołania w starym benchmarku i `generate_b_ame` oraz użycia w notebookach.
Sygnatura i publiczne eksporty pozostają bez zmian. Konstruktor buduje stany
grafowe; własność AME zależy od grafu. Uściślono opis funkcji.

Potwierdzone błędy i poprawki:

- `(1 + 1e-7) * eye(4)` przechodziło walidację `basis` przez zaokrąglanie
  do sześciu miejsc. Teraz wymagamy numerycznego `numpy.ndarray` o kształcie
  (4, 4), skończonych wartości i unitarności z `atol=1e-10, rtol=0`.
  Nie dodano nowego argumentu tolerancji. `change_basis` nadal zwraca niezależną kopię.
- Domyślne `fidelity=1-1e-9` w Weylu usuwało małe fazy poprawnej bazy.
  `fidelity=1.0` też usuwało fazę `1e-8` wskutek arytmetyki zmiennoprzecinkowej.
  Użyto `fidelity=None`, które wyłącza tę specjalizację.
  Dla kąta `1e-5` maksymalny błąd amplitudy zmalał z około `8,9e-6` do `4,7e-16`,
  a dla `1e-8` z `8,9e-9` do `2,4e-16`. Porównania pomijają jedną fazę globalną.
- `change_basis` mnożyło macierz przez odtworzoną jednostkową bazę obliczeniową.
  Wystarcza sprawdzenie i kopia macierzy. Obliczenia używają float64 lub complex128.
- Strategia przygotowania przy `E_new=None` powtarzała bazowy obwód F/CZ.
  Korzysta teraz z `_build_circuit_append_w`. Warunek `dim=3` pozostaje przed tą ścieżką.

Nie zmieniano reguł grafu: równoległe krawędzie powtarzają CZ, a pętle nie dodają
bramek. Przekazany graf nadal ma pierwszeństwo przed `n` i `graph_type` oraz wraca
bez zmian jako ten sam obiekt. Puste grafy i pojedyncze wierzchołki pozostają obsługiwane.

| Nowe testy | Chronione zachowanie i niezależny wynik |
| --- | --- |
| Stany grafowe | Dla wymiarów 3/4 i obu obsługiwanych strategii: gwiazda, linia, krawędzie równoległe, pętla i izolowany wierzchołek. Amplitudy logiczne wynikają z `exp(2πi sum(a_u*a_v)/d) / d**(n/2)`. Test osobno koduje stan w parach kubitów, z wierzchołkiem 0 na najmłodszej parze bitów. |
| Kodowanie i `basis` | Gęste zespolone kodowanie i lokalna baza działają jako iloczyn tensorowy. Sprawdzono bazę Fouriera oraz fazy `1e-5` i `1e-8`, z `E_new` i bez niego. |
| Walidacja i zgodność | Błędne argumenty, nieobsługiwane wymiary i strategie oraz niepoprawne bazy. Kopia macierzy nie zmienia wejścia tylko do odczytu. Przypadki 0/1 wierzchołka pozostają poprawne. |

Nowy `tests/test_create_ame_circuit.py` zawiera 55 przypadków; spis ma 821 wpisów.
Sam konstruktor: 55/55 zaliczonych. Zależny zestaw w środowisku etapu 1,
z wyłączonymi flagami opt-in i `MPLCONFIGDIR=/tmp/qoq-ponytail-matplotlib`:

```bash
.venv-ponytail/bin/python -m pytest -q \
  tests/test_create_ame_circuit.py tests/test_encoding_change_unitary.py \
  tests/test_package_data.py tests/test_clean_repo_smoke.py \
  tests/test_reference_regressions.py tests/test_reference_experiments.py \
  tests/test_reference_measurement_integration.py tests/test_public_api.py \
  tests/test_experiment_aer_integration.py -ra
```

188 testów zaliczonych, bez pominięć, 8 zastanych ostrzeżeń IBM.
Przebieg z coverage: 18,15 s pytest / 20,92 s całego procesu.
Wykonano 113/113 instrukcji modułu (100%). To nie jest pomiar wszystkich gałęzi ani całego repo.
Porównanie 20 przypadków ze starym konstruktorem potwierdziło zgodność stanów
z tolerancją `1e-12`, z pominięciem fazy globalnej. Dla 15 przypadków bez `basis`
liczba operacji pozostała taka sama. Nie określa to kosztu wykonania na QPU.
Odbiorca w starym benchmarku dał sześć zgodnych stanów grafowych: `two_qutrit`,
`ghz3` i `ame43`, każdy dla obu strategii. Kontrola używa niezależnego wzoru faz
dla grafu, nie potwierdza osobno definicji grafów ani własności AME.
Nowe angielskie opisy, komunikat i komentarz przeszły kontrolę strukturalną STE.
Dowody i kopie plików sprzed zmiany: `artifacts/ponytail/stage2c-20261024/`.
Nie powtarzano pełnego zestawu ani pakowania. Nie zmieniano danych i QPY.
Niezależny reviewer sprawdził cały pakiet i dowody. Dodatkowo sprawdził trzy stany
czterech quditów, gęste zespolone bazy i kodowania, asymetryczny multigraf oraz
tablice tylko do odczytu. Potwierdził też zachowanie fazy globalnej pełnych
operatorów bazy. Wynik review: `CLEAN`, bez usterek.
Następny mały pakiet: 2d, generatory i nazwy grafów w `core/graph_states.py`.


## Zamknięcie planowanej pracy

Data: 2026-10-25. Etapy 2–11 zakończono w ustalonym zakresie offline.
Trzy równoległe zadania objęły core/Bell, benchmarki oraz eksperymenty.
Integracja objęła przykłady, notebooki, dane, dokumentację i dystrybucję.
Każdy plik w spisie ma decyzję i uzasadnienie. Brak zmiany oznacza świadome
zachowanie potrzebnego kodu lub dowodu badawczego, nie pominięcie pliku.
Pełne wykonanie historycznych notebooków wymagających sprzętu nie należało
do lokalnej walidacji. Ich ograniczenia zapisano osobno.

Końcowy spis zawiera **836 plików i zero decyzji oczekujących**.

| Etap | Wynik i uzasadnienie | Walidacja |
| --- | --- | --- |
| 2d–3 | Naprawione indeksy stabilizatorów, korzeń grafu, grupy wag, absolutne tolerancje, wspólne wyniki rejestrów i wieloustawieniowe inplace. Ewolucja stanu zastąpiła gęste macierze. | 359 zaliczonych, 1 pominięty. Siedem zmienionych modułów ma 80,77–100% pokrycia instrukcji. [Szczegóły](ponytail_stages2_3.md). |
| 4–6 | Usunięte kopie helpera ścieżek, martwa kontrola W i nadpisana definicja CSV. Uszkodzony zapis nie nadpisuje wyników suite. Konwersja QPY publikuje plik dopiero po poprawnym zapisie. | 1252 zaliczone w szerokim zestawie, 1 pominięty. Dodatkowy test CLI konwertera: zestaw 3/3. [Szczegóły](ponytail_stages4_6.md). |
| 7–8 | Usunięto 25 nieosiągalnych helperów i podwójny odczyt artefaktów. Naprawiono NPZ, typ wersji schematu i odmowę zgadywania marginałów wielu rejestrów. | 1200 zaliczonych, 9 pominiętych. Zmienione moduły łącznie 81,73%; manifest osobno 77,20%. [Szczegóły](ponytail_stages7_8.md). |
| 9 | Czyste repo zapisuje dirty_worktree=False. Testy awarii zachowują dowody i potwierdzone ID bez ponownego zlecenia. | Vertical slice i kontrola paczki: 75 zaliczonych; runner 81%, cały vertical_slice 86%. Składnia 401 komórek w 25 notebookach poprawna. [Szczegóły](ponytail_stages9_11.md). |
| 10 | Zachowano wszystkie dane i raporty. | Hashe wszystkich 460 plików zgodne z początkiem. Pełny bootstrap: 43 zadania, 240 wierszy, 2000 replik; wartości, SE i CI zgodne z publikacją przy 1e-9. |
| 11 | Sdist obejmuje moduły agent_relay wymagane przez testy, fixture oraz dokumentację planu. Zakresy zależności pozostają zgodne. | Wheel i sdist zbudowane, wheel przebudowany ze sdist. Oba zainstalowane CLI działają poza checkoutem. Testy z rozpakowanego sdist: 81/81. |

W tym zadaniu kod produkcyjny w src/ i scripts/ zmniejszono o **1066 linii
netto**, względem zastanego drzewa roboczego. Ta liczba nie obejmuje
wcześniejszych pakietów 2a–2c ani zmian testów i raportów. Nie dodano
zależności projektu, nowego frameworka ani nowych kampanii sprzętowych.
Nie usunięto testów zachowania. Z testu zakazu starej ścieżki usunięto
wyłącznie odwołania do skasowanych prywatnych helperów.

Końcowy pełny zestaw po zamrożeniu źródeł: **3278 zaliczonych, 13
pominiętych, 0 błędów**, 128,45 s. Wynik obejmuje dwa rzeczywiste lokalne
runtime BQSKit, Aer, notebooki lokalne, archiwa, procesy potomne i nowe
regresje. Polecenie używało środowiska etapu 1 i wyłączonych flag opt-in:

```bash
.venv-ponytail/bin/python -m pytest -q tests benchmarks/bell_20260907 \
  --ignore-glob='tests/test_ame46_*.py' -ra
```

Dziesięć pominięć dotyczy prywatnego cft-piastq. Pozostałe trzy to
kosztowna rzeczywista synteza CZ3, QPU IQM i selektor z usługą IQM.
1747 ostrzeżeń dotyczy zastanej deprecacji pluginu IBM. Nie wyciszano ich.
Po ostatniej dwuliniowej poprawce walidatorów ponowiono pełny zestaw.
Wcześniejszy wspólny przebieg także przeszedł: 3271 zaliczonych, 13 pominiętych.

Po zebraniu wspólnego zestawu dodano jeszcze test CLI konwertera.
Końcowy osobny zestaw konwertera: 3/3, w tym ten nowy przypadek.
Łącznie zaliczono **3279 różnych przypadków**, a 13 pominięto.
Końcowe zbieranie z repo i rozpakowanego sdist daje po 3292 przypadki.
Nie powtarzano całego zestawu wyłącznie z powodu dodania tego testu.

Pakowanie sprawdzono z czystej kopii wszystkich plików w zakresie.
Usunięto z kopii jedynie ignorowane środowiska i wyniki robocze przez ich
niewłączanie. Budowanie używało `python -m build --no-isolation`.
Zainstalowany wheel korzystał z przygotowanych zgodnych zależności.
Nie wykonywano nowego rozwiązywania całego środowiska z PyPI.
Verifier uruchamia polecenia z `python -I` w katalogu tymczasowym,
sprawdza zasoby QPY, dane theta i zapisane wyniki obu CLI.
To dowód instalacji poza checkoutem, nie pełnej macierzy dystrybucji.

Samodzielny `examples/two_qutrit_encoding_benchmark.py` także wykonano
z `python -I` poza checkoutem na zainstalowanym wheel. Zapisał poprawny
raport i jedną alternatywę Pareto. Wszystkie 133 moduły Python z wheel
są zgodne bajtowo z końcowymi źródłami w repo.

Celowe zaostrzenia dotyczą błędnych macierzy, niecałkowitej wersji
schematu, niepełnych wyników stabilizatorów, niejednoznacznego odczytu
rejestrów oraz inplace=True dla wielu ustawień. Poprawne używane
interfejsy i formaty zapisanych eksperymentów zachowano.

Dowody maszynowe: `artifacts/ponytail/completion-20261025/`, w tym
`full-suite-final.xml`, `full-suite-final.log`, `data-validation.json`,
`static-validation.json`, `bell-bootstrap/verification.json`,
`build-final.log`, `distribution-final.log` i `sdist-tests.log`.
Raporty etapów wskazują osobne dane coverage i mapy decyzji.
Spis zachowuje początkowe hashe; nowe pliki mają puste pole początkowego
SHA-256, ponieważ nie istniały w pomiarze bazowym.

Nie sprawdzono nowych pomiarów QPU, usług IQM/IBM/PCSS, prywatnego PiastQ,
rzeczywistego M3, kosztownej nowej syntezy CZ3 ani Windows i Pythonów
3.11/3.13. Nie ma globalnego pomiaru pokrycia ani dowodu poprawności
wszystkich możliwych grafów. Nie uruchamiano agentów review.
Zachowano AME46 i pozostałe zastane zmiany użytkownika.
Nie utworzono commitu, gałęzi zdalnej ani PR.
