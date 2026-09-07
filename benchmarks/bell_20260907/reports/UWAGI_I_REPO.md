# Interpretacja benchmarku i analiza repozytorium

## Co porównuje eksperyment

Wyniki dotyczą całego procesu: kodowania, syntezy bramek, kompilacji i dobranego mapowania kubitów. Obie wersje F3 mają identyczne idealne rozkłady wyników. Różnice sprzętowe obejmują również wpływ mapowania i dryfu urządzenia; nie są izolowanym pomiarem samej fazy dopełnienia F3.

Trzy bazy dla każdego stanu wybrano przed pomiarami z pełnego przeszukania 216 przedstawicieli dyskretnej klasy monomialnej repo. Ranking wykorzystuje kalibracyjny model błędu, nie zmierzonego Bella. Dlatego poprawne są również wyniki, w których wybrany kandydat przegrywa z baseline. Nie wykazano, że są to trzy globalnie najlepsze kodowania według nieznanego rzeczywistego szumu urządzeń ani w klasie ciągłych faz.

`coding_bases.json` zawiera jawne macierze wszystkich wybranych baz, przypisanie logicznych 0/1/2 do fizycznych indeksów i fazy jako potęgi ω = exp(2πi/3). Nazwy w tabelach identyfikują te same macierze na obu dostawcach. `canonical_ez` jest bazą odniesienia.

## Ważne obserwacje sprzętowe

- IBM, pierwsza seria: kubit 16 miał w obwodach kalibracyjnych P(1|0)=0,369140625 oraz P(0|1)=0,322265625. Jest to efektywny błąd przygotowania/odczytu w zastosowanym protokole. Target sprzed pomiaru podawał błąd `measure` 0,006103515625. Ten duży rozdźwięk osłabia ranking oparty wyłącznie na kalibracjach dostawcy.
- Dodatkowa seria `raw-no16` stosuje identyczną procedurę kompilacji dla wszystkich baz i obu F3, ale usuwa kubit 16 z dopuszczalnych instrukcji targetu. Jej katalog kompilacji to `no16/ibm/`. Nie wolno łączyć jej punktu skali 1 z wcześniejszym ZNE dla innych obwodów.
- IQM: AME43 wymaga 104 obwodów Bella, a z kalibracją 106. Dostawca ogranicza pojedynczy job do 100 obwodów. Pierwszy job został odrzucony podczas walidacji, przed wykonaniem instrumentów. Właściwe pomiary AME rozdzielono: `ame-a` obejmuje baseline i pierwszego kandydata; `ame-b` baseline oraz pozostałych dwóch. Każdy estymator ma komplet 13 ustawień. Baseline ma dwa niezależne powtórzenia; raport podaje średnią ważoną liczbą shotów i propaguje niepewność obu kalibracji.
- IQM: `accounted_seconds_upper_bound` obejmuje cały czas od przyjęcia przez station control do ukończenia przez server. To konserwatywne ograniczenie czasu zajęcia QPU, z kompilacją i przetwarzaniem wyników, nie kwota z faktury. Osobno zachowano czas wykonania instrumentów. Wpisy bez takiego dowodu zachowują pełną rezerwę.

## Analiza kodu

1. `src/qudits_on_qubits/reference_experiments.py` definiuje stany referencyjne, oczekiwane wartości Bella oraz ustawienia eksperymentów. Z niego pochodzą użyte operatory i ideały 6, 6, 8.
2. `src/qudits_on_qubits/core/benchmark_encoding_bases.py` generuje bazę monomialną jako podporę trzech stanów w przestrzeni dwóch kubitów, permutację i fazy. 648 nazw redukuje się do 216 przedstawicieli po usunięciu globalnej fazy.
3. `src/qudits_on_qubits/benchmarks/direct_basis/` zawiera syntezę obwodów grafowych i analityczną optymalizację fazy nieużywanego stanu F3. Skonsolidowano ważone krawędzie, w obu ramionach jednakowo. Qutryty uporządkowano od najbardziej znaczącego dwukubitowego bloku, zgodnie z modułem przygotowania eksperymentów.
4. `src/qudits_on_qubits/bell_measurements/` buduje lokalne pomiary, grupuje ustawienia i dekoduje wyniki. W `qiskit_measurements.py` poprawiono rzeczywisty błąd: pomijany uczestnik AME z etykietą `None` pozostawał w niekanonicznej przestrzeni kodowej, mimo że wspólny dekoder uznaje fizyczne 11 za wynik poza kodem. Dla podpór monomialnych wystarczają bramki X przenoszące nieużywany stan na 11. Dla ogólnej izometrii działa pełny dekoder unitarny. Kanoniczny przypadek nie wymaga dodatkowych bramek.
5. `tests/test_bell_identity_spectator.py` sprawdza wszystkie cztery podpory, gęstą izometrię, zerowy koszt przypadku kanonicznego i walidację wejścia. Dalsze testy badawcze porównują wartości Bella z repo oraz pełne rozkłady po zapisie i odczycie QPY.

## Dowody i odtwarzanie analizy

- `selection.json`, `*/exhaustive_compile.json`: pełny ranking przed pomiarami.
- `*/final_compile.json`, `*/verification.json`: kompilacja trzema ziarnami i zgodność rozkładów z ideałem.
- `*/jobs/*/receipt.json`: backend, job ID, shots, mapowanie obwodów, hash QPY i budżet.
- `*/jobs/*/submitted.qpy`, `counts.json`, `metrics.json` lub `timeline.json`: wysłane obwody i surowe dowody dostawcy.
- `*/jobs/*/analysis.json`, `bootstrap.npz`: estymatory i 2000 próbek bootstrap, w tym wspólna kalibracja odczytu dla ramion tego samego joba.
- `results.csv`, `differences.csv`, `resources.csv`, `zne.csv`: tabele raportu; brakująca analiza ZNE pojawia się dopiero po komplecie skal 1/3/5.

Skrypty badawcze znajdują się w katalogu nadrzędnym. Odtwarzanie samych analiz (`analyze.py`, `report.py`) nie wysyła nowych zadań QPU. `execute.py submit` jest osobnym, jawnym działaniem. Nie ponawia niejednoznacznego zgłoszenia; `recover` wymaga identyfikacji i porównania rzeczywistego payloadu.

Nie zmieniono wcześniejszych wyników ani cudzych niezacommitowanych zmian. Nie utworzono commitów, push ani PR.

## Co jest optymalizowane w F3

Dla izometrii kodowania E (macierz 4×3, E†E=I₃) fizyczne dopełnienie bramki ma postać

Uφ = E F₃ E† + exp(iφ)(I₄ − EE†),  gdzie F₃[j,k] = ω^(jk)/√3.

Ponieważ Uφ E = E F₃, każda faza φ działa identycznie na poprawnie zakodowanym qutrycie. Zwykłe F3 przyjmuje φ=0. Analityczna optymalizacja w repo wybiera fazę dopełnienia umożliwiającą tańszą syntezę dwukubitową. Dla użytych baz otrzymano π/2 lub 11π/6. W idealnym eksperymencie poprawa Bella wynosi zatem dokładnie zero; zmierzona różnica pochodzi z realizacji sprzętowej i jej szumu. Koszt pełnego zestawu obwodów, po kompilacji, podano oddzielnie od kosztu samej bramki F3.

Dodatkowo sprawdzono deterministyczne strategie z odrzuceniem na ustawieniach pomijanego uczestnika. Maksymalne wartości pozostają 5,6381557247, 5,6381557247 i 7,6381557247. Dowód enumeracyjny zapisano w `classical_abort_check.json`. Nie usuwa to ograniczeń interpretacji wyników po postselekcji.

## Zamknięcie serii sprzętowych

Dodatkowy `raw-no16` i jego ZNE anulowano przed wykonaniem, każdy za potwierdzone 0 s. Wyłączenie kubitu 16 sprawdzono więc wyłącznie przez lokalną kompilację i dokładną symulację; nie ma wyników sprzętowych tej modyfikacji. Wykonano natomiast pełne, oddzielne porównania tych samych baz na IBM Marrakesh i Fez. Fez ma również komplet skal 1/3/5. Selekcja baz pozostała zamrożona z pierwotnego przeszukania na Kingston i IQM. Wszystkie wyniki, również pogorszenia, są w raporcie końcowym.
