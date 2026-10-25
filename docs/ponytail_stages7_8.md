# Ponytail: etapy 7–8 — eksperymenty

Data: 2026-10-25. Zakres: 33 pliki źródłowe i 34 moduły testowe przypisane
do etapów 7–8 w spisie repo. Bez AME46, usług dostawców i wykonania na QPU.

## Wynik

Kod produkcyjny zmniejszono o 945 linii netto. Nie dodano zależności ani nowej
abstrakcji. Naprawiono trzy granice danych i usunięto niewykorzystywaną ścieżkę
prywatnych helperów. Pozostały kod, w tym zapis odporny na przerwanie,
statystyka i kalibracja, pozostaje bez zmian.

| Plik | Decyzja i powód |
| --- | --- |
| `experiments/artifacts.py` | Usunięto dwa pełne odczyty, po których QPY i NumPy ponownie czytały pliki. Ich loadery nadal opakowują błędy odczytu. Archiwum NPZ zapisane jako `E.npy` zgłasza teraz `ExperimentValidationError`, zamiast `AttributeError`. Uchwyt archiwum jest zamykany. Walidacja wymiarów, typu, skończoności i izometrii pozostaje. |
| `experiments/manifest.py` | Wersja schematu wymaga typu `int`. `True`, `1.0` i `2.0` nie są już akceptowane przez równość z liczbą całkowitą. Prawidłowy v1 nadal normalizuje się w pamięci bez zmiany pliku. |
| `experiments/backends/base.py` | Ogólny dekoder nie wybiera już rejestru `meas` ani pierwszego nazwanego histogramu, jeśli wynik ma kilka rejestrów. Bez proweniencji nie można odtworzyć globalnej kolejności bitów z samych marginałów. Taki wynik zgłasza `JobResultError`. Adapter IBM zachowuje osobny dekoder wspólnych strzałów z mapowaniem clbits. |
| `experiments/runner.py` | Wykonanie w pamięci używa istniejących `_validate_submitted` i `_validate_result`. Sprawdza też bezpieczne metadane. Usunięto 25 prywatnych helperów bez ścieżki od publicznych runnerów i bez zewnętrznych wywołań. Plik zmniejszono z 3908 do 2964 linii. |

Usunięte helpery obsługiwały dawną wysyłkę, przygotowanie i wznowienie
niedokończonych eksperymentów v1/v2. Publiczne `resume_experiment` już przed
zmianą odrzucało te wznowienia. Nadal odczytuje ukończone wyniki v1/v2,
ukończone v3, kompletny checkpoint przetwarzania v3 oraz bloki v4.
Nie usunięto mechanizmu wznowienia bloków ani kontroli przed drugim zleceniem.

Przed usunięciem sprawdzono Graphify, statyczny graf wywołań i lokalne użycia
w `src`, `tests`, `scripts`, `examples`, `notebooks`, `benchmarks` i `tools`.
Odniesienia testowe do czterech usuniętych helperów były wyłącznie zakazami
ich wywołania. Z tego testu usunięto te nazwy. Nadal zakazuje odczytu i zapisu
QPY oraz sprawdza wysłanie dokładnie obiektów zwróconych przez kompilator.

## Decyzje dla całego zakresu

Mapa `artifacts/ponytail/stages7_8_decisions.json` zawiera decyzję i polskie
uzasadnienie dla każdego z 67 plików. Agent nadrzędny przenosi decyzje do
wspólnego spisu CSV. Żaden przypisany plik źródłowy nie pozostał bez odczytu.
Duże moduły testowe oceniono grupami zachowań i oczekiwanych wyników.
Nie jest to dowód poprawności każdej możliwej ścieżki.

| Grupa | Decyzja i istotny kontrakt |
| --- | --- |
| Modele, eksporty, `execution`, `ibm_spec`, `safety` | Zostawić: zgodność aliasów i archiwów, jawny tryb wykonania, niezmienność wejść i granica zapisu bez poświadczeń. |
| `store`, manifesty i `raw_evidence` | Zostawić kontrole ścieżek, typów, limitów JSON, hashy, atomowych podmian, fsync i wyłącznych znaczników. Te kontrole chronią wyniki i ID zadania. |
| Harmonogram, przygotowanie i estymacja bloków | Zostawić: niezależne losowania stron, pierwszy pełny prefiks, osobne powtarzające się bloki, iloraz sum oraz leakage wszystkich stron, także widzów. |
| Niepewność i mitygacja | Zostawić: blok jako jednostka granicy Hoeffdinga, union bound bez niezależności składników, parametryczny bootstrap, wspólna kalibracja dla czynników, podpisane quasi-wagi i osobne percentyle części zespolonych. |
| Metryki i wybór workloadu | Zostawić: ocena całego zadania, kalibracja kompletna dla wszystkich kandydatów, jawne granice podgrafu i mapowania fizycznego. |
| Aer, custom i adaptery dostawców | Zostawić podział według rzeczywistych kontraktów. PiastQ zarządza kompilacją, IBM transportuje PUB i wspólne strzały, IQM kompiluje także na ograniczonym podgrafie. |
| Testy | Zachowano wszystkie testy zachowania. Dodano osiem przypadków regresyjnych. Nie redukowano przypadków tylko dlatego, że testują walidację lub błędy zapisu. |

Przykłady niezależnych oczekiwań w istniejących testach: ręczny iloraz sum
przy nierównym leakage, tabela strzałów z permutowanymi bitami, lokalne
projektory pomiarowe, jawny wzór promienia Hoeffdinga, znana linia zespolona
ZNE i ręczne percentyle próbek multinomialnych. Testy zapisów wymuszają błędy
w kolejnych punktach publikacji. Testy dostawców używają lokalnych kontraktów,
a integracje Aer wykonują rzeczywiste lokalne obwody.

## Walidacja

Środowisko i wyłączone flagi opt-in odpowiadają etapowi 1.
Wyniki bazowe dla przydzielonego zakresu: 1192 zaliczone, 9 pominiętych,
28 ostrzeżeń, 17,58 s.

Regresje przed naprawą:

- Artefakty: 1 błąd NPZ, 23 zaliczone przypadki.
- Manifesty: 3 błędy niecałkowitej wersji, 17 zaliczonych przypadków.
- Wynik primitive: 2 błędy utraty rejestru, 2 zaliczone przypadki.

Końcowy przebieg: **1200 zaliczonych, 9 pominiętych, 28 ostrzeżeń, 39,35 s**.
Nowa regresja wielu rejestrów używa rzeczywistych `SamplerPubResult`, `DataBin`
i `BitArray` z zainstalowanego Qiskit. Nie nawiązuje połączenia z dostawcą.
`git diff --check` nie zgłosił błędów. Ponownie sprawdzono pełną zmianę
czterech modułów źródłowych i czterech modułów testowych.

| Zmieniony moduł | Wykonane instrukcje / wszystkie | Pokrycie |
| --- | ---: | ---: |
| `artifacts.py` | 133 / 141 | 94,33% |
| `backends/base.py` | 364 / 428 | 85,05% |
| `manifest.py` | 193 / 250 | 77,20% |
| `runner.py` | 961 / 1201 | 80,02% |
| Razem | 1651 / 2020 | 81,73% |

To pokrycie instrukcji, nie wszystkich gałęzi ani całego repo.
Manifest pozostaje poniżej celu 80%; niewykonane instrukcje obejmują część
odmów dla uszkodzonych pól i błędów odczytu. Nie dodawano testów wyłącznie
po to, aby podnieść procent. Wynik nie dowodzi działania sprzętu.

Dowody: `artifacts/ponytail/stages7_8_tests.xml`,
`artifacts/ponytail/stages7_8_coverage.json` i plik danych coverage obok nich.
Pomiar uruchamia `coverage.Coverage` po imporcie NumPy i Qiskit, a następnie
`pytest.main` z 34 ścieżkami pobranymi ze spisu dla etapów 7–8.
To omija lokalny błąd importu rozszerzeń przy aktywacji `pytest-cov` przed
importem NumPy. Nie zmieniono zależności ani kodu bibliotek.

Nie sprawdzono QPU, sieci IQM/IBM, prywatnego `cft_piastq`, rzeczywistego M3
ani pełnej macierzy systemów i Pythonów. Dziewięć pominięć wynika z braku
`cft_piastq`; ostrzeżenia dotyczą zastanej deprecacji w IBM Runtime.
Cały zestaw offline repo uruchamia agent nadrzędny po zakończeniu integracji.
