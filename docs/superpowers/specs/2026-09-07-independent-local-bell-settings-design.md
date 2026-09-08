# Niezależne lokalne ustawienia Bella — specyfikacja v1 raw

Data rozpoczęcia: 2026-09-07. Uzupełniono: 2026-09-08. Status: projekt kompletny do przeglądu; funkcja nie jest jeszcze zaimplementowana.

Plan: [implementacja raw v1](../plans/2026-09-08-independent-local-bell-settings.md).

## 1. Cel i decyzje użytkownika

Dodajemy do głównego pipeline'u tryb niezależnego, równomiernego losowania lokalnych ustawień pomiarowych w blokach. Mierzymy i porównujemy wartości istniejących nierówności dla `two_qutrit`, `ghz3`, `ame43`. Nie zmieniamy definicji nierówności, współczynników, faz, normalizacji ani istniejących artefaktów referencyjnych. Nie deklarujemy testu zamykającego lukę lokalności.

Użytkownik zatwierdził:

- wspólny obwód dla kompletu niezależnie wybranych lokalnych ustawień; jeden shot daje wspólny wynik wszystkich stron;
- dwa podstawowe parametry: minimalna liczba bloków `setting_draws` i shoty na blok `shots_per_draw`;
- losowanie z powtórzeniami: `secrets` w eksperymentach, wstrzykiwany generator ze stałym seedem w testach;
- równomierne rozkłady lokalne, bez wyrównywania liczebności;
- osobne harmonogramy dla porównywanych kodowań i backendów;
- kontynuowanie losowania po minimum aż do pokrycia wszystkich składników nierówności;
- limit wszystkich bloków `max_setting_draws`, sprawdzany przed wysyłką;
- zapis wszystkich wyników, w tym leakage, bloków zerowych i częściowych uruchomień;
- główną wartość Bella z zerowym wkładem leakage oraz dodatkową po odrzuceniu leakage;
- pierwszą wersję wyłącznie raw, bez readout mitigation, twirlingu i ZNE;
- zachowanie starego trybu pomiaru wszystkich wymaganych ustawień;
- obsługę Aer, IBM, IQM i PIAST/AQT już w pierwszej wersji.

Pozostałe szczegóły poniżej są decyzjami technicznymi wynikającymi z tych wymagań. Nie uruchamiamy sprzętu w ramach zadania dokumentacyjnego.

## 2. Scenariusze i pokrycie

Jedynym źródłem definicji jest `src/qudits_on_qubits/reference_experiments.py`.

| Scenariusz | Lokalne ustawienia | Pełne konfiguracje | Wzorce wymagające pokrycia | Konfiguracje pasujące do składnika |
| --- | --- | ---: | ---: | ---: |
| two_qutrit | A0..A2, B0..B2 | 9 | 9 | 9 |
| ghz3 | A0..A2, B0..B2, C0..C1 | 18 | 12 | 12 |
| ame43 | A0..A2, B0..B2, C0..C1, D0..D1 | 36 | 13 | 15 |

A,B losują każdą opcję z prawdopodobieństwem 1/3; C,D — 1/2. Pełna konfiguracja ma odpowiednio prawdopodobieństwo 1/9, 1/18, 1/36 przed zastosowaniem reguły zakończenia.

Pokrycie dotyczy unikalnych wzorców ustawień składników, nie liczby algebraicznych składników. Potęgi i różne współczynniki wykorzystujące te same ustawienia nie wymagają osobnych bloków.

`None` w referencyjnym wzorcu oznacza tożsamość, a nie ustawienie do wylosowania. Dopasowanie zachodzi, gdy wszystkie etykiety inne niż `None` zgadzają się z pełną konfiguracją. Przykładowo `(Ai,B0,None,D0)` pasuje do obu wyborów C. Jeden blok może wspierać kilka składników; nie rozdzielamy ani nie powielamy fizycznych shotów.

GHZ nie używa `(Ai,B1,C1)` ani `(Ai,B2,C1)`. W AME 21/36 pełnych konfiguracji nie pasuje do żadnego składnika. Takie bloki nadal losujemy, wykonujemy i zapisujemy, ale nie wymagamy ich wystąpienia do zakończenia przygotowania. Dla AME wymagamy pokrycia 13 wzorców, nie wszystkich 15 przydatnych pełnych konfiguracji.

## 3. Publiczny kontrakt i kompatybilność

Proponowany interfejs:

```python
ExperimentSpec(
    state="ame43",
    basis=basis,
    backend=backend,
    measurement=RandomizedBlocks(
        setting_draws=1000,
        shots_per_draw=100,
        max_setting_draws=2000,
        max_circuits_per_job=100,
    ),
)
```

Liczby w przykładzie nie są zaleceniem budżetu sprzętowego. Trzy parametry liczebności są wymagane jawnie; brak automatycznego zwiększania limitu. `max_circuits_per_job=100` jest domyślnym ograniczeniem klienta, dodatkowo ograniczanym przez limit dostawcy. `confidence_level=0.95` należy do `RandomizedBlocks` i konfiguruje nową analizę niepewności.

`measurement=None` zachowuje istniejący tryb i dotychczasowe domyślne `shots=20480`. Konstruktor rozróżnia brak argumentu `shots` od jawnego podania go przez istniejący mechanizm `_UNSET`. Jawne `shots` wraz z `RandomizedBlocks` jest błędem; nowy tryb nie ma trzeciego źródła liczby shotów. Serializacja nowego trybu zawiera `shots: null` oraz konfigurację `measurement`; stare słowniki i checkpointy zachowują stare znaczenie. Jawne `shots=None` w starym trybie pozostaje niepoprawne.

Nowy tryb korzysta z własnego wyniku niepewności, a nie `BootstrapBellResults`. Jawna konfiguracja starego `uncertainty`/`bootstrap` wraz z nowym trybem jest błędem z opisem użycia `measurement.confidence_level`. Nie ignorować jawnych opcji. Nowy publiczny typ eksportować z `experiments`; nie udostępniać generatora testowego jako domyślnego źródła produkcyjnego.

Wszystkie liczebności wymagają `type(value) is int`, dodatniej wartości i `max_setting_draws >= setting_draws`. Poziom ufności jest skończoną liczbą z (0,1). Nieobsługiwana mitygacja i nadpisania raw przez `run_options` powodują błąd przed wysyłką. Stary tryb mitygacji pozostaje bez zmian.

## 4. Generowanie harmonogramu

Niech N_min=`setting_draws`, K=`shots_per_draw`, N_max=`max_setting_draws`.

1. Utwórz trwały katalog przygotowania, zapisz konfigurację i hash definicji Bella.
2. Wyznacz uporządkowane lokalne etykiety i unikalne wymagane wzorce z referencji.
3. Dla każdego bloku wykonaj osobne `secrets.randbelow(len(local_labels))` dla każdej strony.
4. Zapisz pełne ustawienia, `block_id`, prawdopodobieństwo wejścia i dopasowane wzorce.
5. Zakończ w pierwszym momencie, gdy N >= N_min i pokryte są wszystkie wzorce.
6. Gdy N=N_max i nadal brakuje pokrycia, zapisz status `coverage_limit_reached`, harmonogram i brakujące wzorce. Nie wykonuj żadnej wysyłki do backendu.
7. Ukończony harmonogram i jego hash zapisz przed wykonaniem.

Nie wybierać brakujących konfiguracji bezpośrednio, nie odrzucać powtórzeń lub zerowych bloków, nie zaczynać losowania od nowa w ukrytej pętli. Pokrycie zależy wyłącznie od ustawień, nigdy od wyników, wartości Bella lub leakage. Pokrycie ustawień nie gwarantuje poprawnych shotów po postselekcji.

Losowanie całego harmonogramu jest lokalne i poprzedza pomiary. Faktyczny koszt raw wynosi N*K, może przekraczać N_min*K i różnić się między porównywanymi ramionami. Raport podaje wszystkie trzy wielkości.

Generator testowy udostępnia tę samą operację wyboru indeksu. Nie inicjalizować stron tym samym seedem w osobnych identycznych generatorach. Zapisać nazwę źródła, wersję generatora i seed tylko w testach; w produkcji nie pozyskiwać ani nie zapisywać wewnętrznego stanu systemowego RNG. Wznowienie odczytuje harmonogram, nie generuje go ponownie.

## 5. Obwody i wykonanie

Rozdzielić katalog unikalnych obwodów od harmonogramu bloków. Dla każdego pełnego ustawienia użyć istniejącego `append_measurement_for_global_setting` i referencyjnego lookupu obserwabli. Zachować konwencję bitów oraz dekoder wyników po rotacji pomiarowej; nie zakładać, że po rotacji fizyczny wynik odpowiada pierwotnej kolumnie dowolnego kodowania E.

Każdy obwód: przygotowanie tego samego stanu w danym ramieniu, lokalne rotacje na wszystkich stronach, odczyt wspólnego wyniku. Operacje lokalne qutritu mogą wymagać bramek między jego dwoma kubitami. Nie mylić ich z operacjami między stronami. Kompilacja i routing pozostają jawne; nie deklarować przestrzennej izolacji na podstawie obwodu.

Unikalne obwody mogą być kompilowane raz, po czym powtarzane jako osobne pozycje listy. Nie scalać wyników powtarzających się bloków na etapie wykonania. Kopie nadają nazwy identyfikujące pozycje bez mutowania wzorca. Nie wracać do mapy `setting -> counts` jako jedynego źródła danych.

Honorować istniejącą `TranspilationConfig` i obsługiwane `workload_optimization` przez wspólny mechanizm kompilacji katalogu pełnych ustawień. Zachować wynikowy wybór layoutu i mapy bitów. Ograniczenia PIAST managed nadal obowiązują; opcji, których adapter nie potrafi zastosować, nie wolno ignorować.

Podział na zadania jest deterministyczny i zachowuje kolejność zadaną. Limit partii to minimum limitu klienta i zadeklarowanego limitu dostawcy. K nie jest zaokrąglane, mnożone ani automatycznie dzielone. Nieobsługiwane K lub limity obwodu/payloadu wykryć w preflight; nie zmniejszać budżetu w ukryciu.

Zapisywać odrębnie kolejność zadaną i znaną kolejność wykonania. Brak gwarancji chronologii po stronie dostawcy oznaczyć `execution_order: unknown`; nie traktować kolejności wyników jako dowodu chronologii fizycznej.

### Backendy

- **Aer:** istniejący adapter i dokładnie K shotów na pozycję. Seed symulacji jest odrębny od generatora ustawień. Dla kolejnych partii stosować odrębne, deterministycznie wyprowadzone seedy symulacji, aby restart generatora przy każdej partii nie powielał strumienia pomiarowego; zapisać je w metadanych.
- **IQM:** istniejący adapter, natywna kompilacja i `backend.run`. Bez klientowego twirlingu, kalibracji readout i ZNE. Zachować mapy kubitów i ograniczenia dostawcy.
- **PIAST/AQT:** istniejący tryb managed. Przekazywać logiczne obwody QPY; kompilację wykonuje zarządzany runner. Nie dopisywać fikcyjnych lokalnych opcji transpile ani nie wymagać lokalnego targetu Qiskit.
- **IBM:** dodać `IBMHardware` i adapter `SamplerV2` do wspólnego rejestru. Użyć istniejącego `_ibm_runtime.runtime_account_options` bez zapisywania zwracanych sekretów. ISA circuits, jawne K, DD i twirling wyłączone. Przy pobieraniu łączyć rejestry klasyczne dla tych samych shotów, nie rozdzielać korelacji między stronami przez osobne marginalne zliczenia. `service.job(job_id)` służy wznowieniu. Nie kopiować ustawień mitygacji z benchmarku wrześniowego.

Aktualny benchmark IBM pozostaje historycznym punktem odniesienia; nowe przykłady korzystają ze wspólnego API, a nie z przepisywania starych recept i wyników. `vertical_slice` zachowuje dotychczasowy kontrakt; nowy protokół udostępniamy w głównym `experiments` i nowym notebooku obejmującym trzy scenariusze oraz cztery backendy.

## 6. Wartości Bella i leakage

### Wybór estymatora

Używamy sumy osobno oszacowanych korelatorów. Nie używamy jako wyniku głównego wcześniejszej propozycji `mean(score / probability)` z losową długością harmonogramu. Reguła zatrzymania zależy od ustawień, a analiza warunkuje na zapisanym harmonogramie.

Dla składnika t:

- c_t = `term.sampling_coefficient()`;
- M_t = indeksy wszystkich bloków pasujących do wzorca t;
- m_t = liczba tych bloków;
- f_t(a) = exp(2*pi*i/3) ** (sum aktywnych potęg razy wyniki modulo 3);
- I_valid(a)=1 tylko wtedy, gdy wszystkie strony mają poprawny wynik qutritowy, również strony z tożsamością w tym składniku.

Dla bloku b z K shotami:

```text
U_bt = (1/K) * sum_shots I_valid(a) * c_t * f_t(a)
V_b  = accepted_shots_in_block / K
B_raw = sum_t mean_b_in_M_t(U_bt)
B_conditional = sum_t [sum_b_in_M_t(U_bt) / sum_b_in_M_t(V_b)]
```

Są to odpowiednio sumy liczników podzielonych przez wszystkie dopasowane shoty oraz przez poprawne dopasowane shoty. Nie mnożyć ponownie przez 1/p_t: częstotliwość ustawień jest już uwzględniona przez mianownik danego korelatora. Nie dzielić B_raw przez globalny odsetek akceptacji i nie uśredniać warunkowych średnich bloków z równymi wagami.

Bloki bez dopasowania nie są używane w mianowniku konkretnego korelatora, lecz pozostają w raw, całkowitym koszcie i statystykach leakage. To konsekwencja wybranego estymatora korelatorów, a nie odrzucanie bloków z protokołu. Nie dodawać zerowych współczynników do samej nierówności.

Jeśli m_t=0 z powodu niekompletnego wykonania, pełne wartości są niedostępne. Jeśli m_t>0, ale wszystkie dopasowane shoty mają leakage, raw wkład t wynosi zero, natomiast warunkowy składnik i pełny wynik warunkowy są `null` z przyczyną `no_accepted_shots`. Obliczone pozostałe składniki nadal zapisujemy. Brak danych nigdy nie jest zerem lub NaN w JSON.

### Znaczenie marginałów AME

Dla tożsamości sumujemy po lokalnych wynikach pomijanej strony. Jej leakage nadal podlega globalnej regule akceptacji zgodnie z decyzją użytkownika i dotychczasową metryką. Zachować zliczenia per pełna konfiguracja oraz liczebności kontekstów użytych w każdym składniku.

Na idealnych lokalnych pomiarach projektory akceptacji sumują się do projektora kodu niezależnie od ustawienia, więc marginały odtwarzają operator referencyjny. Na sprzęcie dodatkowe rotacje lub przesłuchy mogą zmieniać statystyki: estymujemy wtedy średnie w rzeczywiście zmierzonych kontekstach, nie zakładamy automatycznej równości z dawnym obwodem `None`. Raport zawiera diagnostyczne wyniki per kontekst, bez filtrowania kontekstów po obejrzeniu danych.

### Raportowanie

Zapisujemy części rzeczywistą i urojoną, obie metryki, odsetek globalnego leakage oraz liczby wszystkich i poprawnych shotów globalnie, per blok i per wzorzec. Sumy liczebności per składnik mogą wielokrotnie liczyć ten sam shot i nie są kosztem sprzętowym.

Porównanie z granicą referencyjną ma status diagnostyczny przy jawnych założeniach pomiarowych, szczególnie dla postselekcji. Nie generować p-value nielokalności ani deklaracji zamknięcia luk. Wyniki mitygowane nie należą do v1.

## 7. Niepewność dla zatrzymanego harmonogramu

Wersja pierwsza raportuje konserwatywne przedziały oparte na nierówności Hoeffdinga i sumowaniu prawdopodobieństw błędów. Nie raportuje fikcyjnego standard error ani nie używa starego bootstrapu ustalonych ustawień. Wynik nazywa metodę `conditional_schedule_block_hoeffding_v1`.

Założenie: warunkowo na przygotowanym harmonogramie wyniki różnych bloków są niezależne; dopuszczamy zależności między shotami w jednym bloku. Przedziały dotyczą średnich oczekiwanych po rzeczywiście zmierzonych blokach/kontekstach. Dla interpretacji jako jednej stacjonarnej wartości funkcjonału potrzebna jest stabilność pomiarów i właściwa realizacja lokalnych obserwabli. Metoda nie zabezpiecza przed dowolną pamięcią między blokami lub komunikacją urządzeń.

Niech T oznacza liczbę algebraicznych składników (18,24,26), alpha=1-confidence_level i L=log(6*T/alpha). Dla składnika t o C=abs(c_t) i m=m_t:

```text
radius_U = C * sqrt(2*L/m)       # każda część U jest w [-C,C]
radius_V = sqrt(L/(2*m))         # V jest w [0,1]
```

Utwórz osobne przedziały dla real(mean U), imag(mean U) i mean V, obcinając do fizycznych zakresów. Każdy ma prawdopodobieństwo błędu najwyżej alpha/(3T). Union bound obejmuje wszystkie składniki bez zakładania ich niezależności — istotne przy współdzieleniu bloków AME.

Przedział B_raw jest sumą przedziałów U. Dla B_conditional, gdy dolna granica V>0, dzielimy przedział U przez dodatni przedział V, biorąc minimum i maksimum czterech ilorazów końców; obcinamy wynik do [-C,C]. Jeśli dolna granica V=0, używamy pełnego zakresu [-C,C] dla każdej części warunkowego składnika. Jeśli estymata warunkowa jest niedostępna, jej przedział też jest `null`. Suma przedziałów składników daje przedział całej metryki.

Nawet jeden dopasowany blok lub stałe zaobserwowane wyniki nie dają automatycznie zerowego przedziału. Większe K nie zastępuje większej liczby niezależnych bloków. Przedziały mogą być szerokie; raport pokazuje ich szerokość i liczby bloków, zamiast deklarować dobrą jakość na podstawie samego pokrycia. Węższe modele niepewności mogą być osobnym przyszłym rozszerzeniem.

Dla utrwalonego harmonogramu reguła stopu nie zależy od wyników, więc dobieranie długości przed pomiarami nie zatrzymuje procesu na korzystnej wartości obserwacji. Nie twierdzimy, że ta argumentacja daje nieobciążoność estymatora ilorazowego po postselekcji; jest to estymata warunkowa z jawnie skonstruowanym przedziałem dla ilorazu oczekiwań.

## 8. Trwałość, awarie i wznowienie

Obecny bezpośredni runner zapisuje checkpointy schema 3 i nie wznawia ogólnie nieukończonego wykonania. Starszy walidator `RunManifest` obsługuje schematy 1/2. Nowy tryb dostaje osobny walidowany schemat 4 i jawne rozgałęzienie `run_experiment`/`resume_experiment`; nie rozszerzać po cichu starych checkpointów.

Przykładowe artefakty w katalogu uruchomienia:

```text
experiment.json                # schema 4, stan, spec, hashe i identyfikatory
schedule.json                  # blok, ustawienia, dopasowania, N_min/N_max/N
logical-catalog.qpy            # unikalne logiczne obwody
compiled-catalog.qpy           # jeśli dostępne; PIAST zapisuje reprezentację logiczną
catalog.json                   # ustawienia, indeksy, mapowanie klasyczne, hashe
batches/0000/request.json       # plan pozycji, K, hash, identyfikator próby
batches/0000/submission.json    # potwierdzone job_id i tożsamość backendu
batches/0000/counts.json        # niezmienione zliczenia z odebranej partii
batches/0000/receipt.json       # status, liczebności, pochodzenie, czas dostawcy
analysis/bell.json             # wersjonowany wynik pochodny i diagnostyka
```

Zapis przez istniejący `ExperimentStore`, atomowo, z walidacją ścieżek i hashy. Sekrety, tokeny i obiekty klientów nigdy nie trafiają do manifestu. Zapis nieudanej konfiguracji również musi przejść istniejącą sanitację.

Stany przygotowania: `preparing`, `coverage_limit_reached`, `prepared`. Stany wykonania: `not_submitted`, `running`, `partial`, `completed`, `submission_unknown`, `failed`. Analiza i jakość są osobnymi polami: `pending`, `available`, `partial`, `unavailable`; `quality_flags` zawiera konkretne przyczyny. Słaby wynik nie oznacza awarii wykonania.

Na granicy `ExperimentResult.status`: limit bez pokrycia i przerwane wykonanie z zachowanymi częściami mapują się na FAILED z dokładniejszym stanem w values; niejednoznaczne wysłanie na SUBMISSION_UNKNOWN. Kompletne raw i zakończona analiza dają COMPLETED także wtedy, gdy wynik warunkowy jest niedostępny z powodu pełnego leakage. Błąd samej analizy zachowuje stan POSTPROCESSING umożliwiający lokalne wznowienie. Nie oznaczać niepełnych pomiarów jako COMPLETED.

Przed submit trwale zapisz request i zajmij wyłączną blokadę wysyłki. Po potwierdzeniu zapisz job_id natychmiast. Po odebraniu partii zapisz raw przed analizą i przed kolejną partią. Wszystkie partie preflightuj przed pierwszą wysyłką. Wysyłaj i odbieraj sekwencyjnie w v1, ograniczając liczbę nieodebranych zadań.

Po wznowieniu:

- zweryfikuj artefakty i tożsamość backendu;
- ukończone partie odczytaj lokalnie;
- potwierdzone job_id pobierz przez adapter, nawet jeśli brak lokalnego counts;
- niewysłane partie kontynuuj bez zmiany harmonogramu;
- niejednoznaczne wysłanie wymaga ustalenia statusu u dostawcy; brak ID nie jest dowodem braku zadania;
- ponowna analiza nie wymaga backendu;
- nowy jawny pomiar po potwierdzonej awarii zachowuje poprzednią próbę i jej koszt, bez ukrytego nadpisania.

Nie można zagwarantować odzyskania shotów, których dostawca nie udostępnił. Wszystkie dostępne częściowe dane zachowujemy. Timeout pobierania nie anuluje automatycznie zadania i nie powoduje ponownej wysyłki. Niekompletne zliczenia, zły K lub mapowanie są zapisywane jako dostępny dowód z błędem walidacji i nie wchodzą do kompletnej analizy.

## 9. Weryfikacja i kryteria odbioru

1. Dokładne przestrzenie wejść 9/18/36, wzorce 9/12/13, dopasowania 9/12/15; nie traktować tożsamości jako etykiety do losowania.
2. Skryptowane źródło losowania potwierdza minimum, pierwszy moment pokrycia, limit, zachowanie powtórzeń i zerowych bloków. Limit oznacza zero wywołań submit.
3. Macierzowy oracle z uporządkowanych baz własnych i wartości omega^a odtwarza `logical_bell_operator` z błędem <1e-10, także dla stanów niespecjalnych i dowolnego poprawnego kodowania E. Nie zastępować operatora wyników surową potęgą macierzy obserwabli: jej globalna faza jest rozliczana przez konwencję wyników i `sampling_coefficient`.
4. Surowe i warunkowe estymatory zgadzają się z obecnym postprocessingiem dla zgodnych danych referencyjnych; inne liczby powtórzeń nie zmieniają współczynników nierówności. Testy AME obejmują nakładanie marginałów.
5. Leakage globalne, leakage wyłącznie u strony pomijanej, wszystkie shoty niepoprawne, K=1, małe liczby bloków, null zamiast NaN, przedziały warunkowe z dolną granicą akceptacji zero.
6. Walidacja niepewności przez analityczne zakresy i symulacje z regułą stopu. Przypadek idealnie skorelowanych shotów w bloku nie może dawać zwężenia przedziału o sqrt(K). Dla naruszonych założeń test nie obiecuje pokrycia.
7. Każdy backend: K na pozycję, mapowanie wspólnych wyników, limit partii, brak mitygacji, oddzielne powtarzające się bloki, przywrócenie joba i zapisy surowych danych. IBM testować na wersji zgodnej z pyproject, nie wyłącznie według najnowszej dokumentacji.
8. Awaria drugiej partii zachowuje pierwszą; awaria analizy pozwala wznowić offline; restart po wysłaniu nie powtarza zadania; nieznany submit nie jest automatycznie ponawiany.
9. Stare modele, schematy 1/2/3, notebooki i tryb all-settings przechodzą regresje bez zmiany znaczenia.
10. Nowy notebook bez automatycznej wysyłki sprzętowej demonstruje trzy stany, cztery konfiguracje backendów, oba wyniki, koszt, pokrycie i wznowienie.

## 10. Sprawdzenia wykonane podczas projektowania

Wykonano lokalne obliczenia na definicjach repozytorium, bez pomiarów backendu:

- enumeracja przestrzeni wejść i dopasowań dała liczby z sekcji 2;
- rekonstrukcja operatorów z baz własnych, potęg wyników i `sampling_coefficient`: maksymalne błędy macierzowe 6.97e-16 (two_qutrit), 6.81e-16 (ghz3), 7.16e-16 (ame43);
- enumeracja deterministycznych strategii lokalnych z alfabetem {0,1,2,leakage}, dla równomiernego iloczynowego rozkładu wejść i globalnego zerowania: 4096/65536/1048576 strategii. Maksima wyniosły 5.6381557247154515 / 5.638155724715452 / 7.638155724715452. W kodzie ostatnia granica jest zapisana zaokrąglona jako 7.63816; nie zmieniać jej w tym zadaniu.

Dla składnika z tożsamością enumeracja mnoży aktywne wartości wyników przez średnią akceptację ustawień każdej strony pomijanej. To sprawdzenie nie dotyczy warunkowej postselekcji ani dowolnego empirycznego rozkładu kontekstów w krótkim zatrzymanym harmonogramie. Nie stanowi dowodu poprawności fizycznego backendu.

Nie wykonano jeszcze testów funkcji nowego pipeline'u, ponieważ kod nie został zaimplementowany.

## 11. Źródła

- Definicje i obecne kontrakty: `reference_experiments.py`, `bell_measurements/{sampler_circuits,postprocessing,qiskit_measurements}.py`, `experiments/{models,runner,manifest,store,execution}.py`, `experiments/backends/`, `_ibm_runtime.py`, `benchmarks/bell_20260907/execute.py`.
- Python: [secrets](https://docs.python.org/3/library/secrets.html).
- NumPy: [Random sampling](https://numpy.org/doc/stable/reference/random/).
- IBM: [SamplerV2](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/sampler-v2), [SamplerOptions](https://quantum.cloud.ibm.com/docs/en/api/qiskit-ibm-runtime/options-sampler-options). Sprawdzić zgodność z przypiętym zakresem wersji projektu.
- Hoeffding, [Probability Inequalities for Sums of Bounded Random Variables](https://doi.org/10.1080/01621459.1963.10500830), 1963. Konstrukcja przedziałów w sekcji 7 jest zastosowaniem tej nierówności i union bound do zdefiniowanych tu zmiennych blokowych.
- Hall, [The significance of measurement independence for Bell inequalities and locality](https://arxiv.org/abs/1511.00729).
- Branciard, [Detection Loophole in Bell experiments](https://arxiv.org/abs/1010.1178).

## 12. Poza pierwszą wersją

Mitygacja, optymalizacja nierównomiernego rozkładu, wspólne harmonogramy między ramionami, fizyczny QRNG, certyfikacja nielokalności, dobór liczby bloków na podstawie uzyskanej wartości Bella, automatyczne ponawianie niejednoznacznej wysyłki oraz migracja historycznych wyników na nową interpretację.
