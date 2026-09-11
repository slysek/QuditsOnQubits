**Diagnostyka statystyczna ZNE: IQM Emerald, pięć skal**

40 jobów, 405 504 shoty, skale 1, 1.5, 2, 2.5, 3. Jednostka analizy: osiem bloków po cztery powtórzenia. Teoria Bell = 6.

Analiza eksploracyjna jednej serii. Uwzględniono korelacje między skalami i dolne ograniczenie kowariancji wynikające z szumu shotów. Wymagane założenia: przybliżona normalność średnich blokowych, niezależność bloków i stabilny proces szumowy. Nie badano eksperymentalnie punktu skali 0.

**Czy ekstrapolacja poprawia zgodność z teorią?**

Test dotyczy odległości oczekiwanej wartości estymatora od 6. Warunki poprawy przy wynikach bazowych poniżej 6: B(0) > B(baza) oraz B(0) + B(baza) < 12. Drugi warunek chroni przed uznaniem nadmiernego przestrzelenia za poprawę. Użyto testów jednostronnych w konstrukcji intersection-union, a następnie korekty Holma dla wszystkich ośmiu porównań. Nie jest to test MSE pojedynczego uruchomienia. Próg porównania: 0.05.

| Wariant | Model | Baza | Zmniejszenie odległości od 6 | p | p po korekcie Holma |
|---|---|---|---:|---:|---:|
| DD | Liniowy | Ten sam wariant przy skali 1 | 0.194450 | 0.000517 | 0.004140 |
| DD | Liniowy | RAW | 0.230824 | 0.004545 | 0.027272 |
| DD | Kwadratowy | Ten sam wariant przy skali 1 | 0.506535 | 0.024322 | 0.101146 |
| DD | Kwadratowy | RAW | 0.542909 | 0.020229 | 0.101146 |
| TWIRLING | Liniowy | Ten sam wariant przy skali 1 | 0.211637 | 0.001034 | 0.007237 |
| TWIRLING | Liniowy | RAW | 0.094890 | 0.087778 | 0.189478 |
| TWIRLING | Kwadratowy | Ten sam wariant przy skali 1 | 0.328410 | 0.063159 | 0.189478 |
| TWIRLING | Kwadratowy | RAW | 0.211663 | 0.184623 | 0.189478 |

**Czy dane wymagają krzywizny?**

| Wariant | Druga pochodna | SE | p dwustronne | p po korekcie Holma dla 2 testów |
|---|---:|---:|---:|---:|
| DD | 0.204066 | 0.098150 | 0.076185 | 0.152370 |
| TWIRLING | 0.066727 | 0.127575 | 0.617086 | 0.617086 |

**Zgodność modelu w zmierzonym zakresie i zgodność z teorią**

Test braku dopasowania wykorzystuje kontrasty ortogonalne do macierzy regresji oraz statystykę Hotellinga T2 z kalibracją F. Przedstawione poniżej p są bez korekty wielokrotności; żaden model nie jest odrzucony nawet przed korektą. Duże p nie dowodzi poprawności modelu ani ekstrapolacji do 0.

| Wariant | Model | p braku dopasowania | p testu B(0)=6 | p równoważności z 6 ± 0.1 |
|---|---|---:|---:|---:|
| DD | Liniowy | 0.289108 | 0.000316 | 0.998757 |
| DD | Kwadratowy | 0.335523 | 0.903712 | 0.339490 |
| TWIRLING | Liniowy | 0.944643 | 0.000034 | 0.999921 |
| TWIRLING | Kwadratowy | 0.928383 | 0.149316 | 0.858320 |

Margines 0.1 jest przykładem tolerancji, a nie wymaganiem użytkownika. Test TOST przy poziomie 5% wymaga, aby cały przedział 90% mieścił się w [5.9,6.1]. Dla kwadratowego DD wynosi on [5.6824,6.3626], więc ten warunek nie jest spełniony. Brak odrzucenia B(0)=6 nie jest potwierdzeniem dokładności.

**Wrażliwość na pominięcie jednej skali**

| Wariant | Model | B(0), wszystkie skale | Najmniejsze B(0) po pominięciu skali | Największe B(0) po pominięciu skali |
|---|---|---:|---:|---:|
| DD | Liniowy | 5.665400 | 5.487365 | 5.749717 |
| DD | Kwadratowy | 6.022515 | 5.492321 | 6.289134 |
| TWIRLING | Liniowy | 5.529466 | 5.474907 | 5.549790 |
| TWIRLING | Kwadratowy | 5.646239 | 5.511274 | 5.684613 |

Usuwanie punktów zwiększa niepewność; zakres ten jest diagnostyką wrażliwości, nie osobnym dowodem odrzucającym model. Szczególnie pominięcie skali 1 zwiększa odległość ekstrapolacji od dostępnych danych.

**Ograniczenia i kolejne potwierdzenie**

Dane wspierają częściową poprawę dla liniowego DD + ZNE, również względem RAW. Nie potwierdzają odtworzenia teorii z małym błędem. Wybór modelu następował po obejrzeniu danych; korekta Holma nie usuwa całego efektu tej selekcji. Niezależna seria powinna mieć model, tolerancję i główny test ustalone przed pomiarami.

Folding w tym eksperymencie skaluje liczbę CZ. Nie zeruje niezależnie wszystkich błędów odczytu i pozostałych operacji; poprawna ekstrapolacja skalowanego kanału nie musi dawać dokładnie 6.

[NIST: test Hotellinga T2](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm) · [NIST: testowanie braku dopasowania z powtórzeniami](https://itl.nist.gov/div898/handbook/pmd/section4/pmd446.htm) · [SAS: test równoważności TOST](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_ttest_details19.htm)

Szczegółowe statystyki, wyniki usuwania każdej skali, średnie blokowe oraz skróty plików wejściowych: statistical_diagnostics.json.
