# IQM — dziewięć ustawień Bella w stałej kolejności

Zakres użytkownika: uruchomić zoptymalizowany eksperyment Bella na IQM tak jak
poprzednie RAW, bez losowania ustawień. Kolejność: A0/B0, A0/B1, A0/B2,
A1/B0, A1/B1, A1/B2, A2/B0, A2/B1, A2/B2.

Przyjęto poprzedni backend Emerald i budżet 8192 shotów na ustawienie, razem
73728 shotów. Każdy obwód jest osobnym zadaniem; wynik jednego zostaje odebrany
przed wysłaniem następnego. Bez DD, twirlingu, ZNE, kalibracji odczytu i osobnego
testu POVM. Kod 11 pozostaje w normalizacji, z wagą zero.

Plan:

- [x] Sprawdzić poprzednie kampanie, dane dostępowe bez ujawniania sekretów i aktualną kalibrację.
- [x] Dodać testy budżetu, kolejności, checkpointów, kalibracji i statystyki RAW.
- [x] Dodać runner `scripts/iqm_bell_optimized.py`, oparty na istniejącym executorze IQM.
- [x] Przetestować kod i wykonać niezależny review przed wysłaniem zadań.
- [x] Zamrozić obwody, mapę odczytu, kalibrację, metryki, wagi, hashe i protokół.
- [x] Wykonać autoryzowane 9 zadań IQM po kolei.
- [x] Policzyć RAW, 95% CI od szumu zliczeń, porównać z granicą 5.638155724715451 i wcześniejszym RAW.

Polecenia z katalogu repozytorium i zainstalowanego środowiska:

```powershell
python -m scripts.iqm_bell_optimized prepare --output artifacts/bell_optimized/iqm_emerald_sequential_20260914 --shots 8192
python -m scripts.iqm_bell_optimized execute --output artifacts/bell_optimized/iqm_emerald_sequential_20260914
python -m scripts.iqm_bell_optimized analyze --output artifacts/bell_optimized/iqm_emerald_sequential_20260914
```

`prepare` tylko odczytuje IQM. `execute` wysyła zadania i wznawia wyłącznie znane
identyfikatory. Marker o nieznanym wyniku wysłania blokuje automatyczną ponowną
próbę. `analyze` działa offline na kompletnych, zweryfikowanych wynikach.

Wynik punktowy powyżej granicy i cały 95% CI powyżej granicy są raportowane
oddzielnie. Przedział nie obejmuje dryfu ani systematyki. Porównanie historyczne
nie izoluje wpływu optymalizacji, ponieważ kalibracja, czas i layout mogą się różnić.

## Walidacja przed wysłaniem

22 testy zaliczone (14 nowych + 8 istniejącego executora); pokrycie runnera 83%.
Niezależny reviewer, runda 1: CLEAN. Zamrożony layout [27,28,19,20], kalibracja
86eefe25-9e43-4376-ad98-39b63cc0ebbc. Natywne koszty 5/7 CZ oraz 12/18 R;
idealna suma 6.000000000000002. Ponowny odczyt QPY: różnica histogramów 0.

## Wynik sprzętowy

Wykonano 9 odrębnych zadań Emerald, kolejno A0/B0 do A2/B2, po 8192 shoty.
Łącznie 73728 shotów. RAW = 5.453833197540008; SE = 0.025386606226970095;
95% CI od szumu zliczeń [5.404076363645446, 5.50359003143457].
Granica klasyczna 5.638155724715451 nie została przekroczona.

Poprzedni RAW high: 5.301703827343012; poprawa +0.152129370196996.
Poprzedni RAW readout: 5.26158870236851; poprawa +0.192244495171498.
Wyników nie interpretujemy jako izolowanego efektu optymalizacji: nowa seria
ma również nową kalibrację i inny wybrany layout.

Ponowna analiza identyczna; niezależna suma z surowych zliczeń zgodna do 1e-12;
sprawdzono 9 różnych identyfikatorów zadań, po 8192 shoty i hashe wejść raportu.
Raport: `artifacts/bell_optimized/iqm_emerald_sequential_20260914/report.md`.
