# IQM: DD + ZNE bez twirlingu

**DD + ZNE: 5.777936449**. Granica klasyczna: **5.638155725**.
Różnica od granicy: +0.139780725. Estymata powyżej granicy: tak.

| Wariant | Skala CZ | Bell | Shoty |
|---|---:|---:|---:|
| DD | 1 | 5.541927650 | 50000 |
| DD | 3 | 5.242990503 | 50000 |
| DD | 5 | 4.736356815 | 50000 |
| DD + liniowe ZNE | 0, ekstrapolacja | 5.777936449 | - |

ZNE: nieważony liniowy fit wszystkich trzech punktów 1/3/5. Nie zmieniano modelu po obejrzeniu wyników.
Nachylenie: -0.201392709. Reszty rzeczywiste fitu: [-0.03461609, 0.06923218, -0.03461609].
Dla estymaty ZNE nie wyznaczono niepewności; samo punktowe przekroczenie granicy klasycznej nie potwierdza naruszenia nierówności Bella.

## Parametry

- Ten sam zapisany harmonogram 50 niezależnych lokalnych losowań. W każdej skali 50 obwodów po 1000 shotów; razem 150 000 shotów.
- Wszystkie trzy skale zmierzono w nowej serii. Nie podstawiano wcześniejszego wyniku DD jako skali 1.
- Folding wyłącznie CZ: liczby CZ 5/7, 15/21 i 25/35. Liczby R przed dodaniem DD przez IQM pozostają 12/18; zachowano mapowanie i idealne histogramy.
- Bez twirlingu, korekcji odczytu i postselekcji. DD: STANDARD_DD_STRATEGY IQM, taka sama jak wcześniej.
- Backend: emerald; kubity: ['QB28', 'QB29', 'QB20', 'QB21'].
- Kalibracja: 86eefe25-9e43-4376-ad98-39b63cc0ebbc.

## Porównanie wszystkich wariantów

| Wariant | Bell |
|---|---:|
| RAW, wcześniejsza seria | 5.441336586 |
| Samo DD, wcześniejsza seria | 5.476516947 |
| Sam twirling | 5.293132510 |
| DD + twirling | 5.366634967 |
| DD + twirling + ZNE | 5.565856158 |
| DD, nowa skala 1 | 5.541927650 |
| DD + ZNE, nowa seria | 5.777936449 |

Porównanie obejmuje osobne przebiegi i skale wykonywane kolejno. Różnice punktowe mogą zawierać wpływ dryfu; nie są dowodem istotnej poprawy.

## Job IDs

- Skala 1: 01a0a1c4-a3ea-7cf3-a397-2139363e92eb
- Skala 3: 01a0a1c6-5693-7793-9064-1dd386039f81
- Skala 5: 01a0a1c7-9dfc-79d3-aec9-8729059180d0

## Weryfikacja

Istniejący evaluate_blocks zachowuje wszystkie 1000 shotów każdego bloku w mianowniku; kod 11 ma wagę zero. Osobno zweryfikowano wynik przez agregację zliczeń według ustawienia.
Sprawdzono 3 różne job IDs, wszystkie obwody, kalibrację i 150 000 zliczeń. Odtworzenie offline identyczne. Przedziały Hoeffdinga dla poszczególnych skal w analysis.json / block_analysis_by_scale; audyt w completion_audit.json.
