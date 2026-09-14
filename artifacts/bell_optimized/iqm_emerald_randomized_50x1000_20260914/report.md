# IQM: losowany eksperyment Bella, obwody zoptymalizowane

Backend: emerald; layout: [27, 28, 19, 20].
Kalibracja: 86eefe25-9e43-4376-ad98-39b63cc0ebbc.
50 losowań × 1000 shotów w każdym wariancie; razem 200000 shotów.
Twirling: 8 realizacji na blok. ZNE: skale 1, 3, 5, liniowy fit.
Granica klasyczna: 5.638155725. Wynik idealny: 6.

| Wariant | Skala | Bell | Różnica od granicy klasycznej |
|---|---:|---:|---:|
| RAW | 1 | 5.441336586 | -0.196819138 |
| TWIRL_DD | 1 | 5.366634967 | -0.271520757 |
| TWIRL_DD | 3 | 4.746687392 | -0.891468333 |
| TWIRL_DD | 5 | 4.392546048 | -1.245609676 |
| TWIRL_DD_ZNE | 0 | 5.565856158 | -0.072299566 |

Losowanie: istniejący generate_schedule, niezależny uniform A/B, system_secrets. Ten sam zapisany harmonogram we wszystkich wariantach.
Estymator: suma korelatorów uśrednionych w pasujących blokach. Kod 11 ma wagę zero i pozostaje w mianowniku. Bez postselekcji ani korekcji odczytu.
Przedziały Hoeffdinga dla RAW: analysis.json → raw_block_analysis → uncertainty. ZNE: estymata po ekstrapolacji, bez wyznaczonej niepewności; samo przekroczenie granicy nie stanowi potwierdzenia naruszenia.
DD zamówione u IQM w każdej skali twirlingu, STANDARD_DD_STRATEGY. Warianty wykonywane seriami, więc porównanie może obejmować dryf.

## Liczba losowań każdej kombinacji

- A0,B0: 4
- A0,B1: 4
- A0,B2: 6
- A1,B0: 7
- A1,B1: 2
- A1,B2: 9
- A2,B0: 5
- A2,B1: 9
- A2,B2: 4
