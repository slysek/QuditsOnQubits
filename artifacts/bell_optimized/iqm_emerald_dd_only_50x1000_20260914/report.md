# IQM: eksperyment Bella z samym DD

Wynik DD-only: **5.476516947**. Granica klasyczna: **5.638155725**.
Różnica od granicy: -0.161638777. Estymata powyżej granicy: nie.

| Wariant | Wartość Bella |
|---|---:|
| Poprzedni RAW | 5.441336586 |
| Samo DD | 5.476516947 |
| Poprzednie DD + twirling, skala 1 | 5.366634967 |

DD minus poprzedni RAW: +0.035180361. DD minus poprzednie DD + twirling: +0.109881980.
Porównanie estymat z osobnych przebiegów; możliwy wpływ dryfu.

## Parametry

- 50 zapisanych losowań, po 1000 shotów; razem 50 000 shotów, jeden job IQM.
- Harmonogram i wszystkie 50 obwodów identyczne z poprzednim RAW. Bez ponownego losowania, twirlingu, foldingu, ZNE i korekcji odczytu.
- DD włączone: STANDARD_DD_STRATEGY IQM, sekwencje XYXYYXYX/asap (próg 9), YXYX/asap (próg 5), XX/center (próg 2). Pełna konfiguracja w protocol.json.
- Backend: emerald; kubity: ['QB28', 'QB29', 'QB20', 'QB21'].
- Kalibracja: 86eefe25-9e43-4376-ad98-39b63cc0ebbc.
- Job ID: 01a0a1a3-4813-7d62-981c-e920e56674f2.

## Estymacja i weryfikacja

Istniejący evaluate_blocks: suma korelatorów uśrednionych w pasujących blokach. Kod 11 ma wagę zero i pozostaje w mianowniku; bez postselekcji.
Pełne przedziały Hoeffdinga i założenia w analysis.json / block_analysis / uncertainty. Przy 50 blokach są szerokie; różnic punktowych nie interpretujemy jako dowodu istotnej poprawy.
Zweryfikowano tożsamość obwodów, harmonogram, job ID, kalibrację oraz 50 000 zliczeń. Odtworzenie offline identyczne. Sumy kontrolne w completion_audit.json.
