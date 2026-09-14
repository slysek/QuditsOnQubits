# IQM: sam twirling — benchmark ukończony

**Bell: 5.293132510**. Granica klasyczna: **5.638155725**.
Różnica od granicy: -0.345023215. Estymata powyżej granicy: nie.

| Wariant | Bell |
|---|---:|
| RAW | 5.441336586 |
| Samo DD | 5.476516947 |
| Sam twirling | 5.293132510 |
| DD + twirling, skala 1 | 5.366634967 |

Sam twirling minus RAW: -0.148204077; minus samo DD: -0.183384438; minus DD + twirling: -0.073502458.
Porównanie estymat z osobnych przebiegów; możliwy wpływ dryfu. Samo przekroczenie granicy przez punktową estymatę nie stanowi potwierdzenia naruszenia.

## Parametry i dokończenie

- 50 zapisanych losowań × 1000 shotów: 8 wariantów twirlingu po 125 shotów. Razem 400 obwodów i 50 000 shotów.
- Dokładnie te same obwody twirlingu co w poprzednim wariancie DD + twirling, skala 1; bez ponownego losowania ustawień ani wariantów twirlingu.
- DD jawnie wyłączone, bez ZNE, korekcji odczytu i postselekcji.
- Backend: emerald; kubity: ['QB28', 'QB29', 'QB20', 'QB21'].
- Kalibracja: 86eefe25-9e43-4376-ad98-39b63cc0ebbc.
- Zachowano pierwsze 37 500 shotów. Po aktualizacji klucza wykonano tylko brakującą partię 12 500 shotów; odrzucona próba pozostaje w archiwum.
- Ostatnia partia została wykonana później; przerwa w wykonaniu może zwiększać wpływ dryfu.

## Zakończone zadania

| Partia | Job ID | Shoty |
|---|---|---:|
| 0 | 01a0a1aa-b8b6-7042-8f91-3f92a33cd7d7 | 12500 |
| 1 | 01a0a1ab-849c-7ea1-aa13-a3bbcd97f08a | 12500 |
| 2 | 01a0a1ac-d8b9-7b23-bcaf-cd64f5ddd4ca | 12500 |
| 3 | 01a0a1b9-9a36-7b03-9fef-8cf7872645ac | 12500 |

Odrzucona wcześniej próba (nie uwzględniana w zliczeniach): 01a0a1b3-9f79-7210-b965-27b8f8bead19.

## Estymator i weryfikacja

Istniejący evaluate_blocks: suma korelatorów uśrednionych w pasujących blokach. Kod 11 ma wagę zero i pozostaje w mianowniku.
Wszystkie 50 bloków zawierają dokładnie 8 różnych wariantów twirlingu i 1000 shotów. Skontrolowano mapowanie bitów, tożsamość obwodów, job IDs i kalibrację.
Odtworzenie offline identyczne. Niezależne obliczenie ze zliczeń pogrupowanych według ustawienia dało ten sam wynik.
Konserwatywne przedziały Hoeffdinga i ich założenia: analysis.json / block_analysis / uncertainty. Przy 50 blokach nie pozwalają traktować niewielkich różnic punktowych jako dowodu poprawy.
Bieżący status: ukończone (completion_audit.json). failure.json, pending_job.json i partial_analysis.json opisują stan historyczny sprzed dokończenia. Wyniki efektywne wskazuje result_manifest.json.
