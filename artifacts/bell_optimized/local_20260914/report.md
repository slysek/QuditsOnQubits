# Lokalny benchmark eksperymentu Bella

Wykonanie: idealny Aer, bez modelu szumu i bez zadań sprzętowych.

Budżet: 16 bloków × 18 obwodów × 512 shotów = 147456 shotów.

Każdy blok zawiera dziewięć ustawień obu wariantów. Kolejność ustawień jest losowana; 
kolejność starego i nowego obwodu w parze jest zbalansowana między blokami. 
Powtórzenia pozostają oddzielne. Kod 11 zachowuje masę w normalizacji.

| Wariant | CZ (B0 / B1,B2) | R (B0 / B1,B2) | Idealna suma | Aer | 95% CI po blokach |
|---|---|---|---:|---:|---|
| Referencja | 6 / 9 | 14 / 22 | 6.000000000000 | 6.00872436 | [5.94124566, 6.07620307] |
| Optymalizacja | 5 / 7 | 12 / 18 | 6.000000000000 | 5.97412560 | [5.91905384, 6.02919737] |

Największa różnica idealnych histogramów: 8.6e-16.

Granica lokalna: 5.638155724715.

Różnica sparowana (nowy − stary): -0.03459876; 95% CI [-0.10796582, 0.03876831].

W idealnym symulatorze oba obwody dają ten sam rozkład. Różnica estymat wynika ze skończonej liczby shotów. 
Ten test potwierdza obwody, histogramy i lokalny pipeline; nie przewiduje wyniku IBM ani IQM.

Przejście na sprzęt wymaga aktualnej kalibracji, wyboru backendu, weryfikacji fizycznego layoutu, 
mapy bitów i kosztu po routingu. Sprawdzono lokalną translację do baz IQM R/CZ oraz IBM CZ i ECR.

## Artefakty

- `baseline.qpy`, `optimized.qpy`: po dziewięć kompletnych obwodów z pomiarami.
- `*_baseline.qpy`, `*_optimized.qpy`: lokalne translacje baz dla obu wariantów.
- `execution_plan.json`: kolejność każdego wykonania, seed i budżet.
- `samples.json`, `bell_weights.npy`: surowe zliczenia i wagi do ponownej analizy.
- `result.json`: parametry syntezy, koszty, wersje bibliotek i wyniki bloków.
- `files.sha256.json`: hashe artefaktów.
