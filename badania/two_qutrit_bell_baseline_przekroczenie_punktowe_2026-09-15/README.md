# Two-qutrit Bell — baseline: punktowe przekroczenie granicy klasycznej po liniowym ZNE

Data badania: **15 września 2026**. Backend: **IQM Emerald**. Etykieta badania: **baseline**. Wykorzystano zoptymalizowany obwód two-qutrit Bell z DD oraz ZNE.

> Dla baseline uzyskałem punktowe przekroczenie granicy klasycznej two-qutrit Bell po liniowej ekstrapolacji ZNE. Przekroczenie nie jest statystycznie istotne przy obecnej niepewności.

## Wynik

| Wielkość | Wartość |
|---|---:|
| Granica klasyczna | 5.638155724715451 |
| Liniowe ZNE — estymata B(0) | **5.666470402404844** |
| Przekroczenie punktowe | **+0.02831467768939344** |
| Błąd statystyczny (1σ) | 0.031158148642071087 |
| Przedział ufności 95% | **[5.605401553241439; 5.727539251568249]** |
| p testu liniowości sumy Bella | 0.16634517175726762 |
| Pomiar przy skali 1, z DD | 5.463578075658238 |
| Kwadratowe ZNE z rzeczywistymi momentami skal | 5.5242848185359525 |

Dopisek o przekroczeniu dotyczy **punktowej estymaty po liniowym ZNE**. Przedział 95% obejmuje granicę klasyczną, dlatego wynik nie potwierdza statystycznie naruszenia. Wartość p = 0.1663 dotyczy zgodności sumy Bella z liniowym modelem ZNE, a nie istotności przekroczenia granicy. Przedziały obejmują niepewność zliczeń przy przyjętych założeniach; nie ograniczają błędu samego modelu ekstrapolacji.

## Wykonanie

- Skale ZNE: **1, 1.5, 2, 3, 5**.
- **50 losowań spośród dziewięciu kombinacji ustawień**, 1000 shotów na losowanie i skalę: **250 000 shotów**.
- **DD włączone: STANDARD_DD_STRATEGY IQM. Twirling wyłączony.**
- Dziesięć zadań po 100 obwodów, ze skalami przeplatanymi w każdym zadaniu.
- Cztery warianty foldingu po 250 shotów; dokładnie zadana średnia liczba CZ oraz zapisane rzeczywiste skale każdego wariantu.
- Bez postselekcji i korekcji odczytu. Zdarzenia poza podprzestrzenią qutritu mają wagę zero i pozostają w mianowniku.
- Qubity w kolejności logicznej: **QB28, QB29, QB20, QB21**.
- Kalibracja: `86eefe25-9e43-4376-ad98-39b63cc0ebbc`.

## Zawartość archiwum

- [WYNIK.json](WYNIK.json) — główny wynik, interpretacja, parametry i identyfikatory zadań.
- [WYNIKI.csv](WYNIKI.csv) — wartości pięciu skal i dwóch modeli z niepewnościami.
- [Pełny raport](eksperyment/analysis/report.md) — metoda, bootstrap i ocena modeli.
- [Wykres PNG](eksperyment/analysis/zne_comparison.png) oraz [PDF](eksperyment/analysis/zne_comparison.pdf).
- [Pełna analiza JSON](eksperyment/analysis/analysis.json) i [repliki bootstrap](eksperyment/analysis/bootstrap.npz).
- [Protokół](eksperyment/protocol.json), [harmonogram](eksperyment/schedule.json), obwody QPY, wagi Bella oraz surowe zliczenia w `eksperyment/hardware/`.
- [Audyt zakończenia](eksperyment/completion_audit.json), [niezależna kontrola dekodera](eksperyment/independent_decoder_check.json) i [walidacja](eksperyment/validation.json).
- [MANIFEST_PLIKOW.json](MANIFEST_PLIKOW.json) — SHA256 wszystkich plików archiwum poza samym manifestem.

Plik `eksperyment/local_pipeline_check.json` opisuje wyłącznie lokalny test symulacyjny wykonany przed sprzętem. Zliczenia tego testu nie wchodzą do wyników IQM.

Oryginał badania pozostaje w `artifacts/bell_optimized/iqm_emerald_interleaved_dd_zne_50x1000_20260915/`. Pliki źródłowe skopiowano bez zmiany zawartości; ich historyczne ścieżki i hashe zachowano. Archiwum zawiera wynik i dane badania; odtworzenie analizy korzysta ze skryptów oraz środowiska projektu opisanych w pełnym raporcie.
