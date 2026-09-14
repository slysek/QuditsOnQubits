# Losowany eksperyment Bella na IQM — 50 × 1000

- Istniejący sampler `generate_schedule`: niezależne losowania A/B, minimum i maksimum 50 bloków. Zapis jednego harmonogramu.
- Zoptymalizowany katalog, layout i kalibracja z ukończonego eksperymentu sekwencyjnego. Weryfikacja histogramów i dekodowania względem istniejącego pipeline’u.
- RAW: 50 000 shotów. DD + 8 twirlingów po 125 shotów, skale CZ 1/3/5: po 50 000 shotów. Razem 200 000. Bez dodatkowej korekcji odczytu.
- Rozszerzenie istniejącej kampanii o gotowe obwody natywne i opcjonalną korekcję odczytu; zgodność domyślnego działania notebooka.
- Testy offline: harmonogram, mapowanie, idealny wynik, budżet, ZNE/twirling, odtworzenie, blokady ponownej wysyłki i niewłaściwej kalibracji. Następnie niezależny reviewer.
- IQM: zamrożone wejścia i job IDs; uruchomienie tylko raz. Raport RAW, DD+twirling i ZNE oraz porównanie z granicą klasyczną. Estymata ZNE nie jest samodzielnym dowodem naruszenia.

Walidacja przed IQM (2026-09-14):
- 92 testy przeszły. Pokrycie: nowy runner 81%, istniejąca kampania 93%, razem 88%.
- Zapisano 50 losowań; wszystkie 9 kombinacji pokryte. Częstości: A0/B0=4, A0/B1=4, A0/B2=6, A1/B0=7, A1/B1=2, A1/B2=9, A2/B0=5, A2/B1=9, A2/B2=4.
- Wszystkie 1250 obwodów sprawdzone idealnie; maksymalna różnica po twirlingu/foldingach 6.106e-16.
- Zgodność zoptymalizowanych histogramów ze starym źródłem pipeline’u: 2.041e-7 (stara bramka CZ3 przybliżona); idealny funkcjonał zoptymalizowany 6.000000000000002.
- Emerald, kalibracja 86eefe25-9e43-4376-ad98-39b63cc0ebbc, layout [27,28,19,20] dostępne.
- Przygotowanie: artifacts/bell_optimized/iqm_emerald_randomized_50x1000_20260914. Etap przygotowania nie wysyłał zadań.

Zakończenie:
- Niezależny reviewer: CLEAN. Aktywna gałąź codex/bell-optimized-benchmark.
- IQM ukończyło 13/13 zadań: 200 000 shotów, 13 różnych job IDs, kalibracja zgodna. Odtworzenie offline dało identyczną analizę.
- RAW 5.441336586; DD+twirling: skala1 5.366634967, skala3 4.746687392, skala5 4.392546048; liniowe ZNE 5.565856158.
- Żadna estymata nie przekroczyła granicy 5.638155725. ZNE jest o0.124519572 wyżej od RAW, ale o0.072299566 poniżej granicy.
- report.md, analysis.json i completion_audit.json zapisane w katalogu przygotowania. Wszystkie obwody, zlecenia, job IDs i counts w podkatalogu hardware.
