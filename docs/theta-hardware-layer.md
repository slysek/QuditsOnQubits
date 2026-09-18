# Dodatkowa warstwa sprzętowa benchmarku theta

Warstwa czyta zapisane obwody z `artifacts/theta_benchmark/full-41points-tol5e-4`.
Nie zmienia oryginalnych wyników. Nowe wyniki trafiają do osobnego katalogu
`artifacts/theta_benchmark/full-41points-iqm-layer`.

## Uruchomienie lokalne

Z katalogu projektu, w środowisku zawierającym zależności projektu:

```powershell
python scripts/run_theta_hardware_layer.py `
  --source-run artifacts/theta_benchmark/full-41points-tol5e-4 `
  --output-dir artifacts/theta_benchmark/full-41points-iqm-layer `
  --snapshot artifacts/theta_benchmark/iqm_emerald_snapshot_20260917.json
```

`--resume` wznawia dokładnie tę samą konfigurację i sprawdza hashe zapisanych
pakietów. Zmiana kodu, źródeł, migawki lub konfiguracji wymaga nowego katalogu.
Próba częściowa: `--indices 0 40 --states two_qutrit`; indeks 0 jest wymagany
do porównania z baseline'em. CLI nie łączy się z dostawcą i nie wysyła zadań.

## Dodatkowe liczby

- `gates.csv`: oryginalne F3/CZ3, natywne R/CZ, po routingu; pełny operator
  sprawdzany po uwzględnieniu permutacji wyjść.
- `summary.csv`: stare koszty przygotowania, natywne przygotowanie, przygotowanie
  po routingu, pełny Bell przed/po routingu; CZ, R, głębokość i głębokość CZ.
- `bell_settings.csv`: osobny wiersz dla każdego ustawienia, wspólny prefiks,
  lokalne pomiary, naprawcze SWAP-y, ścieżka do rzeczywistego QPY.
- `report.md`, `hardware_costs.png`, `hardware_costs.pdf`, `ranking.json`:
  porównanie wszystkich kątów i kandydatów według pełnego kosztu Bella.

SWAP-y są rozłożone na natywne bramki i już wliczone w CZ. Końcowa permutacja
pozostaje w metadanych; samo przestawienie numerów wyjść nie wymaga jej odwracania.
Naprawa zapewnia połączenie wewnątrz każdej pary kodującej kutryt. Końcowe lokalne
bramki przygotowania są scalone z blokami pomiarowymi. Każda strona ma ten sam
lokalny blok dla danej bazy niezależnie od baz innych stron; wspólny prefiks
pozostaje identyczny we wszystkich ustawieniach.

## Zakres porównania

41 kątów, 3 stany, 123 przygotowania, 1394 pełne ustawienia Bella i 82 bramki.
W obrębie stanu każdy kąt używa tego samego podgrafu Emerald i budżetu kompilacji.
Routing porównuje poziomy optymalizacji 3 i 1 dla każdego seedu 0, 1, 2. Poziom 0 jest trybem awaryjnym, gdy oba zmieniają wynik ponad tolerancję 1e-10. To porównanie zasobów przy zadanych ograniczeniach, bez globalnej optymalizacji
wyboru kubitów, modelu szumu i harmonogramu impulsów. DD/ZNE nie są wliczone.
Średnie po ustawieniach mają równe wagi; nie definiują przydziału shotów.

Baseline theta=0 pochodzi z tej samej biblioteki F3/CZ3. Historyczny skrócony
baseline Bella o 5–7 CZ i przygotowana kampania theta40 są osobnymi artefaktami.
Ta warstwa ich nie zmienia. Zadania IQM wymagają osobnego zatwierdzenia użytkownika.

Przed zakończeniem runner sprawdza niezmienność wszystkich plików źródłowych.
Zgodność idealnych histogramów sprawdza poprawność kompilacji; nie przewiduje
sprzętowej wartości Bella. Leakage ma wagę zero i pozostaje w mianowniku.
