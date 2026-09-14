# Zoptymalizowany eksperyment Bella

Ten katalog zawiera referencję dotychczasowego eksperymentu Bella dla dwóch
qutrytów, z dziewięcioma ustawieniami A0/A1/A2 × B0/B1/B2. Funkcjonał,
kodowanie i obsługa nieużywanego kodu 11 pozostają takie jak w archiwalnym RAW.
Wyniki z kodem 11 mają wagę zero, lecz są zachowane w mianowniku.

## Pochodzenie

`raw_reference.json` odtworzono z `job_005.qpy` kampanii
`artifacts/iqm_bell_by_setting/high_shots_20260911/`.
Przed eksportem sprawdzono fingerprint każdego obwodu względem `manifest.json`.
Plik zawiera jawne instrukcje R/CZ/measure, fizyczne i klasyczne indeksy,
fingerprint rodzica, wagi Bella oraz SHA256 manifestu i payloadu.

Layout archiwalny `[21,29,20,28]` służy wyłącznie do odtworzenia logicznych
przewodów. Nie jest wyborem layoutu dla nowej kampanii. Logiczne kubity B to
`[0,1]`, A to `[2,3]`; klasyczne bity `[0,1,2,3]` mierzą odpowiednio `[2,3,0,1]`.
Klucz zliczeń Qiskit ma kolejność `c3 c2 c1 c0`.

## Obwody i zakres

Moduł `qudits_on_qubits.bell_measurements.bell_optimized` realizuje skrócenie
opisane w dostarczonym `OPTYMALIZACJA_POMIAROW.md`: wspólne przygotowanie
3 CZ + 6 R, następnie lokalne bloki po 2 CZ + 6 R; B0 pozostaje identycznością.
Pełny obwód B0 ma 5 CZ + 12 R, a B1/B2 ma 7 CZ + 18 R.
Parametry są wyliczane numerycznie, bez zaokrąglania faz do ułamków pi.
Synteza korzysta z `TwoQubitBasisDecomposer` i `IQMOptimizeSingleQubitGates`.
Bloki końcowe służą wyłącznie do pomiarów; nie są ogólnymi zamiennikami
bramek przed dalszą interferencją.

Test obejmuje idealne histogramy eksperymentu, idealną sumę Bella 6,
zgodność wag z istniejącym dekoderem i pipeline zliczeń. Zgodnie z korektą
użytkownika nie dodano osobnego eksperymentu ani testu POVM.

## Lokalny benchmark

Ze środowiska z zainstalowanym projektem, w katalogu głównym repozytorium:

```powershell
python -m scripts.bell_optimized_benchmark --output artifacts/bell_optimized/local_run
```

Domyślnie: 16 bloków × 9 ustawień × 2 warianty × 512 shotów = 147456 shotów.
Każda para zawiera stary i nowy obwód. Kolejność ustawień jest losowana;
pierwszeństwo wariantów jest zbalansowane dla każdego ustawienia. Bloki nie są
łączone. Przedziały 95% używają rozrzutu sum między blokami i rozkładu t-Studenta;
różnica wariantów używa sparowanych bloków. Nie wykonuje się ZNE ani postselekcji.

Wyjście zawiera raport, surowe zliczenia, kompletną kolejność, wagi, QPY dla obu
wariantów i trzech baz sprzętowych, parametry syntezy, wersje bibliotek i hashe.
Katalog wynikowy musi być nowy. Niepełny zapis po awarii nie jest wynikiem ukończonym.

IQM R/CZ i IBM CZ/ECR sprawdzane są offline, bez backendu i bez routingu.
Uruchomienie używa idealnego Aer. Nie ma opcji wysyłania zadań sprzętowych.
Przed IBM/IQM trzeba wybrać urządzenie i aktualną kalibrację, sprawdzić fizyczny
layout, końcową mapę bitów i koszt po routingu. Wyniki idealnego Aer nie przewidują
średniej na rzeczywistym procesorze.
