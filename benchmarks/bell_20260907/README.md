# Benchmark Bella: odtworzenie na drugim komputerze

Pakiet zawiera **pełną wcześniejszą analizę oraz powtórzenie Fez/Garnet po 5000 shotów**. Można ponownie obliczyć wartości Bella, bootstrap SE/95% CI, różnice F3/baseline i historyczne ZNE wyłącznie z zapisanych wyników prawdziwych QPU. Odtworzenie analizy nie wymaga poświadczeń IBM/IQM i nie zużywa kredytów.

- [Raport: 5000 shotów, Fez + IQM](reports/rerun5000/RAPORT.md)
- [Pełny pierwszy benchmark: Kingston, Marrakesh, Fez, IQM i ZNE](reports/RAPORT_KONCOWY.md)
- [Metody i ograniczenia powtórzenia](reports/rerun5000/METODY.md)
- [Macierze kodowań](reports/coding_bases.json)
- [Manifest danych i sumy SHA-256](manifest.json)

## Instalacja — Python 3.12

W katalogu głównym sklonowanego repo wybierz gałąź PR:

```text
git checkout codex/bell-benchmark-repro
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv-bell
& .venv-bell/Scripts/python.exe -m pip install -r benchmarks/bell_20260907/requirements.txt
& .venv-bell/Scripts/python.exe -m pip install --no-deps -e .
& .venv-bell/Scripts/python.exe benchmarks/bell_20260907/reproduce.py --bootstrap
```

Linux/macOS:

```bash
python3.12 -m venv .venv-bell
.venv-bell/bin/python -m pip install -r benchmarks/bell_20260907/requirements.txt
.venv-bell/bin/python -m pip install --no-deps -e .
.venv-bell/bin/python benchmarks/bell_20260907/reproduce.py --bootstrap
```

Instalacja pobiera biblioteki; późniejszy replay działa offline. `requirements.txt` przypina wersje bibliotek obliczeniowych i SDK użytych w pomiarach. Transytywne zależności przetestowanej instalacji przypina dodatkowo `constraints.txt`. Jest to środowisko benchmarku, a nie instalacja wszystkich opcjonalnych funkcji repo. Pełny bootstrap może potrwać kilkanaście minut, zależnie od CPU. Wyniki trafiają do `artifacts/reproduced-bell-20260907/`. Skrypt odmawia nadpisania istniejącego katalogu; przy następnym uruchomieniu podaj nowy `--output`.

## Tryby odtworzenia

Poniższe komendy zakładają aktywne środowisko i `python` z tego środowiska:

```bash
# Tylko integralność archiwum; wystarczy standardowa biblioteka Pythona.
python benchmarks/bell_20260907/reproduce.py --verify-only

# Wszystkie wartości Bella z counts; SE/CI z zapisanych replik bootstrapu.
python benchmarks/bell_20260907/reproduce.py --output artifacts/bell-quick

# Wszystkie wartości, 2000 replik bootstrapu i kontrola zgodności SE/CI.
python benchmarks/bell_20260907/reproduce.py --bootstrap --output artifacts/bell-full

# Tylko nowa seria 5000 shotów.
python benchmarks/bell_20260907/reproduce.py --series rerun5000 --bootstrap --output artifacts/bell-5000
```

Zalecany do pełnego odtworzenia jest `--bootstrap`. Tryb szybki jawnie zapisuje `bootstrap_recomputed: false`; nie przedstawia historycznych replik jako nowych obliczeń. Pełny tryb porównuje wartości, SE i granice CI każdego joba z publikacją z tolerancją `1e-9`. Sprawdza też wszystkie 144 opublikowane podstawowe wyniki: 96 z wcześniejszej kampanii i 48 z powtórzenia. Założenia bootstrapu opisano w raportach; numeryczna zgodność nie usuwa błędów systematycznych urządzeń.

## Pliki wyjściowe

- `verification.json`: liczba zweryfikowanych zadań, zakres i wynik kontroli zgodności.
- `results.csv`: 240 zagregowanych wierszy dla wszystkich serii i skal; cztery estymatory, SE, CI, liczba shotów i odsetek odrzuceń postselekcji. Serie i backendy mają osobne klucze.
- `differences.csv`: F3 opt − standard, kandydat − baseline z tym samym F3, kandydat opt − baseline standard oraz nowa seria − poprzednia.
- `zne.csv`: wyłącznie poprzednia seria z kompletem skal 1/3/5; intercept liniowy oraz diagnostyka krzywizny. W serii 5000 shotów ZNE pominięto.
- `data/`: rozpakowane oryginalne counts, potwierdzenia, QPY, kalibracje i wartości referencyjne. Pełny bootstrap dopisuje przeliczone analizy przy odpowiednich jobach.

Statyczne raporty, wykresy i oryginalne CSV są już dostępne w `reports/`; nie trzeba uruchamiać kodu, aby je przeczytać. Nazwy kolumn i zestawienie wierszy nowego eksportera mogą różnić się od układu historycznych raportów; wyniki są identyfikowane przez serię, backend, stan, kodowanie, F3 i skalę.

## Zawartość i pochodzenie

`data.zip` ma około 19 MB i 383 pliki. Manifest obejmuje 49 jobów: 43 ukończone (w tym pilot), 5 anulowanych i 1 nieudany. Pilot nie wchodzi do porównania baz. Oryginalne counts, QPY i repliki bootstrapu zachowano bajtowo; każdy plik ma SHA-256. Potwierdzenia nieudanych/anulowanych jobów pozostają w ewidencji, lecz ich niewykonane QPY pominięto. Nie dołączono `.env`, poświadczeń, logów ani plików procesów lokalnych.

Źródłem był commit `156305e8b21d78ba7e729070d86e97e7d21e734e` i lokalna poprawka dekodera nieaktywnego uczestnika AME dla podpór monomialnych. Ta poprawka oraz jej testy są częścią PR. Dołączono też wymagane pomocnicze moduły normalizacji indeksów Qiskit/QPY i odczytu konfiguracji IBM. Inne niezwiązane zmiany z komputera autora nie są częścią pakietu.

Pliki `reports/*provenance*.json` są historycznymi zapisami środowiska i mogą zawierać stare ścieżki autora. Nie są wykonywane. Odnośniki w raportach są przenośne. Wiążący manifest dystrybucji to `manifest.json`; historyczne hashe raportów sprzed zmiany odnośników nie opisują opublikowanej kopii.

`study.py`, `hardware.py`, `target_screen.py`, `execute.py`, `rerun5000.py` zachowują kod projektowania i wykonania historycznej kampanii. `report.py` jest pierwotnym generatorem raportu pierwszego etapu. Obsługiwanym punktem wejścia do odtworzenia publikacji jest **`reproduce.py`**. Historyczne skrypty sprzętowe oczekują dawnego układu `weighted/` i nie służą jako gotowy launcher nowej kampanii.

## Ponowne wykonanie na QPU

Powtórzenie analizy zapisanych counts może odtworzyć liczby. Nowy eksperyment na IBM/IQM będzie miał nowe losowe wyniki, inne kalibracje i może mieć inną jakość. Pakiet nie wysyła nowych zadań automatycznie.

Po rozpakowaniu dokładne wysłane obwody są w `data/jobs/<provider>/<label>/submitted.qpy`; mapowanie ustawień, backend, shoty i opcje są w `receipt.json`. Można je odczytać przez `qiskit.qpy.load`. Przed nowym pomiarem trzeba ustawić własne poświadczenia poza repo, sprawdzić aktualną dostępność/topologię backendu, zgodność ISA, ceny i odrębny budżet. Nie należy ponownie używać historycznych identyfikatorów jobów ani traktować historycznego limitu kampanii jako salda własnego konta.

Seria 5000 shotów: Fez `rep_delay=75e-6`, reset włączony, DD XY4, twirling 8 × 625; IQM Garnet po jednym pełnym wariancie i dwóch kalibracjach na job. Każde ustawienie ma 5000 shotów **przed** postselekcją. Wyniki po postselekcji/korekcji nie stanowią bezlukowego testu Bella. Szczególnie słabe wyniki IQM z powtórzenia pozostawiono w danych; ich przyczyna nie została ustalona.
