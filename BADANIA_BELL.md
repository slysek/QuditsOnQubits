# Eksperymenty two-qutrit Bell na IQM Emerald

Branch `codex/bell-optimized-benchmark` zachowuje kod zoptymalizowanego obwodu, testy, protokoły i wyniki benchmarków z 14–15 września 2026. Punktem wyjścia jest commit `d1fcaad33ebf0b46f2589d7dbe24c1563db5cd05`. Publikacja jest osobną gałęzią badawczą.

## Najnowszy wynik

[Archiwum baseline w folderze badania](badania/two_qutrit_bell_baseline_przekroczenie_punktowe_2026-09-15/README.md) zawiera pełny raport, wykresy, niepewności, ocenę modelu ZNE i dane źródłowe.

Przeplatane skale **1, 1.5, 2, 3, 5**, **DD włączone, twirling wyłączony**. Na każdą skalę przypada 50 zapisanych losowań spośród dziewięciu kombinacji ustawień, po 1000 shotów: łącznie **250 000 shotów**. Cztery warianty foldingu po 250 shotów pozwalają dokładnie realizować średnią skalę ułamkową. Analiza modelu kwadratowego uwzględnia rzeczywiste pierwsze i drugie momenty skal.

Liniowe ZNE daje **5.666470402404844**, przy granicy klasycznej **5.638155724715451**. Przedział statystyczny 95% wynosi **[5.605401553241439, 5.727539251568249]**: przekroczenie jest punktowe, bez potwierdzenia istotności statystycznej. Przedział nie obejmuje błędu modelu ekstrapolacji.

## Kod i wcześniejsze wykonania

- [Zoptymalizowany obwód i weryfikacja równoważności](src/qudits_on_qubits/bell_measurements/bell_optimized.py).
- [Lokalny benchmark pipeline'u](scripts/bell_optimized_benchmark.py).
- [IQM: dziewięć ustawień po kolei](scripts/iqm_bell_optimized.py).
- [Istniejący pipeline losowanych ustawień z zoptymalizowanymi obwodami](scripts/iqm_bell_optimized_randomized.py).
- [Niepewność i ocena ZNE dla skal 1, 3, 5](scripts/iqm_bell_randomized_uncertainty.py).
- [Przeplatane skale ułamkowe](scripts/iqm_bell_interleaved_zne.py) i [analiza wyników](scripts/iqm_bell_interleaved_analysis.py).
- [Komplet zapisanych benchmarków](artifacts/bell_optimized): test lokalny, dziewięć ustawień kolejno, baseline losowany, DD, twirling, DD + twirling + ZNE oraz DD + ZNE w dwóch planach skal.

Pliki `local_pipeline_check.json` opisują testy symulacyjne. Wyniki sprzętowe pochodzą z zapisanych zliczeń zadań IQM. Archiwum w `badania` jest kopią najnowszego wykonania; jego manifest obejmuje wszystkie pliki poza samym manifestem. `.gitattributes` zachowuje dokładne bajty zamrożonych danych i kodu, aby konwersja końców linii nie naruszyła SHA256.

## Odtworzenie lokalne

Uruchamiaj polecenia z katalogu głównego tej gałęzi. Środowisko: Python 3.12, zależności z `pyproject.toml`, w tym Qiskit 2.1.x i `iqm-client` 35.x. Wersje użyte w badaniu zapisano w protokołach.

```console
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_bell_optimized.py tests/test_bell_optimized_benchmark.py tests/test_iqm_bell_optimized.py tests/test_iqm_bell_optimized_randomized.py tests/test_iqm_bell_randomized_uncertainty.py tests/test_iqm_bell_interleaved_zne.py tests/test_iqm_bell_interleaved_analysis.py
```

Poniższy przykład odtwarza liniową estymatę i jej błąd ze zliczeń, bez zmieniania archiwum i bez połączenia z IQM:

```python
from pathlib import Path
from scripts import iqm_bell_interleaved_zne as run
from scripts import iqm_bell_interleaved_analysis as analysis

directory = Path("artifacts/bell_optimized/iqm_emerald_interleaved_dd_zne_50x1000_20260915")
protocol, schedule, jobs, weights = run.load_prepared(directory)
stats = analysis.statistics(run.read_samples(directory), weights)
linear = analysis.fit_polynomial(stats, degree=1)
print(linear["value"], linear["shot_se"], linear["shot_ci95"])
```

Pełne zapisane dopasowania, bootstrap i przedziały znajdują się w [analysis.json](artifacts/bell_optimized/iqm_emerald_interleaved_dd_zne_50x1000_20260915/analysis/analysis.json). Szczegółowe założenia opisuje [raport](artifacts/bell_optimized/iqm_emerald_interleaved_dd_zne_50x1000_20260915/analysis/report.md).

Historyczne ścieżki absolutne w protokołach opisują miejsce pierwotnego wykonania. Nie są wymagane do odczytu zliczeń z tej gałęzi. Skrypty sprzętowe wymagają oddzielnego, jawnego trybu wykonania; publikacja i powyższe testy nie zlecają nowych pomiarów.
