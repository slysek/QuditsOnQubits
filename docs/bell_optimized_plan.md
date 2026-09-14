# Plan benchmarku Bella

Branch: `codex/bell-optimized-benchmark`. Zachować istniejące zmiany. Bez commitów,
pusha i wysyłania zadań sprzętowych.

Zakres: dotychczasowy eksperyment Bella dla dwóch qutrytów, z obwodami 5/7 CZ
z dostarczonych notatek. Zgodnie z korektą użytkownika bez osobnego testu POVM.

- [x] Odtworzyć dziewięć RAW z zahashowanych QPY, razem z mapą bitów i wagami.
- [x] Wydzielić wspólne przygotowanie i skrócić pięć bloków końcowych pomiarów.
- [x] Sprawdzić idealne histogramy, sumę Bella 6 i koszty 5/7 CZ oraz 12/18 R.
- [x] Przepleść stare i nowe obwody w 16 osobnych blokach po 512 shotów; bez deduplikacji.
- [x] Wykonać pipeline Aer, zapisać QPY, zliczenia, kolejność, wersje i przedziały ufności po blokach.
- [x] Sprawdzić lokalną translację do baz IQM R/CZ oraz IBM CZ/ECR.
- [x] Uruchomić pytest z pokryciem, wykonać benchmark i niezależny przegląd reviewera.

Pliki: `src/qudits_on_qubits/bell_measurements/bell_optimized.py`,
`scripts/bell_optimized_benchmark.py`, `experiment_inputs/bell_optimized/`,
`tests/test_bell_optimized.py`, `tests/test_bell_optimized_benchmark.py`.

Walidacja: istniejący Python `artifacts/piastq-validation-env/Scripts/python.exe`,
`PYTHONPATH=src`. IBM i IQM na sprzęcie to następny etap: aktualna kalibracja,
backend, layout, routing, końcowy koszt i osobne wysłanie.

## Wynik walidacji

- 18 nowych testów: obwody, idealne histogramy, istniejący dekoder Bella,
  bloki, niepoprawne zliczenia, replay i lokalny pipeline Aer.
- 45 testów regresji istniejącego pipeline'u RAW, analizy ustawień,
  referencji i integracji Aer. Łącznie 63 testy zaliczone.
- Pokrycie nowych modułów: 90% (moduł obwodów 91%, CLI 88%). Pomiar przez
  `coverage run --include='*/bell_optimized.py,*/bell_optimized_benchmark.py'`
  omija problem wczesnego importu NumPy przez modułowy tryb pytest-cov.
- Niezależny reviewer, runda 1: CLEAN.
- Pełny lokalny benchmark: 147456 shotów. Referencja 6.00872436;
  optymalizacja 5.97412560, 95% CI [5.91905384, 6.02919737].
- Różnica sparowana: -0.03459876, 95% CI [-0.10796582, 0.03876831].
- Idealna suma obu wariantów: 6. Największa różnica idealnych histogramów:
  8.6e-16. Hashe 13 artefaktów sprawdzone; ponowna analiza daje identyczne wyniki.
- Raport: `artifacts/bell_optimized/local_20260914/report.md`.
- Brak uruchomień sprzętowych. Ostrzeżenia testowe dotyczą deprecjacji
  `IBMFractionalTranslationPlugin` w zainstalowanym qiskit-ibm-runtime.
