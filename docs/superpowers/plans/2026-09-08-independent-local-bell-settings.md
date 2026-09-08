# Independent local Bell settings — implementation plan (raw v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:subagent-driven-development only if the user subsequently chooses delegation. Steps use checkbox syntax for tracking.

**Goal:** Wdrożyć niezależne, równomierne losowanie lokalnych ustawień Bella w blokach, z pokryciem składników, obiema metrykami leakage, trwałymi wynikami i obsługą Aer, IBM, IQM oraz PIAST/AQT.

**Architecture:** Oddzielić harmonogram od katalogu obwodów oraz danych bloków. Dodać jawny tryb `RandomizedBlocks` do głównego API, osobny runner trwałego wykonania i analizę korelatorów warunkowaną na zapisanym harmonogramie. Zachować istniejący tryb all-settings i jego formaty wyników.

**Tech Stack:** Python w zakresie pyproject, NumPy, Qiskit, Aer, IBM Runtime SamplerV2, istniejące adaptery IQM i PIAST managed, ExperimentStore/QPY/JSON, pytest.

**Spec:** [2026-09-07-independent-local-bell-settings-design.md](../specs/2026-09-07-independent-local-bell-settings-design.md).

Data planu: 2026-09-08. Status: implementacja raw v1 wykonana; przebieg weryfikacji i ograniczenia opisano w [raporcie wdrożenia](../reviews/2026-09-08-independent-local-bell-settings-validation.md). Nie uruchamiano pomiarów sprzętowych. Lista poniżej zachowuje pierwotny plan i kryteria projektowe.

---

## Zasady wykonania

Repozytorium: `C:/Users/szymo/QuditsOnQubits/QuditsOnQubits`. Ścieżki poniżej są dokładnymi ścieżkami względem tego katalogu. Uruchamiać polecenia z jego korzenia w środowisku z zainstalowanym projektem i zależnościami dev.

W trakcie dokumentowania pojawiły się niezwiązane, niezacommitowane zmiany m.in. w models, runner, manifest, store, uncertainty i ZNE. Przed edycją przeczytać bieżący diff; nie zastępować tych plików wersją z wcześniejszego commita. Kotwicami zmian są nazwy funkcji, nie historyczne numery linii.

`docs/superpowers` jest ignorowane przez `.gitignore`. Dokumenty są zapisane lokalnie; nie zmieniać reguły globalnej. Jeśli dokumentacja ma wejść do zatwierdzanego commita, dodać wyłącznie oba konkretne pliki przez `git add -f`.

Plan nie obejmuje mitygacji, nowej nierówności, przebudowy historycznego benchmarku ani uruchomień sprzętowych. Nie wykonywać nowych płatnych prób jako sposobu testowania zapisu lub wznowienia.

W każdym zadaniu: najpierw test zachowania, uruchomienie potwierdzające brak funkcji/błąd, implementacja, ponowne uruchomienie wskazanej grupy testów. Commitować tylko pliki danego zadania i tylko przy przyjętym w sesji przepływie git; nie używać `git add .` w tym checkoutcie.

## Mapa plików

| Plik | Odpowiedzialność |
| --- | --- |
| `src/qudits_on_qubits/experiments/measurement.py` (nowy) | RandomizedBlocks, serializacja konfiguracji |
| `src/qudits_on_qubits/experiments/setting_schedule.py` (nowy) | lokalne etykiety, wzorce, niezależne losowanie, pokrycie i harmonogram |
| `src/qudits_on_qubits/experiments/block_estimation.py` (nowy) | dekodowanie raw, dopasowania, U/V, obie wartości Bella |
| `src/qudits_on_qubits/experiments/block_uncertainty.py` (nowy) | przedziały Hoeffdinga i przedziały ilorazowe |
| `src/qudits_on_qubits/experiments/block_manifest.py` (nowy) | walidacja schema 4 i stanów partii |
| `src/qudits_on_qubits/experiments/block_runner.py` (nowy) | przygotowanie, partie, checkpointy, wykonanie i wznowienie |
| `src/qudits_on_qubits/experiments/backends/ibm.py` (nowy) | IBMHardware adapter SamplerV2 |
| `src/qudits_on_qubits/experiments/models.py` | jawny wybór measurement, IBMHardware, kompatybilność |
| `src/qudits_on_qubits/experiments/preparation.py` | budowa katalogu z pełnych ustawień |
| `src/qudits_on_qubits/bell_measurements/sampler_circuits.py` | opcjonalne jawne ustawienia, zachowanie starej ścieżki |
| `src/qudits_on_qubits/experiments/runner.py` | rozgałęzienie nowego trybu i schema 4 |
| `src/qudits_on_qubits/experiments/{__init__,execution}.py` i `backends/__init__.py` | eksporty i rejestracja IBM |
| `src/qudits_on_qubits/experiments/backends/aer.py` | jawny seed partii w nowym trybie bez zmiany starego kontraktu |
| `notebooks/bell_randomized_raw.ipynb` (nowy), `README.md` | przykłady i interpretacja danych |

Nie rozszerzać starego `uncertainty.py` o nowy estymator; obecne bootstrappy i mitygacja mają zachować swój kontrakt. `vertical_slice` pozostaje ścieżką historyczną.

## Zadanie 1: niezależny oracle matematyczny

**Pliki:** utworzyć `tests/test_randomized_bell_reference.py`.

- [ ] Zapisać testy przestrzeni wejść i dopasowań: 9/18/36 pełnych konfiguracji, 9/12/13 wzorców, 9/12/15 przydatnych pełnych konfiguracji. Wzorzec pasuje, gdy każda etykieta różna od None zgadza się z konfiguracją.
- [ ] Zapisać oracle rekonstruujący operator z konwencji wyników; użyć poniższej kompletnej funkcji testowej. W szczególności NIE używać samych potęg macierzy obserwabli: globalna faza jest już uwzględniona przez `sampling_coefficient`.

```python
import numpy as np
from qudits_on_qubits.reference_experiments import get_reference_experiment

def sampled_operator(reference):
    root = np.exp(2j * np.pi / 3)
    result = np.zeros_like(reference.logical_bell_operator())
    for term in reference.bell_functional.terms:
        active = {factor.party: factor for factor in term.factors}
        product = np.ones((1, 1), dtype=complex)
        for party in reference.state.party_order:
            factor = active.get(party)
            if factor is None:
                local = np.eye(3)
            else:
                basis, _ = reference.observable(factor.setting_label).ordered_eigenbasis()
                diagonal = np.diag([
                    root ** ((factor.outcome_power * outcome) % 3)
                    for outcome in range(3)
                ])
                local = basis @ diagonal @ basis.conj().T
            product = np.kron(product, local)
        result += term.sampling_coefficient() * product
    return result

def test_outcome_convention_reconstructs_every_reference():
    for state in ("two_qutrit", "ghz3", "ame43"):
        reference = get_reference_experiment(state)
        np.testing.assert_allclose(
            sampled_operator(reference), reference.logical_bell_operator(),
            atol=1e-10, rtol=0,
        )
```

- [ ] Rozszerzyć test na dowolny znormalizowany stan, stan referencyjny i stan produktowy; sprawdzać wartości z projektorów pomiarowych, a nie wywoływać dwukrotnie tego samego estymatora.
- [ ] Zapisać test enumeracji lokalnych wyników {0,1,2,leakage}. Dla aktywnej strony leakage daje zero; dla pomijanej strony mnożyć przez ułamek jej ustawień bez leakage. Maksima dla równomiernych wejść: 5.638155724715452, 5.638155724715452, 7.638155724715452. Enumerować partiami 65536 strategii, nie tworzyć globalnej tablicy operatorów. Nie zmieniać zaokrąglonej granicy AME zapisanej w referencji.
- [ ] Uruchomić `python -m pytest -q tests/test_randomized_bell_reference.py`. Oczekiwanie: wszystkie oracles przechodzą na aktualnych definicjach, jeszcze bez implementacji nowego runnera.

## Zadanie 2: konfiguracja i harmonogram

**Pliki:** utworzyć `measurement.py`, `setting_schedule.py`, `tests/test_setting_schedule.py`; zmienić `experiments/models.py` i eksporty.

- [ ] Zdefiniować immutable `RandomizedBlocks(setting_draws, shots_per_draw, max_setting_draws, max_circuits_per_job=100, confidence_level=0.95)`. Pierwsze trzy pola wymagane. Walidacja: dodatnie int bez bool, max >= minimum, skończony poziom ufności w (0,1). Serializacja ma jawne `mode="independent_local_uniform_blocks"` i wersję 1.
- [ ] Zdefiniować immutable `SettingBlock(block_id: int, settings: tuple[str,...], matched_pattern_indices: tuple[int,...])` oraz `SettingSchedule(state, blocks, local_settings, required_patterns, missing_pattern_indices, complete, source, seed)`. Dane wejściowe skopiować do niezmiennych struktur.
- [ ] Implementować `generate_schedule(reference, config, *, _randbelow=None, _source="system_secrets", _seed=None)`. Generator domyślny: `secrets.randbelow`. Jawnie skontrolować, że generator testowy zwraca int w dozwolonym zakresie.
- [ ] Pętla ma następującą kolejność; walidacja i tworzenie obiektów otaczają ten fragment:

```python
uncovered = set(range(len(patterns)))
blocks = []
for block_id in range(config.max_setting_draws):
    settings = tuple(labels[draw(len(labels))] for labels in local_settings)
    matched = tuple(
        index for index, pattern in enumerate(patterns)
        if all(label is None or label == actual
               for label, actual in zip(pattern, settings, strict=True))
    )
    blocks.append(SettingBlock(block_id, settings, matched))
    uncovered.difference_update(matched)
    if len(blocks) >= config.setting_draws and not uncovered:
        break
```

- [ ] Testować pętlę skryptowanym źródłem, nie statystycznym progiem częstości. Dla two_qutrit podać indeksy wszystkich dziewięciu par oraz początkowe powtórzenia; sprawdzić zachowanie powtórzeń, minimum i pierwszego momentu kompletności. Dla źródła zwracającego stale zero limit ma zwrócić niekompletny harmonogram, nie uruchamiać kolejnej pętli.
- [ ] Dla AME użyć pełnej konfiguracji pasującej jednocześnie do `(Ai,B0,None,D0)` i `(None,B0,C1,D0)`; potwierdzić dwa dopasowania i jeden blok. Potwierdzić zachowanie wylosowanej konfiguracji bez dopasowań.
- [ ] `ExperimentSpec`: dodać keyword-only measurement. Użyć `_UNSET` dla domyślnego argumentu shots: stara ścieżka rozwiązuje brak do 20480, nowa do None. Jawne shots w nowym trybie odrzucić. Serializacja nowej ścieżki ma shots=null, measurement=dict, uncertainty=null; rekonstrukcja przekazuje `_UNSET`, a nie None, do argumentów, które nowy tryb sam rozwiązuje. Stare słowniki pozostają identyczne.
- [ ] Odrzucić jawne stare uncertainty/bootstrap w nowym trybie oraz aktywne flagi mitygacji. `measurement.confidence_level` jest jedynym źródłem konfiguracji nowych przedziałów.
- [ ] Uruchomić `python -m pytest -q tests/test_setting_schedule.py tests/test_experiment_models.py tests/test_public_api.py`.

## Zadanie 3: katalog pełnych obwodów

**Pliki:** zmienić `bell_measurements/sampler_circuits.py`, `experiments/preparation.py`; utworzyć `tests/test_randomized_bell_preparation.py`.

- [ ] Dodać do `build_sampler_circuits_for_candidate` keyword-only `explicit_settings=None`. Brak argumentu zachowuje istniejące ustawienia, sortowanie, metadane i liczby obwodów.
- [ ] W nowej gałęzi sprawdzić pełne etykiety właściwych stron, brak None, długość zgodną z referencją i brak obcych etykiet. Katalog deduplikuje konfiguracje w porządku pierwszego wystąpienia; harmonogram nie jest deduplikowany.
- [ ] `prepare_measurements` przyjmuje opcjonalny kompletny harmonogram. Generuje katalog potrzebnych pełnych ustawień przez dotychczasowy helper, zachowując referencyjne terms, encoding_outcome_map i mapy bitów. Nie generować operatorów przez Pauli Estimator.
- [ ] Oddzielić `catalog_index_by_setting` od `catalog_index_by_block_id`. Dwie identyczne konfiguracje mogą mieć jeden wzorzec QPY i dwa odrębne identyfikatory wykonania.
- [ ] Testować dodatkowe konfiguracje GHZ i AME niewystępujące w starej liście; wszystkie muszą mieć prawidłowe lokalne rotacje i pomiary wszystkich stron. Sprawdzić stan po usunięciu końcowych rotacji oraz marginały na niespecjalnym stanie.
- [ ] Testować kanoniczne E, permutację kodowania i gęstą izometrię. Używać istniejącego dekodera wyników po rotacji, nie pozycji nieużytego słowa przed rotacją.
- [ ] Uruchomić `python -m pytest -q tests/test_randomized_bell_preparation.py tests/test_reference_measurement_integration.py tests/test_bell_identity_spectator.py tests/test_bell_circuit_serialization.py`.

## Zadanie 4: obie wartości Bella z danych bloków

**Pliki:** utworzyć `block_estimation.py`, `tests/test_randomized_bell_estimation.py`.

- [ ] Ustalić kontrakt `summarize_terms(reference, schedule, counts_by_block, bit_indices_by_setting, outcome_map)` zwracający uporządkowane próbki per term. `counts_by_block` jest mapą unikalnych block_id na surowe zliczenia; każdy wymagany kompletny blok musi mieć dokładnie K shotów. Mapy mogą wskazywać wspólne ustawienie bez nadpisywania bloków.
- [ ] Dekodować jeden bitstring bloku raz przez `bitstring_to_qutrit_outcomes`. Walidować szerokość, wyniki, nieujemne całkowite zliczenia, brak bool, mapy bitów oraz K. Dla całkowicie poprawnego wyniku policzyć potęgi wszystkich dopasowanych składników; dla leakage dowolnej strony wszystkie ich wkłady tego shota wynoszą zero.
- [ ] Stosować następujący kernel do zdekodowanych zliczeń, zachowując współczynnik dokładnie raz:

```python
import cmath

def term_block_values(term, decoded_counts, expected_shots):
    if sum(decoded_counts.values()) != expected_shots:
        raise ValueError("incomplete block")
    root = cmath.exp(2j * cmath.pi / 3)
    weighted_sum = 0j
    accepted = 0
    for outcomes, count in decoded_counts.items():
        if any(outcome is None for outcome in outcomes):
            continue
        accepted += count
        exponent = sum(
            factor.outcome_power * outcomes[factor.party]
            for factor in term.factors
        ) % 3
        weighted_sum += count * term.sampling_coefficient() * root ** exponent
    return weighted_sum / expected_shots, accepted / expected_shots
```

- [ ] Dla każdego term zebrać U i V tylko z dopasowanych bloków. Raw term = sum(U)/len(U). Conditional term = sum(U)/sum(V). Cała wartość to suma termów, zgodnie ze specyfikacją §6. Nie dodawać 1/p, nie dzielić przez globalny acceptance i nie uśredniać ilorazów blokowych.
- [ ] Zero dopasowanych kompletnych bloków: wynik całkowity null z `missing_blocks`. Suma V=0: warunkowy składnik i całość null z `no_accepted_shots`, raw pozostaje dostępny. W JSON kompleksy zapisywać jako `{real, imag}`, niedostępność jako null plus reason, nigdy NaN.
- [ ] Fixture różnej akceptacji dwóch bloków: U=(1,0), V=(1,0), współczynnik 1. Raw=0.5, conditional=1; nie 0.5. Dodać przypadek U=(1,-0.1), V=(1,0.1): conditional=0.9/1.1, nie średnia (1-1)/2.
- [ ] Fixture powtórzeń i bloków zerowych: zmiana liczby powtórzeń bez zmiany warunkowych rozkładów nie zmienia wartości korelatorów, lecz zmienia koszt i niepewność. Zerowych bloków nie usuwać z raw ani globalnego leakage.
- [ ] Fixture AME: jeden blok zasila dwa wzorce; leakage wyłącznie u strony pomijanej zeruje także ten składnik, zgodnie z zatwierdzoną globalną polityką.
- [ ] Uruchomić `python -m pytest -q tests/test_randomized_bell_estimation.py tests/test_bell_postprocessing_weights.py tests/test_reference_regressions.py`.

## Zadanie 5: niepewność blokowa i interpretacja

**Pliki:** utworzyć `block_uncertainty.py`, `tests/test_randomized_bell_uncertainty.py`.

- [ ] Implementować `term_intervals(U, V, coefficient, confidence_level, term_count)`. Wyznaczać radius_U i radius_V dokładnie według specyfikacji §7. Dla współczynnika zerowego oba zakresy U są [0,0]. Dla pustych próbek zwracać niedostępność.
- [ ] Wspólny helper dzielenia przedziałów:

```python
def ratio_interval(numerator, denominator, bound):
    low, high = denominator
    if low <= 0:
        return (-bound, bound)
    quotients = [
        numerator[0] / low, numerator[0] / high,
        numerator[1] / low, numerator[1] / high,
    ]
    return (max(-bound, min(quotients)), min(bound, max(quotients)))
```

- [ ] Sumować końce przedziałów wszystkich termów. Zwracać real i imag, confidence_level, metodę, założenie niezależności bloków warunkowo na harmonogramie i zakres interpretacji. Nie tworzyć sztucznego pola standard_error=0.
- [ ] Testować małe m, K=1, brak akceptacji, dolną granicę akceptacji zero, licznik o obu znakach, coefficient=0 oraz sumę współdzielących dane składników. Szerokość przedziału nie zależy bezpośrednio od K przy tych samych U,V.
- [ ] Symulacja w teście: losować harmonogram do pokrycia z ustalonym seedem, potem generować niezależne blokowe wyniki o znanych średnich. W każdym bloku powielić jeden losowy wynik K razy, żeby sprawdzić brak fałszywego przyrostu liczby niezależnych próbek. Zweryfikować nominalne pokrycie przedziałów z tolerancją wynikającą z liczby replikacji, nie z arbitralnego ciasnego progu.
- [ ] Raportować wyniki per pełny kontekst AME jako diagnostykę. Nie filtrować kontekstów na podstawie uzyskanego B. Granica referencyjna pozostaje punktem porównania z jawnie opisanymi ograniczeniami, nie generować p-value nielokalności.
- [ ] Uruchomić `python -m pytest -q tests/test_randomized_bell_uncertainty.py tests/test_randomized_bell_estimation.py`.

## Zadanie 6: format trwałego wykonania i wznowienie

**Pliki:** utworzyć `block_manifest.py`, `tests/test_block_manifest.py`, `tests/test_block_checkpointing.py`; wykorzystać publiczne metody `ExperimentStore`.

- [ ] Wdrożyć walidowany schema 4 opisany w specyfikacji §8. API modułu: `validate_block_manifest(document)`, `save_block_manifest(store, run, document)`, `load_block_manifest(store, run)`, `verify_block_artifacts(store, run, document)`. Zapis JSON przez `write_plain_json`, QPY przez `write_circuits`; odczyt publicznymi metodami store. Nie osłabiać istniejącej walidacji ścieżek i reparse points.
- [ ] Walidator wiąże hash harmonogramu z block_id, katalogiem i requests. Sprawdza: unikalność bloków, ciągłość ich indeksów, N_min<=N<=N_max dla kompletnego przygotowania, stop na pierwszym pokryciu po minimum, zgodność każdej partii z harmonogramem, brak duplikacji pozycji między partiami, dokładne K i tożsamość backendu. Nie ufa persisted matched_pattern_indices bez ponownego dopasowania do zapisanej referencji.
- [ ] Każdy request zapisać przed próbą submit. Dodać blokadę wyłączną wysyłki pojedynczej partii; obecność starej blokady nie jest dowodem, że job nie powstał.
- [ ] Obsłużyć macierz wznowienia:

| Dane trwałe | Następny krok |
| --- | --- |
| poprawne counts i receipt | odczyt lokalny, zero submit |
| potwierdzone job_id, brak counts | restore_job i result, zero submit |
| request, potwierdzony stan niewysłany | submit raz |
| zapis rozpoczęcia submit, brak potwierdzenia | submission_unknown; rozstrzygnąć u dostawcy, nie ponawiać |
| counts o złym K lub mapie | zachować dowód, nie włączać do kompletnej analizy |
| komplet raw, błąd analizy | ponowić wyłącznie analizę offline |
| coverage_limit_reached | zachować przygotowanie, zero submit |

- [ ] Niejednoznaczne odpowiedzi dostawcy rozstrzygać po identyfikatorze próby/job tags tylko tam, gdzie dostawca je wspiera. Jeśli wymagana jest ręczna identyfikacja joba, przyjąć ją przez jawny mechanizm recovery i sprawdzić backend, K, liczbę i tożsamość obwodów przed przypięciem. Nie zakładać istnienia uniwersalnego API idempotencji.
- [ ] Testować awarię po każdym zapisie: przed submit, po zwróceniu ID, po zapisaniu raw, przed analysis. Restart nie może usuwać plików ani powtarzać zakończonych partii. Uszkodzony hash blokuje wykonanie, ale nie usuwa danych.
- [ ] Testować zawartość raw po awarii drugiej partii i po pełnym leakage. Testy używają lokalnego store oraz kontrolowanych adapterów, bez sieci.
- [ ] Uruchomić `python -m pytest -q tests/test_block_manifest.py tests/test_block_checkpointing.py tests/test_experiment_store.py tests/test_experiment_manifest.py`.

## Zadanie 7: IBM jako pełny adapter wspólnego pipeline'u

**Pliki:** utworzyć `backends/ibm.py`, `tests/test_experiment_ibm_adapter.py`; zmienić models, execution, eksporty i rejestr backendów.

- [ ] Dodać immutable `IBMHardware(device, account_name=None, instance=None)` z kind `ibm_hardware`, execution_mode HARDWARE. Używać istniejących walidatorów bezpiecznych nazw. Konfiguracja nie przechowuje tokenu. Rozwiązywanie konta delegować do `_ibm_runtime.runtime_account_options`.
- [ ] Adapter implementuje wszystkie metody `BackendAdapter`: resolve, capabilities, availability, preflight, compile, submit, result, restore_job, metadata. Wstrzykiwalne `service` i `sampler_factory` umożliwiają test bez konta.
- [ ] Kompilować przez backendowy target do ISA, zachować mapy klasycznych bitów i wyników. Preflight sprawdza backend, brak niepodstawionych parametrów, pomiary, liczby kubitów, K i limity partii. Limity pochodzą z adaptera/SDK i konfiguracji klienta, nie z zapamiętanej nazwy urządzenia.
- [ ] Raw opcje Samplera są jawne:

```python
RAW_SAMPLER_OPTIONS = {
    "dynamical_decoupling": {"enable": False},
    "twirling": {"enable_gates": False, "enable_measure": False},
}
```

- [ ] `submit` tworzy `SamplerV2(mode=backend, options=resolved_raw_options)` i wywołuje `run(circuits, shots=shots)`. Nie kopiować `ibm_runtime_options` z benchmarku, bo ta funkcja włącza mitygację. Nadpisanie shots, twirlingu lub DD przez run_options jest błędem nowego trybu. Zachować dozwolone limity czasu i parametry wykonania z ich walidacją.
- [ ] `result`: jedna pozycja Sampler PUB na jedną pozycję partii. Wspólne wyniki wielu rejestrów rekonstruować z dopasowanych shotów; nie sklejać osobnych marginalnych histogramów. Test z idealnie skorelowanymi i antyskorelowanymi bitami ma wykryć błędne łączenie rejestrów.
- [ ] `restore_job` używa `service.job(job_id)` i sprawdza zgodność z requestem. Zachować tryb pobierania po timeout bez resubmisji. Błędy sanitować tak jak w istniejących adapterach.
- [ ] Zaktualizować union Backend, `_backend_from_safe_dict`, `_FIXED_MODES`, `_IDENTITY_KINDS`, rejestr i publiczne eksporty. Dodać test bez bibliotek/konta: import głównego pakietu nie ma tworzyć klienta ani nawiązywać połączenia.
- [ ] Uruchomić `python -m pytest -q tests/test_experiment_ibm_adapter.py tests/test_experiment_provider_contracts.py tests/test_experiment_models.py tests/test_public_api.py`.

## Zadanie 8: wykonanie bloków na czterech backendach

**Pliki:** utworzyć `block_runner.py`, `tests/test_randomized_bell_runner.py`, `tests/test_randomized_bell_resume.py`; zmienić runner i adapter Aer w niezbędnym zakresie.

- [ ] Publiczny dispatch `run_experiment`: po walidacji typu spec, przed dotychczasowym przygotowaniem, przekierować `measurement is not None` do `run_randomized_experiment`. Przekazać adapter, repo_root, timeout, run_options i istniejące zegary testowe. W nowej ścieżce odrzucić stare injected evaluator/mitigation seams; nie ignorować ich.
- [ ] Nowy runner realizuje kolejno: walidacja raw -> trwałe przygotowanie -> harmonogram -> limit/pokrycie -> katalog -> adapter -> kompilacja -> plan partii -> preflight wszystkich partii -> trwałe requests -> sekwencyjne submit/result/zapis -> lokalna analiza. Przed limitem pokrycia nie uruchamia submit ani kalibracji.
- [ ] Kompilację katalogu oprzeć na istniejącym `_compile_measurement_workload`, zachowując `TranspilationConfig`, obsługiwane `workload_optimization`, layout i fizyczne mapy bitów. Jeżeli wydzielenie helpera do wspólnego modułu okaże się konieczne, przenieść jego dotychczasowe zachowanie z testami bez zmiany algorytmu. Nie ignorować konfiguracji w nowym dispatchu. PIAST nadal odrzuca lokalne opcje zarządzanej kompilacji.
- [ ] Partie tworzyć z kopii skompilowanych wzorców, z nazwą `block_{block_id:08d}`. W każdej pozycji wykonywać K shotów. `batch_index`, `position`, `block_id` i `catalog_index` zapisać jawnie.
- [ ] Efektywny limit partii = min(config.max_circuits_per_job, capabilities.max_circuits), pomijając None. Dodać test przykładu 7 bloków i limitu 3: partie [3,3,1], zachowane identyfikatory i dokładnie 7*K shotów.
- [ ] Aer obecnie odrzuca seed_simulator różny od spec. Nie obchodzić tej walidacji w starym trybie. Dla nowej ścieżki tworzyć adapter partii z `AerIdeal(seed_simulator=batch_seed)`, współdzieląc skompilowany katalog. batch_seed wyprowadzić deterministycznie z seeda symulacji i indeksu partii przez NumPy SeedSequence; zapisać. Przy utracie nieprzywracalnego lokalnego joba można jawnie odtworzyć tę partię z zapisanego seeda, zachowując historię próby; nie stosować tej reguły do sprzętu.
- [ ] IQM: użyć istniejącego compile/run i mappingów bez twirlingu; nie scalać powtarzających się pozycji. PIAST: użyć domyślnej TranspilationConfig i logicznych obwodów; nie próbować lokalnego transpile przeciw managed target. W każdym przypadku raportować compilation_owner i znane ograniczenia chronologii.
- [ ] `resume_experiment` rozpoznaje schema 4 przed starym sprawdzeniem schematów. Deleguje do `resume_randomized_experiment`; schematy 1/2/3 przechodzą przez dotychczasowy kod. Odczyt ukończonego wyniku schema 4 działa bez adaptera i bez dostępu do sieci.
- [ ] Nowy wynik używa `ExperimentResult.values` z measurement_mode, oboma B, uncertainty, leakage, coverage, budget i quality_flags. Stan częściowy schema 4 mapuje się na istniejący status wyniku dopiero na granicy API; nie dopisywać staremu walidatorowi fikcyjnego COMPLETED dla niekompletnych danych.
- [ ] Ustalić mapowanie statusów: coverage_limit_reached/partial/failed -> FAILED; submission_unknown -> SUBMISSION_UNKNOWN; komplet raw z zakończoną analizą -> COMPLETED, również przy conditional=null z pełnego leakage. Błąd analizy -> POSTPROCESSING z raw zachowanym do wznowienia. Szczegółowe stany schema 4 i przyczyny zawsze pozostają w manifest i values.
- [ ] Parametryzować testy kontraktowe Aer/IBM/IQM/PIAST: K, powtórzenia, zero wkładu, limit, raw options, preflight, save/retrieve i awaria drugiej partii. Wykonać lokalny Aer end-to-end dla wszystkich trzech scenariuszy i co najmniej dwóch kodowań.
- [ ] Uruchomić `python -m pytest -q tests/test_randomized_bell_runner.py tests/test_randomized_bell_resume.py tests/test_experiment_aer_integration.py tests/test_experiment_iqm_adapter.py tests/test_experiment_piastq_adapter.py tests/test_experiment_piastq_managed_integration.py tests/test_experiment_ibm_adapter.py`.

## Zadanie 9: przykłady i regresje

**Pliki:** utworzyć `notebooks/bell_randomized_raw.ipynb`, `tests/test_randomized_bell_notebook.py`; zmienić README.

- [ ] Notebook pokazuje wybór three scenarios, dwa kodowania, RandomizedBlocks i wszystkie cztery konfiguracje backendów. Tylko komórka lokalnego Aer może być wykonywana w teście; komórki sprzętowe mają jawny warunek uruchomienia ustawiony domyślnie na false.
- [ ] Wyświetlać N_min, N_max, N_actual, K, rzeczywisty koszt, pokrycie wzorców, liczbę bloków zerowych, raw B, conditional B lub powód niedostępności, leakage i przedziały z nazwą metody.
- [ ] Pokazać odczyt zapisanego wyniku oraz wznowienie z katalogu. Komórka porównawcza używa odrębnych harmonogramów dla ramion; nie przenosi harmonogramu pod pretekstem wspólnego seeda.
- [ ] README wyjaśnia różnicę między blokami a shotami, regułę stopu, limit bez wysyłki, znaczenie tożsamości AME, globalny leakage i ograniczenia interpretacji granicy klasycznej. Nie nazywać secrets fizycznym QRNG.
- [ ] Uruchomić `python -m pytest -q tests/test_randomized_bell_notebook.py tests/test_two_qutrit_canonical_baseline_notebook.py tests/test_ghz3_canonical_baseline_notebook.py tests/test_ame43_canonical_baseline_notebook.py tests/test_two_qutrit_vertical_slice_cli.py`.
- [ ] Na końcu uruchomić `python -m pytest -q tests`. Jeżeli środowisko nie ma opcjonalnej integracji, odróżnić przewidziany skip od błędu. Nie deklarować zweryfikowanego hardware na podstawie lokalnych atrap.
- [ ] Przejrzeć diff względem zaakceptowanego zakresu: brak zmian definicji Bella, brak przepisywania istniejących danych, brak włączonej mitygacji i brak niezamierzonych zmian równolegle rozwijanego ZNE.

## Dowody wymagane do zakończenia implementacji

| Wymaganie | Dowód |
| --- | --- |
| niezależny równomierny wybór, minimum, pokrycie i limit | zadanie 2, skryptowane sekwencje i przestrzenie z zadania 1 |
| obwód dla kompletu lokalnych ustawień | zadanie 3, oracle macierzowy i lokalny Aer |
| niezmienione nierówności, fazy, potęgi | zadania 1 i 4 |
| raw oraz conditional, wszystkie dane leakage zachowane | zadania 4, 6 i 8 |
| poprawna interpretacja losowej długości harmonogramu | korelatory i przedziały warunkowane na harmonogramie, zadania 4–5 |
| mitygacja wyłączona, cztery backendy | zadania 7–8 i testy kontraktowe |
| wyniki nie giną i jobs nie są powtarzane w ukryciu | fault injection i restart w zadaniach 6–8 |
| stary tryb i historyczne dane zachowują znaczenie | regresje modeli, schematów i notebooków |
| użytkownik może odtworzyć wynik offline | notebook i test wznowienia bez adaptera |

Po implementacji raportować osobno: wykonane testy lokalne, zweryfikowane kontrakty dostawców i ewentualne późniejsze próby sprzętowe. Nowy plan mitygacji powstaje dopiero po odbiorze tej wersji raw.
