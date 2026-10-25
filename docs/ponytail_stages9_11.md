# Ponytail: etapy 9–11

Data: 2026-10-25. Zakres: pliki przypisane do etapów 9–11 w
[spisie](ponytail_file_inventory.csv). AME46 pozostaje poza zakresem.
Zachowano istniejące zmiany użytkownika. Nie zmieniano danych pomiarowych.

## Etap 9 — API, przykłady, notebooki i narzędzia

W `vertical_slice/runner.py` poprawiono metadane pochodzenia wyników.
Puste, poprawne wyjście `git status --porcelain` oznacza czyste repo.
Wcześniej helper zamieniał je na `None`, a manifest zapisywał nieznany stan.
Teraz zapisuje `dirty_worktree=False`. Błąd polecenia nadal oznacza `None`.
Pusty identyfikator commitu nadal nie jest przyjmowany.

Regresja przed zmianą: 1 błąd i 2 zaliczone przypadki. Te same przypadki
po zmianie sprawdzają repo czyste, zmienione oraz niedostępny odczyt statusu.
Dodano trzy przypadki awarii kompilacji, zlecenia i odbioru wyniku.
Sprawdzają zapis dowodów błędu, zachowanie potwierdzonego identyfikatora
zadania oraz brak powtórnego zlecenia.

| Pliki / grupa | Decyzja i zakres sprawdzenia |
| --- | --- |
| `vertical_slice/{models,bell,cli,runner,__init__}.py` | Zachować kontrakty, walidację, niemutowalne migawki i hashe. Poprawić wyłącznie rozpoznanie czystego repo. Testy rzeczywistego Aer, zapisu/odczytu, błędów, dekodowania i modeli. |
| Główne `__init__.py` | Zachować leniwe eksporty i dotychczasowe pierwszeństwo nazw `ExecutionMode` i `RunManifest` z `experiments`. Zmiana tego pierwszeństwa zmieniłaby publiczne API. |
| Dwa przykłady Python i `theta_demo.zip` | Zachować. Loader sprawdza SHA-256 przed rozpakowaniem, ogranicza ścieżki i usuwa katalog tymczasowy po błędzie. Dane zawierają wszystkie 41 punktów. Przykłady i notebook mają testy rzeczywistego wykonania lokalnego. |
| `scripts/iqm_bell_short.py`, `iqm_bell_zne_models.py`, `iqm_randomized_bell_campaign.py` | Zachować osobne protokoły kampanii. Kontrole strzałów, bloków, kalibracji, korelacji i jawnej zgody sprzętowej mają znaczenie naukowe. Współdzielenie całych protokołów nie dałoby małej, bezpiecznej zmiany. Testy obejmują modele, zapisane wyniki i granice wykonania. |
| `scripts/load_best_circuit.py` | Zachować wybór raw/transpiled, jednoznaczny ranking i obowiązkowe `E.npy`. Test sprawdza oba rodzaje obwodów i brak kodowania. |
| `tools/agent_relay/` i fixture dostawcy | Zachować. Rozdzielenie protokołu, stanu i procesów jest uzasadnione. Testy używają lokalnych atrap CLI, sprawdzają limity, blokadę, checkpointy, zatrzymanie i błędy. Nie wywołano modeli. |
| `notebooks/working/iqm/provider.py` | Zachować rzeczywiste kalibracje i jawny wybór symulacji. Testy sprawdzają konwersję fidelities, T1/T2, pochodzenie profilu i brak ukrytego zastępowania niedostępnego QPU. |
| `scripts/verify_two_qutrit_clean_room.ps1` | Zachować skrypt Windows. Przeczytano budowanie, instalację, tolerancję wyniku i sprzątanie. Nie wykonano PowerShell. Równoważny kontrakt paczki sprawdza linuksowy verifier. |
| 25 notebooków | Zachować przykłady publiczne i historyczne materiały badawcze. Wszystkie 401 komórek kodu mają poprawną składnię po transformacji IPython. Testy notebooków sprawdzają wykonywalne scenariusze lokalne, konfigurację, brak sekretów i wyłączone domyślnie kampanie w obsługiwanych przykładach. Starsze notebooki badawcze sprawdzono strukturalnie; nie wykonano ich pełnych kampanii. |
| Testy etapów 9–11 | Zachować niezależne wyniki Bella, odtwarzanie archiwów, rzeczywisty Aer i ochronę danych. Kontrole treści notebooków zachowują zabezpieczenia przed przypadkowym QPU i ujawnieniem sekretów. Nie usuwano ich tylko dlatego, że używają AST lub tekstu. |

Zestaw `vertical_slice` z regresjami pakowania: 75 zaliczonych, 10 zastanych
ostrzeżeń IBM. Pokrycie instrukcji: 86% całego `vertical_slice`, 81%
`runner.py`. Nie jest to pokrycie gałęzi ani całego repo.

## Etap 10 — zapisane dane i odtwarzanie

Sprawdzono wszystkie 460 plików przypisanych do tego etapu. Każdy SHA-256
jest zgodny ze spisem początkowym. Odczytano 117 plików QPY, 5 NPY,
260 JSON i 28 CSV. Archiwum ZIP przeszło kontrolę CRC. Pozostałe pliki
to kod, dokumentacja, obrazy i znaczniki katalogów. Nie zmieniano żadnego
z tych danych ani zapisanych raportów.

| Pliki / grupa | Decyzja i zakres sprawdzenia |
| --- | --- |
| `benchmarks/bell_20260907/reproduce.py` i testy | Zachować wspierany punkt wejścia. Sprawdza pełną listę i hashe ZIP przed rozpakowaniem, strzały, mianowniki, metadane postselekcji i porównania publikacji. Odmawia nadpisania istniejącego katalogu. |
| Historyczne `study`, `hardware`, `target_screen`, `execute`, `rerun5000`, `analyze`, `report` | Zachować kod źródłowy kampanii i osobne formaty historyczne. Kontrole budżetu, niepewnego zlecenia, kalibracji i odtwarzania są potrzebne. Dokumentacja już odróżnia te skrypty od gotowego launchera nowej kampanii. Nie uruchamiano ich na QPU. |
| `manifest.json`, `data.zip`, `reports/` | Zachować niezmienne dowody publikacji, również wyniki negatywne. Szybkie odtworzenie: 43 zakończone zadania, 240 wierszy i zgodne wartości opublikowane. |
| Trzy zapisane kampanie IQM w `artifacts/` | Zachować oryginalne counts, kalibracje, QPY i statystyki. Testy przeliczają wartości, bloki i modele oraz sprawdzają odciski zleconych obwodów. |
| `experiment_inputs/`, oba katalogi QPY i znaczniki katalogów wyników | Zachować używane zasoby i wejścia z pochodzeniem. Odczyt QPY/NPY i hashe przeszły. Brak użycia konkretnego historycznego pliku nie uzasadnia usunięcia dowodu badawczego. |

Pełne odtworzenie `reproduce.py --bootstrap` również przeszło: wszystkie
43 zakończone zadania i 240 wierszy, po 2000 replik bootstrapu.
Skrypt przeliczył rozkłady pomiarów i kalibracji oraz porównał wartości,
błędy standardowe i końce przedziałów z publikacją przy tolerancji `1e-9`.
Wynik zapisuje `bootstrap_recomputed: true`. Nie wykonano żadnego nowego
pomiaru sprzętowego.

## Etap 11 — konfiguracja i paczka

Zachowano zakresy zależności i podział pakietów. Wszystkie 20 podstawowych
wymagań w `requirements.txt` odpowiada `pyproject.toml`. `uv pip check`
potwierdził zgodność 174 zainstalowanych pakietów po dodaniu wymaganych już
narzędzi budowania `setuptools` i `wheel` oraz instalacji projektu.

W sdist wykryto brak modułów `tools/agent_relay/`, mimo obecności ich testu.
Zbieranie testów z rozpakowanego archiwum kończyło się
`ModuleNotFoundError: No module named 'tools'`. `MANIFEST.in` obejmuje teraz
te moduły i dokumentację. Verifier wymaga ich oraz fixture testowego.
Dwa nowe przypadki sprawdzają odrzucenie niepełnej paczki.
Manifest obejmuje też CSV i TXT dokumentacji planu.
Zastane `prune notebooks/private` pozostało bez zmian.

| Pliki / grupa | Decyzja i zakres sprawdzenia |
| --- | --- |
| `pyproject.toml`, `requirements.txt` | Zachować zakresy Python 3.11–3.13 i obecne zależności. Sprawdzono zgodność deklaracji i środowiska. Nie dodano zależności projektu. |
| `MANIFEST.in`, `scripts/verify_distribution.py`, test verifiera | Uzupełnić brakujące narzędzia, fixture i dokumentację planu. Zachować runtime QPY i dane theta w wheel. |
| Oba workflow CI | Zachować macierz systemów i wersji, testy, odtwarzanie i budowanie ze sdist. Lokalne sprawdzenie nie zastępuje wykonania tej macierzy. |
| `.env.example`, `.gitignore`, `.gitattributes` | Zachować granice poświadczeń, lokalnych wyników i niezmiennych danych. Nie odczytywano plików kont i sekretów. |
| README, przewodniki i specyfikacja | Zachować rozróżnienie kompilacji, symulacji i pomiaru oraz osobne schematy artefaktów. Sprawdzono lokalne odnośniki 27 dokumentów, bez brakujących celów. |
| Licencja, cytowanie, changelog, polityki i szablony GitHub | Zachować informacje prawne i reguły współpracy. Testy metadanych sprawdzają zgodność wersji, dat i odnośników wydania. |
| `log.txt` | Zachować zastany log użytkownika. Nie dołączać go do dystrybucji. |

Budowanie w roboczym repo przerwano podczas skanowania manifestu. Repo
zawiera setki tysięcy ignorowanych wyników i lokalne środowiska. Budowanie
z kopii wszystkich plików objętych spisem, bez ignorowanych wyników,
zakończyło się poprawnie. Jest to kontrola czystego drzewa źródeł,
nie pomiar czasu budowania dużego roboczego katalogu.

## Dowody i ograniczenia

Logi i wyniki maszynowe znajdują się w ignorowanym katalogu
`artifacts/ponytail/completion-20261025/`. Końcowy wynik wspólnego zestawu,
bootstrapu i instalacji jest zapisany w
[planie](ponytail_refactor_plan.md#zamknięcie-planowanej-pracy).

Pierwszy zbiorczy zestaw etapów 9–11 uruchomiono przez stdin. Dwa testy
procesów `spawn` nie mogły wtedy zaimportować `<stdin>`: 491 zaliczonych,
2 pominięte, 2 błędy sposobu uruchomienia. Końcowy zestaw używa
`python -m pytest`, co usuwa tę przyczynę. Nie zmieniano tych testów.

Nie sprawdzono QPU, bieżących usług dostawców, prywatnego PiastQ,
rzeczywistej kosztownej syntezy CZ3 ani macierzy Windows/Python 3.11/3.13.
Kontrola składni notebooków nie dowodzi poprawności ich pełnych kampanii.
Nie dodano nowego frameworka, API, cache ani podziału zależności.
