# Worklog: Bell qutrit graph states measurements

## 2026-05-12

### Etap 1: inicjalizacja

- Utworzono dokument `docs/bell_qutrit_graph_states_measurements.tex`.
- Utworzono worklog `docs/bell_qutrit_graph_states_measurements_WORKLOG.md`.
- Jako zrodlo wybrano lokalny PDF `pdfs/scalableameoperators.pdf`.

### Etap 2: pierwsze wypelnienie

- Wpisano podstawowe sekcje o operatorach qutrytowych, stanach grafowych, AME(4,3)
  i GHZ.
- Po pozniejszej kontroli stwierdzono, ze wersja byla niepelna:
  - Bell expression dla dwoch qutrytow byl zbyt ogolny.
  - Przyblizenie `6 cos(pi/9)` bylo bledne.
  - Brakowalo jawnej reguly zamiany `(XZ^k)^n -> A_tilde_k^(n)` dla `n=1,2`.
  - Brakowalo Bell expression dla AME(4,3) z PDF-a.
  - GHZ mial niejasny i niespojny opis wyboru `N_1`.
  - Dokument zawieral puste sekcje o Qiskit/notebookach, ktore nie naleza do tego etapu.

### Etap 3: kontrola i poprawki

- Ponownie sprawdzono wybrane fragmenty PDF-a, ograniczajac sie do zakresu promptu:
  Bell scenario, operatory `X`, `Z`, graph states, stabilizatory, regula zamiany dla `d=3`,
  dwa qutryty, AME(4,3), ogolna konstrukcja dla graph states oraz GHZ.
- Przepisano `docs/bell_qutrit_graph_states_measurements.tex` jako dokument skupiony
  tylko na wymaganych fragmentach.
- Dodano scenariusz Bella w notacji prawdopodobienstw i zespolonych wartosci oczekiwanych.
- Dodano qutrytowe operatory:
  `X|j> = |j+1 mod 3>`, `Z|j> = omega^j |j>`, `omega = exp(2*pi*i/3)`.
- Dodano stabilizatory stanow grafowych `G_i = X_i prod_j Z_j^{r_ij}`.
- Dodano regule zamiany dla `d=3`, `k=0,1,2`, `n=1,2`, wraz z
  `lambda_1 = exp(pi*i/18)` i `lambda_2 = lambda_1^*`.
- Dla dwoch qutrytow wpisano Bell expression z PDF-a:
  `I_max^(3) = sum_{n=1}^2 sum_{k=0}^2 <A_tilde_k^(n) B_k^n>`.
- Poprawiono boundy dla dwoch qutrytow:
  `beta_Q = 6`, `beta_L = 6 cos(pi/9) ~= 5.63816`.
- Dla AME(4,3) wpisano stabilizatory, Bell expression z PDF-a oraz boundy:
  `beta_Q = 8`, `beta_L = 7.63816`.
- Dla GHZ wybrano standardowy graf-gwiazde z centralnym wierzcholkiem jako `1`,
  co dla `d=3`, `N=3`, `N_1=2` daje `beta_Q = 6`.
- W ogolnej konstrukcji dopisano jawny warunek normalizacji na wspolczynniki
  `c_{1,k}` i `c_{2,k}`.
- Zapisano jako niejasnosc, ze PDF nie podaje jawnie klasycznego boundu dla tego
  konkretnego 3-qutrytowego GHZ.
- Nie uruchamiano notebookow.

### Status

- Etap z promptu zakonczony na dokumencie LaTeX i worklogu.
- Zgodnie z poleceniem praca zostaje przerwana po aktualizacji worklogu.

### Etap 4: implementacja i audyt notebookow

- Statycznie przeanalizowano komorki notebookow:
  `notebooks/ame43_random_chsh.ipynb`,
  `notebooks/2qutrit_RyGates.ipynb`,
  `notebooks/ghz_ry_chsh_recovered.ipynb`.
- Nie uruchamiano pelnych notebookow.
- Uruchomiono tylko maly fragment obliczajacy `lambda_1` i `lambda_2` dla `d=3`;
  wynik zgadza sie z `exp(pi*i/18)` oraz `exp(-pi*i/18)`.
- Do dokumentu dopisano model implementacyjny kodowania qutrytu przez izometrie
  `E: C^3 -> C^4`, wraz z warunkiem `E^dagger E = I_3`.
- Dopisano blokowanie qubitow:
  qutryt 0 = `[0,1]`, qutryt 1 = `[2,3]`, qutryt 2 = `[4,5]`,
  qutryt 3 = `[6,7]`.
- Podkreslono, ze nie wolno zakladac `E = E_Z W`; ogolne `E` moze miec support
  na `|11>`.
- Dodano transformacje operatorow:
  `O_phys = E O E^dagger`,
  `O_full = E O E^dagger + I_perp`,
  `I_perp = I_4 - E E^dagger`.
- Dodano transformacje projektorow pomiarowych `P_a_phys = E P_a E^dagger`.
- Dodano opis leakage dla Estimatora i Samplera, w tym
  `p_leak = 1 - <psi|P_E^tensor N|psi>`.
- Dodano zasade Estimatora: operatory niehermitowskie rozbijac na
  `Re(O)=(O+O^dagger)/2` i `Im(O)=(O-O^dagger)/(2i)` albo budowac
  hermitowski operator Bella jako `term + term^dagger`.
- Dodano zasade Samplera: mierzyc w bazach wlasnych operatorow pomiarowych,
  mapowac bloki dwubitowe na wyniki qutrytowe, wykrywac leakage i liczyc
  `<prod_i A_i^{n_i}> = sum_a omega^{a dot n} p(a|x)`.
- Dodano audyt kazdego notebooka:
  - co probuje liczyc,
  - zwiazek z PDF-em,
  - traktowanie operatorow, `n=1`, `n=2`, c.c./dagger, `lambda_n`, `omega`,
  - uzycie Estimatora i rozdzial na czesci hermitowskie,
  - mapowanie qubitow na qutryty,
  - status rotacji `Ry01/Ry02/Ry12` jako pobocznej optymalizacji.
- Dodano pseudokod funkcji implementacyjnych:
  `make_XZ_qutrit`, `make_A_tilde_qutrit_d3`, `embed_operator_E`,
  `embed_projector_E`, `split_nonhermitian`,
  `build_bell_operator_two_qutrit`, `build_bell_operator_ghz_graph`,
  `build_bell_operator_ame43`, `bell_value_estimator`,
  `bell_value_sampler`, `leakage_probability`.
- Dodano koncowa checkliste walidacji implementacji.

### Status po etapie 4

- Dokument zawiera czesc teoretyczna, czesc implementacyjna, audyt notebookow,
  pseudokod oraz checkliste walidacji.
- Praca zakonczona na aktualizacji dokumentu i worklogu.
