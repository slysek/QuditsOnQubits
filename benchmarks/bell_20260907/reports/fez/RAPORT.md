# Benchmark kodowań monomialnych: IQM i IBM fez

> Seria IBM: **ibm_fez**. Pełny ranking 216 przedstawicieli baz wykonano wcześniej na Kingston i IQM; tutaj sprawdzono te same zamrożone trzy kandydatury, bez nowego pełnego rankingu na tym procesorze. IQM to niezmienione dane pierwszego etapu. Porównania urządzeń obejmują mapowanie i czas pomiaru. Dodatkową serię Kingston bez kubitu 16 anulowano przed wykonaniem za potwierdzone 0 s.

Status: kompletny pomiar podstawowy (48 ramion).

## Protokół i zakres

Badane stany i operatory Bella pochodzą z rejestru repo: `two_qutrit`, `ghz3` (stan grafowy lokalnie równoważny GHZ trzech qutrytów) i `ame43` (AME(4,3)). Kodowanie baseline to `canonical_ez`: 0→00, 1→01, 2→10; 11 jest poza kodem.

Przeszukano pełną dyskretną klasę repo: 4 podpory × 6 permutacji × 27 kombinacji faz {1,ω,ω²} = 648 baz. Po usunięciu globalnej fazy pozostaje 216. Ta klasa nie obejmuje wszystkich ciągłych faz monomialnych. Dla każdego stanu wszystkie 216 baz oraz baseline skompilowano lokalnie na obu rzeczywistych targetach z ziarnem 907. Wynik selekcji to średnia, po dwóch dostawcach, stosunku kosztu błędu do baseline. Koszt = średnia po ustawieniach sumy −log(1−error) po bramkach i odczytach. To model selekcyjny, nie prognoza wartości Bella ani dowód globalnego optimum. Trzy wskazane bazy mają najniższy koszt w tym przeszukaniu; pomiary sprzętowe są późniejszą walidacją.

F3 zwykłe ma fazę 0 na nieużywanym stanie. Wersja zoptymalizowana używa analitycznej fazy dopełnienia (π/2 albo 11π/6, zależnie od bazy). Działanie na kodzie qutrytu jest identyczne; dokładna synteza pojedynczego F3 redukuje zwykle CZ z 3 do 2. Nie jest to zmiana kodowania na bazę Fouriera. Oba ramiona korzystają z tej samej optymalizacji ważonych krawędzi grafu (w AME43 dwie CZ zastępuje CZ²).

Każdą wybraną bazę i baseline dopracowano tą samą procedurą kompilacji z trzema ziarnami (907–909), wybierając minimalny modelowany koszt całego obwodu pomiarowego. Kolejność obwodów w jobie jest losowana deterministycznie. Wyniki i job ID pozostają w katalogach `weighted/<provider>/jobs/`.

Każdy końcowy QPY sprawdzono po ponownym odczytaniu: idealna wartość Bella, brak niepoprawnych codewordów oraz zgodność pełnego rozkładu pomiarowego z obwodem wejściowym. Naprawiono dekodowanie pomijanego uczestnika AME43 dla niekanonicznych przestrzeni kodowych; dla monomialnych podpór wymaga tylko X, bez dodatkowej CZ.

IBM: `ibm_fez`, SamplerV2, DD XY4 oraz twirling bramek i pomiarów (16 randomizacji). IQM: realny Garnet. Dane do lokalnej korekcji odczytu pochodzą z dwóch rzeczywistych obwodów kalibracyjnych (wszystkie używane kubity w 0 oraz w 1) w danym jobie. Przyjęto tensorowy model niezależnych błędów odczytu; nie koryguje on dowolnych korelacji ani błędów bramek.

## Wyniki sprzętowe

± to 1 bootstrap SE. Tabele CSV/JSON zawierają 95% przedziały. Bootstrap obejmuje statystykę pomiarów oraz resampling kalibracji odczytu, wspólny dla ramion z tego samego joba. Nie obejmuje dryfu, błędu modelu odczytu ani modelu ZNE. Porównania i wybór największego zmierzonego wyniku są eksploracyjne; przedziały nie mają korekty wielokrotnych porównań.

Bell po postselekcji oznacza odrzucenie shotów z co najmniej jednym wynikiem poza kodem. Bell bez postselekcji przypisuje takim shotom wkład zero zgodnie z repo. Wynik po postselekcji lub mitygacji nie jest bezlukowym dowodem nielokalności; kubity uczestników są na tym samym procesorze.

### two_qutrit

Ideał: 6; nominalna granica klasyczna: 5.6381557.

| Baza | Logiczne 0,1,2: indeksy fizyczne | Faza F3 opt / π |
|---|---|---:|
| canonical_ez | (0, 1, 2) | 1.833333 |
| sup023_P021_ph000 | (0, 3, 2) | 0.500000 |
| sup023_P021_ph012 | (0, 3, 2) | 0.500000 |
| sup023_P021_ph020 | (0, 3, 2) | 0.500000 |

**IQM**

| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |
|---|---|---:|---:|---:|---:|
| canonical_ez | standard | 4.7302 ± 0.0782 | 4.4631 ± 0.0744 | 5.0148 ± 0.0877 | 5.66% |
| canonical_ez | optimal | 4.9010 ± 0.0771 | 4.6364 ± 0.0742 | 5.1974 ± 0.0859 | 5.41% |
| sup023_P021_ph000 | standard | 4.6612 ± 0.0785 | 4.2532 ± 0.0736 | 4.9447 ± 0.0878 | 8.73% |
| sup023_P021_ph000 | optimal | 4.6224 ± 0.0791 | 4.2360 ± 0.0740 | 4.9055 ± 0.0882 | 8.36% |
| sup023_P021_ph012 | standard | 4.7681 ± 0.0789 | 4.3395 ± 0.0730 | 5.0509 ± 0.0893 | 9.03% |
| sup023_P021_ph012 | optimal | 4.8504 ± 0.0767 | 4.4303 ± 0.0718 | 5.1463 ± 0.0860 | 8.66% |
| sup023_P021_ph020 | standard | 4.7401 ± 0.0824 | 4.2824 ± 0.0756 | 5.0286 ± 0.0927 | 9.61% |
| sup023_P021_ph020 | optimal | 4.7664 ± 0.0789 | 4.3766 ± 0.0741 | 5.0597 ± 0.0897 | 8.20% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.1707 ± 0.1090 | 0.0000 ± 0.0000 |
| sup023_P021_ph000 | -0.0388 ± 0.1108 | -0.2785 ± 0.1104 |
| sup023_P021_ph012 | 0.0823 ± 0.1079 | -0.0506 ± 0.1092 |
| sup023_P021_ph020 | 0.0263 ± 0.1111 | -0.1346 ± 0.1098 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph012**, Bell **4.8504 ± 0.0767**. Różnica względem baseline z F3 opt: **-0.0506 ± 0.1092**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.0823 ± 0.1079**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

**IBM**

| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |
|---|---|---:|---:|---:|---:|
| canonical_ez | standard | 4.6401 ± 0.0801 | 4.2667 ± 0.0746 | 4.9299 ± 0.0902 | 8.07% |
| canonical_ez | optimal | 4.5030 ± 0.0779 | 4.1554 ± 0.0735 | 4.7948 ± 0.0870 | 7.76% |
| sup023_P021_ph000 | standard | 4.7319 ± 0.0788 | 4.2812 ± 0.0726 | 5.0342 ± 0.0886 | 9.51% |
| sup023_P021_ph000 | optimal | 4.8832 ± 0.0807 | 4.4012 ± 0.0741 | 5.2003 ± 0.0917 | 9.86% |
| sup023_P021_ph012 | standard | 4.8655 ± 0.0809 | 4.4061 ± 0.0750 | 5.1873 ± 0.0913 | 9.44% |
| sup023_P021_ph012 | optimal | 4.7293 ± 0.0797 | 4.2804 ± 0.0740 | 5.0388 ± 0.0900 | 9.48% |
| sup023_P021_ph020 | standard | 4.7319 ± 0.0773 | 4.2668 ± 0.0712 | 5.0430 ± 0.0883 | 9.84% |
| sup023_P021_ph020 | optimal | 4.7682 ± 0.0777 | 4.3298 ± 0.0724 | 5.0719 ± 0.0879 | 9.24% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | -0.1371 ± 0.1107 | 0.0000 ± 0.0000 |
| sup023_P021_ph000 | 0.1513 ± 0.1127 | 0.3802 ± 0.1108 |
| sup023_P021_ph012 | -0.1362 ± 0.1147 | 0.2263 ± 0.1113 |
| sup023_P021_ph020 | 0.0362 ± 0.1096 | 0.2651 ± 0.1112 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph000**, Bell **4.8832 ± 0.0807**. Różnica względem baseline z F3 opt: **0.3802 ± 0.1108**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.1513 ± 0.1127**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

### ghz3

Ideał: 6; nominalna granica klasyczna: 5.6381557.

| Baza | Logiczne 0,1,2: indeksy fizyczne | Faza F3 opt / π |
|---|---|---:|
| canonical_ez | (0, 1, 2) | 1.833333 |
| sup023_P021_ph001 | (0, 3, 2) | 0.500000 |
| sup023_P021_ph011 | (0, 3, 2) | 0.500000 |
| sup023_P021_ph021 | (0, 3, 2) | 0.500000 |

**IQM**

| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |
|---|---|---:|---:|---:|---:|
| canonical_ez | standard | 3.8663 ± 0.1064 | 3.5262 ± 0.0984 | 4.2663 ± 0.1228 | 8.76% |
| canonical_ez | optimal | 3.9144 ± 0.1032 | 3.5470 ± 0.0948 | 4.3687 ± 0.1230 | 9.68% |
| sup023_P021_ph001 | standard | 3.8601 ± 0.1118 | 3.1856 ± 0.0942 | 4.4142 ± 0.1398 | 17.85% |
| sup023_P021_ph001 | optimal | 3.7979 ± 0.1111 | 3.1998 ± 0.0953 | 4.3525 ± 0.1398 | 16.10% |
| sup023_P021_ph011 | standard | 3.7817 ± 0.1087 | 3.1227 ± 0.0926 | 4.3483 ± 0.1368 | 17.17% |
| sup023_P021_ph011 | optimal | 3.7888 ± 0.1125 | 3.0416 ± 0.0939 | 4.3442 ± 0.1415 | 19.32% |
| sup023_P021_ph021 | standard | 3.4915 ± 0.1108 | 2.9428 ± 0.0952 | 4.0121 ± 0.1347 | 15.71% |
| sup023_P021_ph021 | optimal | 3.7588 ± 0.1119 | 3.1436 ± 0.0960 | 4.3053 ± 0.1383 | 16.72% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.0481 ± 0.1447 | 0.0000 ± 0.0000 |
| sup023_P021_ph001 | -0.0622 ± 0.1572 | -0.1166 ± 0.1477 |
| sup023_P021_ph011 | 0.0071 ± 0.1548 | -0.1256 ± 0.1493 |
| sup023_P021_ph021 | 0.2674 ± 0.1577 | -0.1556 ± 0.1540 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph001**, Bell **3.7979 ± 0.1111**. Różnica względem baseline z F3 opt: **-0.1166 ± 0.1477**.
Dla tej bazy zmiana F3 zwykłe → opt daje **-0.0622 ± 0.1572**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

**IBM**

| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |
|---|---|---:|---:|---:|---:|
| canonical_ez | standard | 4.0946 ± 0.0746 | 3.5424 ± 0.0658 | 4.5096 ± 0.0888 | 13.48% |
| canonical_ez | optimal | 4.0945 ± 0.0750 | 3.5792 ± 0.0663 | 4.5084 ± 0.0904 | 12.69% |
| sup023_P021_ph001 | standard | 3.9785 ± 0.0773 | 3.3990 ± 0.0672 | 4.3892 ± 0.0920 | 14.59% |
| sup023_P021_ph001 | optimal | 4.0436 ± 0.0768 | 3.4612 ± 0.0675 | 4.4565 ± 0.0900 | 14.52% |
| sup023_P021_ph011 | standard | 3.9353 ± 0.0772 | 3.3836 ± 0.0676 | 4.3379 ± 0.0906 | 13.98% |
| sup023_P021_ph011 | optimal | 3.9558 ± 0.0793 | 3.4167 ± 0.0696 | 4.3640 ± 0.0939 | 13.50% |
| sup023_P021_ph021 | standard | 3.9835 ± 0.0768 | 3.3599 ± 0.0666 | 4.3958 ± 0.0910 | 15.54% |
| sup023_P021_ph021 | optimal | 3.9882 ± 0.0777 | 3.3997 ± 0.0678 | 4.3965 ± 0.0929 | 14.96% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | -0.0001 ± 0.1049 | 0.0000 ± 0.0000 |
| sup023_P021_ph001 | 0.0651 ± 0.1097 | -0.0509 ± 0.1080 |
| sup023_P021_ph011 | 0.0206 ± 0.1100 | -0.1387 ± 0.1089 |
| sup023_P021_ph021 | 0.0047 ± 0.1090 | -0.1063 ± 0.1085 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph001**, Bell **4.0436 ± 0.0768**. Różnica względem baseline z F3 opt: **-0.0509 ± 0.1080**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.0651 ± 0.1097**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

### ame43

Ideał: 8; nominalna granica klasyczna: 7.6381557.

| Baza | Logiczne 0,1,2: indeksy fizyczne | Faza F3 opt / π |
|---|---|---:|
| canonical_ez | (0, 1, 2) | 1.833333 |
| sup012_P210_ph002 | (2, 1, 0) | 0.500000 |
| sup023_P021_ph000 | (0, 3, 2) | 0.500000 |
| sup023_P021_ph011 | (0, 3, 2) | 0.500000 |

**IQM**

| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |
|---|---|---:|---:|---:|---:|
| canonical_ez | standard | 4.0068 ± 0.0942 | 3.2630 ± 0.0784 | 4.4804 ± 0.1092 | 19.96% |
| canonical_ez | optimal | 4.1773 ± 0.0936 | 3.4209 ± 0.0792 | 4.6677 ± 0.1096 | 19.08% |
| sup012_P210_ph002 | standard | 3.5635 ± 0.1340 | 2.7732 ± 0.1076 | 4.0254 ± 0.1569 | 23.39% |
| sup012_P210_ph002 | optimal | 3.2455 ± 0.1329 | 2.5450 ± 0.1057 | 3.6708 ± 0.1552 | 22.88% |
| sup023_P021_ph000 | standard | 3.4471 ± 0.1490 | 2.3558 ± 0.1041 | 3.9673 ± 0.1790 | 31.22% |
| sup023_P021_ph000 | optimal | 3.8488 ± 0.1477 | 2.5650 ± 0.1027 | 4.4577 ± 0.1813 | 32.53% |
| sup023_P021_ph011 | standard | 3.9506 ± 0.1449 | 2.6899 ± 0.1031 | 4.5255 ± 0.1733 | 31.27% |
| sup023_P021_ph011 | optimal | 3.7974 ± 0.1486 | 2.5899 ± 0.1046 | 4.3827 ± 0.1773 | 32.03% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.1705 ± 0.1319 | 0.0000 ± 0.0000 |
| sup012_P210_ph002 | -0.3179 ± 0.1889 | -0.9318 ± 0.1608 |
| sup023_P021_ph000 | 0.4016 ± 0.2094 | -0.3286 ± 0.1747 |
| sup023_P021_ph011 | -0.1533 ± 0.2042 | -0.3800 ± 0.1755 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph000**, Bell **3.8488 ± 0.1477**. Różnica względem baseline z F3 opt: **-0.3286 ± 0.1747**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.4016 ± 0.2094**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

**IBM**

| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |
|---|---|---:|---:|---:|---:|
| canonical_ez | standard | 4.8673 ± 0.0954 | 3.7624 ± 0.0772 | 5.2429 ± 0.1092 | 22.61% |
| canonical_ez | optimal | 5.0417 ± 0.0912 | 3.9687 ± 0.0748 | 5.4293 ± 0.1026 | 21.46% |
| sup012_P210_ph002 | standard | 4.6251 ± 0.0972 | 3.4234 ± 0.0760 | 5.0662 ± 0.1122 | 26.31% |
| sup012_P210_ph002 | optimal | 4.7809 ± 0.0955 | 3.5143 ± 0.0744 | 5.2241 ± 0.1104 | 26.39% |
| sup023_P021_ph000 | standard | 4.4013 ± 0.0970 | 3.2763 ± 0.0746 | 4.8144 ± 0.1114 | 26.18% |
| sup023_P021_ph000 | optimal | 4.7398 ± 0.0961 | 3.4709 ± 0.0740 | 5.1823 ± 0.1116 | 26.94% |
| sup023_P021_ph011 | standard | 4.5330 ± 0.0945 | 3.3275 ± 0.0733 | 4.9466 ± 0.1093 | 27.15% |
| sup023_P021_ph011 | optimal | 4.5628 ± 0.0975 | 3.3397 ± 0.0742 | 4.9958 ± 0.1142 | 26.80% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.1744 ± 0.1336 | 0.0000 ± 0.0000 |
| sup012_P210_ph002 | 0.1559 ± 0.1382 | -0.2607 ± 0.1341 |
| sup023_P021_ph000 | 0.3385 ± 0.1398 | -0.3018 ± 0.1342 |
| sup023_P021_ph011 | 0.0298 ± 0.1349 | -0.4789 ± 0.1340 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup012_P210_ph002**, Bell **4.7809 ± 0.0955**. Różnica względem baseline z F3 opt: **-0.2607 ± 0.1341**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.1559 ± 0.1382**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

## ZNE (analiza dodatkowa)

Liniowa ekstrapolacja ze skal CZ 1,3,5, z ustalonymi wagami 13/12, 1/3, −5/12. Wszystkie wyniki, również pogorszenia i niepewne ekstrapolacje, zapisano w `zne.csv`. Nie wybierano modelu pod najwyższy wynik. Bootstrap nie obejmuje błędu modelu ZNE.

| Dostawca | Stan | Baza | F3 | ZNE + odczyt + postselekcja |
|---|---|---|---|---:|
| iqm | ame43 | canonical_ez | optimal | 5.3755 ± 0.1376 |
| iqm | ame43 | canonical_ez | standard | 5.0956 ± 0.1380 |
| iqm | ame43 | sup012_P210_ph002 | optimal | 4.0077 ± 0.1933 |
| iqm | ame43 | sup012_P210_ph002 | standard | 4.2532 ± 0.1973 |
| iqm | ame43 | sup023_P021_ph000 | optimal | 5.0616 ± 0.2272 |
| iqm | ame43 | sup023_P021_ph000 | standard | 4.1132 ± 0.2228 |
| iqm | ame43 | sup023_P021_ph011 | optimal | 4.7339 ± 0.2286 |
| iqm | ame43 | sup023_P021_ph011 | standard | 4.8427 ± 0.2206 |
| iqm | ghz3 | canonical_ez | optimal | 4.9439 ± 0.1512 |
| iqm | ghz3 | canonical_ez | standard | 4.7547 ± 0.1510 |
| iqm | ghz3 | sup023_P021_ph001 | optimal | 4.7703 ± 0.1683 |
| iqm | ghz3 | sup023_P021_ph001 | standard | 5.0672 ± 0.1669 |
| iqm | ghz3 | sup023_P021_ph011 | optimal | 4.9686 ± 0.1721 |
| iqm | ghz3 | sup023_P021_ph011 | standard | 4.8743 ± 0.1667 |
| iqm | ghz3 | sup023_P021_ph021 | optimal | 4.9289 ± 0.1666 |
| iqm | ghz3 | sup023_P021_ph021 | standard | 4.6789 ± 0.1627 |
| iqm | two_qutrit | canonical_ez | optimal | 5.9362 ± 0.1155 |
| iqm | two_qutrit | canonical_ez | standard | 5.7794 ± 0.1145 |
| iqm | two_qutrit | sup023_P021_ph000 | optimal | 6.0798 ± 0.1229 |
| iqm | two_qutrit | sup023_P021_ph000 | standard | 6.2969 ± 0.1257 |
| iqm | two_qutrit | sup023_P021_ph012 | optimal | 6.4966 ± 0.1187 |
| iqm | two_qutrit | sup023_P021_ph012 | standard | 6.4473 ± 0.1264 |
| iqm | two_qutrit | sup023_P021_ph020 | optimal | 6.5197 ± 0.1221 |
| iqm | two_qutrit | sup023_P021_ph020 | standard | 6.2390 ± 0.1294 |
| ibm | ame43 | canonical_ez | optimal | 6.1638 ± 0.1442 |
| ibm | ame43 | canonical_ez | standard | 5.9760 ± 0.1507 |
| ibm | ame43 | sup012_P210_ph002 | optimal | 5.9852 ± 0.1566 |
| ibm | ame43 | sup012_P210_ph002 | standard | 5.9886 ± 0.1527 |
| ibm | ame43 | sup023_P021_ph000 | optimal | 5.9189 ± 0.1575 |
| ibm | ame43 | sup023_P021_ph000 | standard | 5.4425 ± 0.1555 |
| ibm | ame43 | sup023_P021_ph011 | optimal | 5.6407 ± 0.1584 |
| ibm | ame43 | sup023_P021_ph011 | standard | 5.8008 ± 0.1559 |
| ibm | ghz3 | canonical_ez | optimal | 5.1857 ± 0.1235 |
| ibm | ghz3 | canonical_ez | standard | 5.3034 ± 0.1203 |
| ibm | ghz3 | sup023_P021_ph001 | optimal | 5.0443 ± 0.1261 |
| ibm | ghz3 | sup023_P021_ph001 | standard | 4.9896 ± 0.1250 |
| ibm | ghz3 | sup023_P021_ph011 | optimal | 4.8807 ± 0.1269 |
| ibm | ghz3 | sup023_P021_ph011 | standard | 4.8783 ± 0.1219 |
| ibm | ghz3 | sup023_P021_ph021 | optimal | 5.0224 ± 0.1245 |
| ibm | ghz3 | sup023_P021_ph021 | standard | 4.9819 ± 0.1238 |
| ibm | two_qutrit | canonical_ez | optimal | 5.3903 ± 0.1158 |
| ibm | two_qutrit | canonical_ez | standard | 5.6104 ± 0.1218 |
| ibm | two_qutrit | sup023_P021_ph000 | optimal | 5.8127 ± 0.1220 |
| ibm | two_qutrit | sup023_P021_ph000 | standard | 5.5676 ± 0.1179 |
| ibm | two_qutrit | sup023_P021_ph012 | optimal | 5.4470 ± 0.1190 |
| ibm | two_qutrit | sup023_P021_ph012 | standard | 5.7834 ± 0.1208 |
| ibm | two_qutrit | sup023_P021_ph020 | optimal | 5.5877 ± 0.1203 |
| ibm | two_qutrit | sup023_P021_ph020 | standard | 5.5884 ± 0.1166 |

## Koszt końcowych obwodów

Liczby CZ dotyczą całego zestawu ustawień Bella (9/12/13 obwodów), przed dodaniem DD/twirlingu przez IBM i przed ewentualnym foldingiem ZNE.

| Dostawca | Stan | Baza | CZ zwykłe F3 | CZ opt F3 | Redukcja |
|---|---|---|---:|---:|---:|
| iqm | two_qutrit | canonical_ez | 207 | 189 | 18 |
| iqm | two_qutrit | sup023_P021_ph000 | 219 | 210 | 9 |
| iqm | two_qutrit | sup023_P021_ph012 | 219 | 210 | 9 |
| iqm | two_qutrit | sup023_P021_ph020 | 219 | 210 | 9 |
| iqm | ghz3 | canonical_ez | 537 | 483 | 54 |
| iqm | ghz3 | sup023_P021_ph001 | 546 | 522 | 24 |
| iqm | ghz3 | sup023_P021_ph011 | 546 | 522 | 24 |
| iqm | ghz3 | sup023_P021_ph021 | 546 | 522 | 24 |
| iqm | ame43 | canonical_ez | 1020 | 968 | 52 |
| iqm | ame43 | sup012_P210_ph002 | 980 | 954 | 26 |
| iqm | ame43 | sup023_P021_ph000 | 980 | 954 | 26 |
| iqm | ame43 | sup023_P021_ph011 | 980 | 954 | 26 |
| ibm | two_qutrit | canonical_ez | 252 | 234 | 18 |
| ibm | two_qutrit | sup023_P021_ph000 | 219 | 210 | 9 |
| ibm | two_qutrit | sup023_P021_ph012 | 219 | 210 | 9 |
| ibm | two_qutrit | sup023_P021_ph020 | 219 | 210 | 9 |
| ibm | ghz3 | canonical_ez | 567 | 531 | 36 |
| ibm | ghz3 | sup023_P021_ph001 | 534 | 510 | 24 |
| ibm | ghz3 | sup023_P021_ph011 | 534 | 510 | 24 |
| ibm | ghz3 | sup023_P021_ph021 | 534 | 510 | 24 |
| ibm | ame43 | canonical_ez | 1191 | 1139 | 52 |
| ibm | ame43 | sup012_P210_ph002 | 1158 | 1132 | 26 |
| ibm | ame43 | sup023_P021_ph000 | 1158 | 1132 | 26 |
| ibm | ame43 | sup023_P021_ph011 | 1158 | 1132 | 26 |

## Budżet i dowody wykonania

| Dostawca | Job | Status | Czas [s] | Rodzaj |
|---|---|---|---:|---|
| iqm | 01a079b5-5e3f-7c50-b6d2-95004f33ab10 | completed | 3.248 | górne ograniczenie QPU z timeline |
| iqm | 01a079e7-4288-7050-a634-4e2821729702 | failed | 0.214 | górne ograniczenie QPU z timeline |
| iqm | 01a079ef-c1d3-7f91-b55a-8ddfbdf4495f | completed | 10.835 | górne ograniczenie QPU z timeline |
| iqm | 01a079f0-15d9-7321-a80e-bc50068fb54a | completed | 15.042 | górne ograniczenie QPU z timeline |
| iqm | 01a079e6-a749-76c2-9642-b3d066fc23e8 | completed | 45.530 | górne ograniczenie QPU z timeline |
| iqm | 01a079e3-ec18-75f0-be0a-16f32165daef | completed | 24.757 | górne ograniczenie QPU z timeline |
| iqm | 01a079f7-edf3-7992-ae26-e1c092188989 | completed | 11.144 | górne ograniczenie QPU z timeline |
| iqm | 01a079f8-3391-7e22-aea1-76c90ab4b5ea | completed | 15.685 | górne ograniczenie QPU z timeline |
| iqm | 01a079f3-c28a-7b20-b873-4870e423d292 | completed | 18.988 | górne ograniczenie QPU z timeline |
| iqm | 01a079ec-d0d5-77a1-ac6c-7c235370ea1a | completed | 13.063 | górne ograniczenie QPU z timeline |
| iqm | 01a07a03-cee1-7471-9117-7b171ab49880 | completed | 11.743 | górne ograniczenie QPU z timeline |
| iqm | 01a07a04-1a1d-7422-ac18-94998d7031cf | completed | 16.436 | górne ograniczenie QPU z timeline |
| iqm | 01a079fd-ae76-7ae2-bbaf-b9d05582c93e | completed | 18.942 | górne ograniczenie QPU z timeline |
| iqm | 01a079f8-6eb5-7f83-a3d8-1a58c2199cac | completed | 13.312 | górne ograniczenie QPU z timeline |
| ibm | daf2e5e42tqs73avbu3g | completed | 86.000 | rozliczony |
| ibm | daf4mqdnj4cs73aft2v0 | completed | 46.000 | rozliczony |
| ibm | daf32fe42tqs73avchn0 | cancelled | 0.000 | rozliczony |
| ibm | daf2hvbdd5gc73d98h30 | cancelled | 0.000 | rozliczony |
| ibm | daf3fl5nj4cs73afqnj0 | cancelled | 0.000 | rozliczony |
| ibm | daf2ibe42tqs73avc2cg | cancelled | 0.000 | rozliczony |
| ibm | daf3fp5nj4cs73afqnq0 | cancelled | 0.000 | rozliczony |
| ibm | daf4uq3dd5gc73d9c730 | completed | 85.000 | rozliczony |
| ibm | daf5435nj4cs73aftmr0 | completed | 46.000 | rozliczony |
| ibm | daf545l1ierc738mth90 | completed | 48.000 | rozliczony |

Limit tej kampanii: 540 s IBM, 1000 s IQM (500 kredytów, stawka podana przez użytkownika: 1 kredyt / 2 s). Wpis bez potwierdzonego zużycia zachowuje pełną rezerwę. Czas oczekiwania w kolejce nie jest czasem QPU.

IBM ma twardy limit czasu każdego zadania. IQM nie udostępnia analogicznego limitu: dalsze rezerwacje wyznacza zmierzony koszt pilota lub górne ograniczenie z timeline, sumy kalibrowanych czasów bramek, mnożnik bezpieczeństwa 2 oraz 30 s zapasu. Górne ograniczenie IQM obejmuje cały przedział od przyjęcia przez station control do ukończenia przez server, włącznie z kompilacją i przetwarzaniem wyników; nie jest fakturą. Batch IQM ma rezerwę najwyżej 240 s. Osobny proces nadzoruje czas przetwarzania i zleca anulowanie 30 s przed końcem rezerwy. To kontrola klienta, nie gwarancja dostawcy; zależy od łączności i czasu anulowania.

Dokumentacja dostawców: [IBM: limit czasu](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time), [IBM: zużycie](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time), [IQM Resonance](https://iqm.tech/products/iqm-resonance/).

[Raport końcowy wszystkich procesorów](../RAPORT_KONCOWY.md). Wyniki ZNE wymagają oceny krzywizny opisanej w raporcie końcowym; nie traktować ich automatycznie jako potwierdzonej poprawy fizycznej.
