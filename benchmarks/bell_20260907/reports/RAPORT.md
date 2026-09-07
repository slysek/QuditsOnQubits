# Benchmark kodowań monomialnych na IQM i IBM

Status: kompletny pomiar podstawowy (48 ramion).

## Protokół i zakres

Badane stany i operatory Bella pochodzą z rejestru repo: `two_qutrit`, `ghz3` (stan grafowy lokalnie równoważny GHZ trzech qutrytów) i `ame43` (AME(4,3)). Kodowanie baseline to `canonical_ez`: 0→00, 1→01, 2→10; 11 jest poza kodem.

Przeszukano pełną dyskretną klasę repo: 4 podpory × 6 permutacji × 27 kombinacji faz {1,ω,ω²} = 648 baz. Po usunięciu globalnej fazy pozostaje 216. Ta klasa nie obejmuje wszystkich ciągłych faz monomialnych. Dla każdego stanu wszystkie 216 baz oraz baseline skompilowano lokalnie na obu rzeczywistych targetach z ziarnem 907. Wynik selekcji to średnia, po dwóch dostawcach, stosunku kosztu błędu do baseline. Koszt = średnia po ustawieniach sumy −log(1−error) po bramkach i odczytach. To model selekcyjny, nie prognoza wartości Bella ani dowód globalnego optimum. Trzy wskazane bazy mają najniższy koszt w tym przeszukaniu; pomiary sprzętowe są późniejszą walidacją.

F3 zwykłe ma fazę 0 na nieużywanym stanie. Wersja zoptymalizowana używa analitycznej fazy dopełnienia (π/2 albo 11π/6, zależnie od bazy). Działanie na kodzie qutrytu jest identyczne; dokładna synteza pojedynczego F3 redukuje zwykle CZ z 3 do 2. Nie jest to zmiana kodowania na bazę Fouriera. Oba ramiona korzystają z tej samej optymalizacji ważonych krawędzi grafu (w AME43 dwie CZ zastępuje CZ²).

Każdą wybraną bazę i baseline dopracowano tą samą procedurą kompilacji z trzema ziarnami (907–909), wybierając minimalny modelowany koszt całego obwodu pomiarowego. Kolejność obwodów w jobie jest losowana deterministycznie. Wyniki i job ID pozostają w katalogach `weighted/<provider>/jobs/`.

Każdy końcowy QPY sprawdzono po ponownym odczytaniu: idealna wartość Bella, brak niepoprawnych codewordów oraz zgodność pełnego rozkładu pomiarowego z obwodem wejściowym. Naprawiono dekodowanie pomijanego uczestnika AME43 dla niekanonicznych przestrzeni kodowych; dla monomialnych podpór wymaga tylko X, bez dodatkowej CZ.

IBM: `ibm_kingston`, SamplerV2, DD XY4 oraz twirling bramek i pomiarów (16 randomizacji). IQM: realny Garnet. Dane do lokalnej korekcji odczytu pochodzą z dwóch rzeczywistych obwodów kalibracyjnych (wszystkie używane kubity w 0 oraz w 1) w danym jobie. Przyjęto tensorowy model niezależnych błędów odczytu; nie koryguje on dowolnych korelacji ani błędów bramek.

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
| canonical_ez | standard | 4.1334 ± 0.0804 | 3.8421 ± 0.0751 | 4.8018 ± 0.1377 | 7.80% |
| canonical_ez | optimal | 4.1128 ± 0.0794 | 3.8169 ± 0.0739 | 4.9573 ± 0.1418 | 7.92% |
| sup023_P021_ph000 | standard | 4.9899 ± 0.0792 | 4.6614 ± 0.0752 | 5.1512 ± 0.0846 | 6.60% |
| sup023_P021_ph000 | optimal | 5.0177 ± 0.0766 | 4.6714 ± 0.0722 | 5.1807 ± 0.0817 | 6.90% |
| sup023_P021_ph012 | standard | 5.2204 ± 0.0778 | 4.8651 ± 0.0738 | 5.3920 ± 0.0826 | 6.81% |
| sup023_P021_ph012 | optimal | 5.1141 ± 0.0764 | 4.7675 ± 0.0729 | 5.2807 ± 0.0817 | 6.79% |
| sup023_P021_ph020 | standard | 5.1357 ± 0.0781 | 4.7744 ± 0.0737 | 5.3026 ± 0.0832 | 7.05% |
| sup023_P021_ph020 | optimal | 5.0911 ± 0.0788 | 4.7467 ± 0.0743 | 5.2556 ± 0.0836 | 6.76% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | -0.0206 ± 0.1130 | 0.0000 ± 0.0000 |
| sup023_P021_ph000 | 0.0277 ± 0.1075 | 0.9049 ± 0.1138 |
| sup023_P021_ph012 | -0.1063 ± 0.1072 | 1.0013 ± 0.1074 |
| sup023_P021_ph020 | -0.0446 ± 0.1117 | 0.9783 ± 0.1106 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph012**, Bell **5.1141 ± 0.0764**. Różnica względem baseline z F3 opt: **1.0013 ± 0.1074**.
Dla tej bazy zmiana F3 zwykłe → opt daje **-0.1063 ± 0.1072**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

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
| canonical_ez | standard | 4.5239 ± 0.0718 | 4.0473 ± 0.0658 | 4.7821 ± 0.0798 | 10.61% |
| canonical_ez | optimal | 4.8032 ± 0.0723 | 4.3305 ± 0.0664 | 5.0805 ± 0.0808 | 10.14% |
| sup023_P021_ph001 | standard | 4.6107 ± 0.0729 | 4.0930 ± 0.0660 | 4.8785 ± 0.0818 | 11.28% |
| sup023_P021_ph001 | optimal | 4.7284 ± 0.0737 | 4.2142 ± 0.0671 | 5.0016 ± 0.0825 | 11.14% |
| sup023_P021_ph011 | standard | 4.5539 ± 0.0731 | 4.0542 ± 0.0662 | 4.8157 ± 0.0809 | 11.02% |
| sup023_P021_ph011 | optimal | 4.5669 ± 0.0733 | 4.0421 ± 0.0663 | 4.8298 ± 0.0819 | 11.52% |
| sup023_P021_ph021 | standard | 4.6955 ± 0.0721 | 4.1702 ± 0.0661 | 4.9684 ± 0.0812 | 11.28% |
| sup023_P021_ph021 | optimal | 4.6477 ± 0.0763 | 4.1264 ± 0.0693 | 4.9148 ± 0.0844 | 11.20% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.2793 ± 0.1031 | 0.0000 ± 0.0000 |
| sup023_P021_ph001 | 0.1178 ± 0.1044 | -0.0748 ± 0.1043 |
| sup023_P021_ph011 | 0.0130 ± 0.1032 | -0.2363 ± 0.1044 |
| sup023_P021_ph021 | -0.0478 ± 0.1038 | -0.1555 ± 0.1030 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph001**, Bell **4.7284 ± 0.0737**. Różnica względem baseline z F3 opt: **-0.0748 ± 0.1043**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.1178 ± 0.1044**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

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
| canonical_ez | standard | 4.6429 ± 0.0935 | 3.7103 ± 0.0771 | 6.9945 ± 0.2599 | 19.97% |
| canonical_ez | optimal | 4.7365 ± 0.0913 | 3.8132 ± 0.0760 | 7.2661 ± 0.2683 | 19.73% |
| sup012_P210_ph002 | standard | 4.1567 ± 0.1131 | 2.3366 ± 0.0668 | 8.1779 ± 0.4126 | 43.56% |
| sup012_P210_ph002 | optimal | 4.0396 ± 0.1114 | 2.3131 ± 0.0672 | 7.6025 ± 0.4004 | 42.98% |
| sup023_P021_ph000 | standard | 4.0527 ± 0.1140 | 2.3445 ± 0.0691 | 7.8470 ± 0.4053 | 41.87% |
| sup023_P021_ph000 | optimal | 3.8345 ± 0.1110 | 2.1945 ± 0.0664 | 7.4577 ± 0.4052 | 43.12% |
| sup023_P021_ph011 | standard | 3.9682 ± 0.1099 | 2.2928 ± 0.0673 | 7.6193 ± 0.3906 | 42.41% |
| sup023_P021_ph011 | optimal | 3.7786 ± 0.1114 | 2.1730 ± 0.0673 | 7.2053 ± 0.3929 | 42.82% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.0936 ± 0.1289 | 0.0000 ± 0.0000 |
| sup012_P210_ph002 | -0.1172 ± 0.1579 | -0.6969 ± 0.1449 |
| sup023_P021_ph000 | -0.2182 ± 0.1579 | -0.9020 ± 0.1427 |
| sup023_P021_ph011 | -0.1897 ± 0.1578 | -0.9579 ± 0.1466 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup012_P210_ph002**, Bell **4.0396 ± 0.1114**. Różnica względem baseline z F3 opt: **-0.6969 ± 0.1449**.
Dla tej bazy zmiana F3 zwykłe → opt daje **-0.1172 ± 0.1579**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

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
| ibm | two_qutrit | canonical_ez | 261 | 243 | 18 |
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
| ibm | daf4uq3dd5gc73d9c730 | completed | 85.000 | rozliczony |
| ibm | daf4mqdnj4cs73aft2v0 | completed | 46.000 | rozliczony |
| ibm | daf32fe42tqs73avchn0 | cancelled | 0.000 | rozliczony |
| ibm | daf2hvbdd5gc73d98h30 | cancelled | 0.000 | rozliczony |
| ibm | daf5435nj4cs73aftmr0 | completed | 46.000 | rozliczony |
| ibm | daf3fl5nj4cs73afqnj0 | cancelled | 0.000 | rozliczony |
| ibm | daf2ibe42tqs73avc2cg | cancelled | 0.000 | rozliczony |
| ibm | daf545l1ierc738mth90 | completed | 48.000 | rozliczony |
| ibm | daf3fp5nj4cs73afqnq0 | cancelled | 0.000 | rozliczony |

Limit tej kampanii: 540 s IBM, 1000 s IQM (500 kredytów, stawka podana przez użytkownika: 1 kredyt / 2 s). Wpis bez potwierdzonego zużycia zachowuje pełną rezerwę. Czas oczekiwania w kolejce nie jest czasem QPU.

IBM ma twardy limit czasu każdego zadania. IQM nie udostępnia analogicznego limitu: dalsze rezerwacje wyznacza zmierzony koszt pilota lub górne ograniczenie z timeline, sumy kalibrowanych czasów bramek, mnożnik bezpieczeństwa 2 oraz 30 s zapasu. Górne ograniczenie IQM obejmuje cały przedział od przyjęcia przez station control do ukończenia przez server, włącznie z kompilacją i przetwarzaniem wyników; nie jest fakturą. Batch IQM ma rezerwę najwyżej 240 s. Osobny proces nadzoruje czas przetwarzania i zleca anulowanie 30 s przed końcem rezerwy. To kontrola klienta, nie gwarancja dostawcy; zależy od łączności i czasu anulowania.

Dokumentacja dostawców: [IBM: limit czasu](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time), [IBM: zużycie](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time), [IQM Resonance](https://iqm.tech/products/iqm-resonance/).

[Raport końcowy wszystkich procesorów](RAPORT_KONCOWY.md). Powyższe tabele IBM dotyczą pierwszej serii Kingston; wspólny budżet obejmuje także późniejsze serie. [Analiza repo](UWAGI_I_REPO.md), [diagnostyka ZNE IQM](ZNE_DIAGNOSTYKA.md).
