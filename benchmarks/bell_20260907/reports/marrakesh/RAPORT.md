# Benchmark kodowań monomialnych: IQM i IBM marrakesh

> Seria IBM: **ibm_marrakesh**. Pełny ranking 216 przedstawicieli baz wykonano wcześniej na Kingston i IQM; tutaj sprawdzono te same zamrożone trzy kandydatury, bez nowego pełnego rankingu na tym procesorze. IQM to niezmienione dane pierwszego etapu. Porównania urządzeń obejmują mapowanie i czas pomiaru. Dodatkową serię Kingston bez kubitu 16 anulowano przed wykonaniem za potwierdzone 0 s.

Status: kompletny pomiar podstawowy (48 ramion).

## Protokół i zakres

Badane stany i operatory Bella pochodzą z rejestru repo: `two_qutrit`, `ghz3` (stan grafowy lokalnie równoważny GHZ trzech qutrytów) i `ame43` (AME(4,3)). Kodowanie baseline to `canonical_ez`: 0→00, 1→01, 2→10; 11 jest poza kodem.

Przeszukano pełną dyskretną klasę repo: 4 podpory × 6 permutacji × 27 kombinacji faz {1,ω,ω²} = 648 baz. Po usunięciu globalnej fazy pozostaje 216. Ta klasa nie obejmuje wszystkich ciągłych faz monomialnych. Dla każdego stanu wszystkie 216 baz oraz baseline skompilowano lokalnie na obu rzeczywistych targetach z ziarnem 907. Wynik selekcji to średnia, po dwóch dostawcach, stosunku kosztu błędu do baseline. Koszt = średnia po ustawieniach sumy −log(1−error) po bramkach i odczytach. To model selekcyjny, nie prognoza wartości Bella ani dowód globalnego optimum. Trzy wskazane bazy mają najniższy koszt w tym przeszukaniu; pomiary sprzętowe są późniejszą walidacją.

F3 zwykłe ma fazę 0 na nieużywanym stanie. Wersja zoptymalizowana używa analitycznej fazy dopełnienia (π/2 albo 11π/6, zależnie od bazy). Działanie na kodzie qutrytu jest identyczne; dokładna synteza pojedynczego F3 redukuje zwykle CZ z 3 do 2. Nie jest to zmiana kodowania na bazę Fouriera. Oba ramiona korzystają z tej samej optymalizacji ważonych krawędzi grafu (w AME43 dwie CZ zastępuje CZ²).

Każdą wybraną bazę i baseline dopracowano tą samą procedurą kompilacji z trzema ziarnami (907–909), wybierając minimalny modelowany koszt całego obwodu pomiarowego. Kolejność obwodów w jobie jest losowana deterministycznie. Wyniki i job ID pozostają w katalogach `weighted/<provider>/jobs/`.

Każdy końcowy QPY sprawdzono po ponownym odczytaniu: idealna wartość Bella, brak niepoprawnych codewordów oraz zgodność pełnego rozkładu pomiarowego z obwodem wejściowym. Naprawiono dekodowanie pomijanego uczestnika AME43 dla niekanonicznych przestrzeni kodowych; dla monomialnych podpór wymaga tylko X, bez dodatkowej CZ.

IBM: `ibm_marrakesh`, SamplerV2, DD XY4 oraz twirling bramek i pomiarów (16 randomizacji). IQM: realny Garnet. Dane do lokalnej korekcji odczytu pochodzą z dwóch rzeczywistych obwodów kalibracyjnych (wszystkie używane kubity w 0 oraz w 1) w danym jobie. Przyjęto tensorowy model niezależnych błędów odczytu; nie koryguje on dowolnych korelacji ani błędów bramek.

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
| canonical_ez | standard | 4.1539 ± 0.1132 | 3.8200 ± 0.1050 | 4.2536 ± 0.1181 | 8.29% |
| canonical_ez | optimal | 4.3156 ± 0.1143 | 3.9993 ± 0.1062 | 4.4182 ± 0.1192 | 7.55% |
| sup023_P021_ph000 | standard | 4.6230 ± 0.1174 | 3.9605 ± 0.1028 | 4.7563 ± 0.1245 | 14.37% |
| sup023_P021_ph000 | optimal | 4.5469 ± 0.1114 | 3.8992 ± 0.0983 | 4.6780 ± 0.1183 | 14.28% |
| sup023_P021_ph012 | standard | 4.5908 ± 0.1172 | 3.9129 ± 0.1029 | 4.7259 ± 0.1233 | 14.78% |
| sup023_P021_ph012 | optimal | 4.3973 ± 0.1148 | 3.7642 ± 0.1011 | 4.5223 ± 0.1207 | 14.37% |
| sup023_P021_ph020 | standard | 4.4747 ± 0.1163 | 3.8151 ± 0.1020 | 4.6023 ± 0.1225 | 14.78% |
| sup023_P021_ph020 | optimal | 4.5713 ± 0.1129 | 4.0220 ± 0.1020 | 4.7026 ± 0.1189 | 12.04% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.1617 ± 0.1604 | 0.0000 ± 0.0000 |
| sup023_P021_ph000 | -0.0761 ± 0.1588 | 0.2313 ± 0.1581 |
| sup023_P021_ph012 | -0.1934 ± 0.1668 | 0.0817 ± 0.1614 |
| sup023_P021_ph020 | 0.0966 ± 0.1607 | 0.2557 ± 0.1624 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph020**, Bell **4.5713 ± 0.1129**. Różnica względem baseline z F3 opt: **0.2557 ± 0.1624**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.0966 ± 0.1607**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

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
| canonical_ez | standard | 3.5286 ± 0.1096 | 2.9436 ± 0.0921 | 3.7578 ± 0.1182 | 17.55% |
| canonical_ez | optimal | 3.5369 ± 0.1098 | 2.9487 ± 0.0917 | 3.7627 ± 0.1196 | 17.59% |
| sup023_P021_ph001 | standard | 3.5700 ± 0.1164 | 2.8083 ± 0.0933 | 3.7933 ± 0.1282 | 21.39% |
| sup023_P021_ph001 | optimal | 3.4306 ± 0.1158 | 2.7265 ± 0.0928 | 3.6409 ± 0.1261 | 20.82% |
| sup023_P021_ph011 | standard | 3.3187 ± 0.1161 | 2.6021 ± 0.0924 | 3.5271 ± 0.1267 | 21.86% |
| sup023_P021_ph011 | optimal | 3.2633 ± 0.1172 | 2.5628 ± 0.0941 | 3.4638 ± 0.1267 | 21.73% |
| sup023_P021_ph021 | standard | 3.6072 ± 0.1138 | 2.8288 ± 0.0915 | 3.8344 ± 0.1253 | 21.58% |
| sup023_P021_ph021 | optimal | 3.2872 ± 0.1134 | 2.5906 ± 0.0910 | 3.4921 ± 0.1247 | 21.19% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.0083 ± 0.1537 | 0.0000 ± 0.0000 |
| sup023_P021_ph001 | -0.1394 ± 0.1629 | -0.1062 ± 0.1593 |
| sup023_P021_ph011 | -0.0554 ± 0.1662 | -0.2735 ± 0.1598 |
| sup023_P021_ph021 | -0.3200 ± 0.1580 | -0.2497 ± 0.1568 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph001**, Bell **3.4306 ± 0.1158**. Różnica względem baseline z F3 opt: **-0.1062 ± 0.1593**.
Dla tej bazy zmiana F3 zwykłe → opt daje **-0.1394 ± 0.1629**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

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
| canonical_ez | standard | 5.0726 ± 0.1273 | 4.0520 ± 0.1083 | 5.5334 ± 0.1469 | 19.82% |
| canonical_ez | optimal | 5.4779 ± 0.1228 | 4.3829 ± 0.1038 | 5.9864 ± 0.1434 | 19.58% |
| sup012_P210_ph002 | standard | 4.9420 ± 0.1296 | 3.7418 ± 0.1030 | 5.4240 ± 0.1532 | 24.94% |
| sup012_P210_ph002 | optimal | 4.6405 ± 0.1346 | 3.5381 ± 0.1082 | 5.0885 ± 0.1552 | 23.18% |
| sup023_P021_ph000 | standard | 5.0119 ± 0.1312 | 3.7762 ± 0.1049 | 5.5121 ± 0.1525 | 24.53% |
| sup023_P021_ph000 | optimal | 5.1795 ± 0.1302 | 3.9960 ± 0.1052 | 5.6929 ± 0.1524 | 23.48% |
| sup023_P021_ph011 | standard | 5.0070 ± 0.1308 | 3.8030 ± 0.1038 | 5.4965 ± 0.1521 | 24.80% |
| sup023_P021_ph011 | optimal | 4.9795 ± 0.1369 | 3.7554 ± 0.1081 | 5.4813 ± 0.1603 | 24.28% |

| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |
|---|---:|---:|
| canonical_ez | 0.4053 ± 0.1787 | 0.0000 ± 0.0000 |
| sup012_P210_ph002 | -0.3015 ± 0.1886 | -0.8375 ± 0.1831 |
| sup023_P021_ph000 | 0.1676 ± 0.1845 | -0.2984 ± 0.1825 |
| sup023_P021_ph011 | -0.0274 ± 0.1862 | -0.4984 ± 0.1826 |

Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **sup023_P021_ph000**, Bell **5.1795 ± 0.1302**. Różnica względem baseline z F3 opt: **-0.2984 ± 0.1825**.
Dla tej bazy zmiana F3 zwykłe → opt daje **0.1676 ± 0.1845**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.

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
| ibm | daf32fe42tqs73avchn0 | cancelled | 0.000 | rozliczony |
| ibm | daf2hvbdd5gc73d98h30 | cancelled | 0.000 | rozliczony |
| ibm | daf5435nj4cs73aftmr0 | completed | 46.000 | rozliczony |
| ibm | daf3fl5nj4cs73afqnj0 | cancelled | 0.000 | rozliczony |
| ibm | daf2ibe42tqs73avc2cg | cancelled | 0.000 | rozliczony |
| ibm | daf545l1ierc738mth90 | completed | 48.000 | rozliczony |
| ibm | daf3fp5nj4cs73afqnq0 | cancelled | 0.000 | rozliczony |
| ibm | daf4mqdnj4cs73aft2v0 | completed | 46.000 | rozliczony |

Limit tej kampanii: 540 s IBM, 1000 s IQM (500 kredytów, stawka podana przez użytkownika: 1 kredyt / 2 s). Wpis bez potwierdzonego zużycia zachowuje pełną rezerwę. Czas oczekiwania w kolejce nie jest czasem QPU.

IBM ma twardy limit czasu każdego zadania. IQM nie udostępnia analogicznego limitu: dalsze rezerwacje wyznacza zmierzony koszt pilota lub górne ograniczenie z timeline, sumy kalibrowanych czasów bramek, mnożnik bezpieczeństwa 2 oraz 30 s zapasu. Górne ograniczenie IQM obejmuje cały przedział od przyjęcia przez station control do ukończenia przez server, włącznie z kompilacją i przetwarzaniem wyników; nie jest fakturą. Batch IQM ma rezerwę najwyżej 240 s. Osobny proces nadzoruje czas przetwarzania i zleca anulowanie 30 s przed końcem rezerwy. To kontrola klienta, nie gwarancja dostawcy; zależy od łączności i czasu anulowania.

Dokumentacja dostawców: [IBM: limit czasu](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time), [IBM: zużycie](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time), [IQM Resonance](https://iqm.tech/products/iqm-resonance/).

[Raport końcowy wszystkich procesorów](../RAPORT_KONCOWY.md). Wyniki ZNE wymagają oceny krzywizny opisanej w raporcie końcowym; nie traktować ich automatycznie jako potwierdzonej poprawy fizycznej.
