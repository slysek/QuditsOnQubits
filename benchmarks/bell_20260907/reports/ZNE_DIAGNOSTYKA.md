# Kontrola modelu ZNE

Kontrola dodatkowa: C = Bell(1) - 2 Bell(3) + Bell(5). Model liniowy przewiduje C=0. Przedział C pochodzi z tego samego bootstrapu pomiarów i kalibracji. Flaga oznacza, że 95% przedział nie obejmuje zera; to diagnostyka eksploracyjna bez korekty wielokrotnych porównań. Brak flagi nie dowodzi poprawności modelu.

Wyniku ekstrapolacji przekraczającego wartość idealną nie traktujemy automatycznie jako potwierdzonej poprawy. Szczególnie mocno nieliniowe krzywe IQM pokazują, że sama niepewność shotów nie opisuje błędu modelu. Pełne wartości, również niespójne z liniowością, pozostają w tabelach.

| Dostawca | Stan | Baza | F3 | C (odczyt + postselekcja) | SE | Flaga modelu |
|---|---|---|---|---:|---:|---|
| iqm | ame43 | canonical_ez | optimal | 2.7860 | 0.3010 | TAK |
| iqm | ame43 | canonical_ez | standard | 3.1066 | 0.2987 | TAK |
| iqm | ame43 | sup012_P210_ph002 | optimal | 3.4673 | 0.4288 | TAK |
| iqm | ame43 | sup012_P210_ph002 | standard | 4.1041 | 0.4312 | TAK |
| iqm | ame43 | sup023_P021_ph000 | optimal | 3.6313 | 0.5131 | TAK |
| iqm | ame43 | sup023_P021_ph000 | standard | 4.8281 | 0.4812 | TAK |
| iqm | ame43 | sup023_P021_ph011 | optimal | 4.2033 | 0.5217 | TAK |
| iqm | ame43 | sup023_P021_ph011 | standard | 4.5532 | 0.4961 | TAK |
| iqm | ghz3 | canonical_ez | optimal | 1.4674 | 0.3198 | TAK |
| iqm | ghz3 | canonical_ez | standard | 2.0039 | 0.3171 | TAK |
| iqm | ghz3 | sup023_P021_ph001 | optimal | 2.0558 | 0.3369 | TAK |
| iqm | ghz3 | sup023_P021_ph001 | standard | 1.3255 | 0.3400 | TAK |
| iqm | ghz3 | sup023_P021_ph011 | optimal | 1.7146 | 0.3578 | TAK |
| iqm | ghz3 | sup023_P021_ph011 | standard | 1.8846 | 0.3460 | TAK |
| iqm | ghz3 | sup023_P021_ph021 | optimal | 1.0126 | 0.3340 | TAK |
| iqm | ghz3 | sup023_P021_ph021 | standard | 1.0189 | 0.3455 | TAK |
| iqm | two_qutrit | canonical_ez | optimal | 0.6455 | 0.2973 | TAK |
| iqm | two_qutrit | canonical_ez | standard | 0.9841 | 0.2923 | TAK |
| iqm | two_qutrit | sup023_P021_ph000 | optimal | -1.0828 | 0.3269 | TAK |
| iqm | two_qutrit | sup023_P021_ph000 | standard | -1.8768 | 0.3203 | TAK |
| iqm | two_qutrit | sup023_P021_ph012 | optimal | -1.8597 | 0.3090 | TAK |
| iqm | two_qutrit | sup023_P021_ph012 | standard | -2.1271 | 0.3213 | TAK |
| iqm | two_qutrit | sup023_P021_ph020 | optimal | -2.3666 | 0.3159 | TAK |
| iqm | two_qutrit | sup023_P021_ph020 | standard | -1.2422 | 0.3292 | TAK |
