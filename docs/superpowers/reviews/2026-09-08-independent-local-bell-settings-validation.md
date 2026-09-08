# WdroÄąÄ˝enie niezaleÄąÄ˝nych lokalnych ustawieÄąâ€ž Bella Ă˘â‚¬â€ť raw v1

Data: 2026-09-08. Specyfikacja: [projekt](../specs/2026-09-07-independent-local-bell-settings-design.md). Plan: [zadania implementacji](../plans/2026-09-08-independent-local-bell-settings.md).

## Zakres

Dodano `RandomizedBlocks`, niezaleÄąÄ˝ne rÄ‚Ĺ‚wnomierne losowanie kaÄąÄ˝dej strony przez `secrets.randbelow`, testowe ÄąĹźrÄ‚Ĺ‚dÄąâ€ša z seedem, peÄąâ€šny katalog obwodÄ‚Ĺ‚w i zatrzymanie na pierwszym pokryciu wymaganych wzorcÄ‚Ĺ‚w po osiĂ„â€¦gniĂ„â„˘ciu minimum. Obwody nie sĂ„â€¦ scalane na poziomie blokÄ‚Ĺ‚w; powtÄ‚Ĺ‚rzenia i konfiguracje bez wkÄąâ€šadu do nierÄ‚Ĺ‚wnoÄąâ€şci pozostajĂ„â€¦ w pomiarach i kosztach.

Metryki `raw` i `conditional` uÄąÄ˝ywajĂ„â€¦ dotychczasowych definicji Bella i dekodera po rotacji pomiarowej. Leakage dowolnej strony, takÄąÄ˝e pomijanej w skÄąâ€šadniku AME, zeruje caÄąâ€šy wkÄąâ€šad shota. Warunkowanie odbywa siĂ„â„˘ osobno dla kaÄąÄ˝dego korelatora. PrzedziaÄąâ€šy Hoeffdinga sĂ„â€¦ warunkowane na zapisanym harmonogramie i uÄąÄ˝ywajĂ„â€¦ liczby blokÄ‚Ĺ‚w. Mitygacja pozostaje wyÄąâ€šĂ„â€¦czona.

Nowy schema 4 zapisuje referencjĂ„â„˘, harmonogram, ÄąĹźrÄ‚Ĺ‚dÄąâ€šo stanu i kodowanie, katalogi QPY, mapy bitÄ‚Ĺ‚w, partie, prÄ‚Ĺ‚by wysyÄąâ€ški, identyfikatory zadaÄąâ€ž, zliczenia i raporty. Raporty analizy majĂ„â€¦ niezmienne generacje wskazywane przez manifest oraz wygodny alias `analysis/bell.json`. Dodatni wynik kontroli sum nie zastĂ„â„˘puje walidacji semantycznej harmonogramu, map i liczebnoÄąâ€şci.

Przy awarii analizy surowe dane pozostajĂ„â€¦ dostĂ„â„˘pne. Po niejednoznacznej wysyÄąâ€šce nie ma automatycznego ponowienia. NiepeÄąâ€šne dane zwrÄ‚Ĺ‚cone przez adapter zapisuje siĂ„â„˘ jako niepoprawny dowÄ‚Ĺ‚d pomiarowy; nie oznaczajĂ„â€¦ ukoÄąâ€žczonego eksperymentu. Publiczne API dotychczasowych trybÄ‚Ĺ‚w i formaty 1/2/3 pozostajĂ„â€¦ obsÄąâ€šugiwane.

## Weryfikacja

- Oracle macierzowy wszystkich trzech referencji, takÄąÄ˝e dla dowolnych stanÄ‚Ĺ‚w oraz kanonicznych, permutowanych i gĂ„â„˘stych izometrii E; zgodnoÄąâ€şĂ„â€ˇ do 1e-10.
- Enumeracja deterministycznych strategii lokalnych z leakage oraz zgodnoÄąâ€şĂ„â€ˇ przestrzeni 9/18/36, wzorcÄ‚Ĺ‚w 9/12/13 i konfiguracji o niezerowym wkÄąâ€šadzie 9/12/15.
- Skryptowane harmonogramy, niezaleÄąÄ˝ne wybory, powtÄ‚Ĺ‚rzenia, pierwszy moment pokrycia i limit bez wysyÄąâ€ški.
- NierÄ‚Ĺ‚wna akceptacja, peÄąâ€šne leakage, leakage widza, zliczenia z wielu rejestrÄ‚Ĺ‚w, zmienione mapy bitÄ‚Ĺ‚w, przedziaÄąâ€šy przy skorelowanych shotach i zatrzymanym harmonogramie.
- Lokalne Aer dla wszystkich trzech stanÄ‚Ĺ‚w i dwÄ‚Ĺ‚ch kodowaÄąâ€ž; ponowny odczyt bez adaptera.
- Kontrakty IBM, IQM i PIAST/AQT z lokalnymi atrapami dostawcÄ‚Ĺ‚w, w tym timeout i ponowny odbiÄ‚Ĺ‚r potwierdzonego joba, zera oraz powtÄ‚Ĺ‚rzenia.
- Awarie po zapisie znacznika prÄ‚Ĺ‚by, job ID, raw, receipt i raportu; druga prÄ‚Ĺ‚ba odczytu offline; bÄąâ€šĂ„â„˘dne K; uszkodzone hashe; niezgodne mapy; sanitacja metadanych.

ÄąĹˇrodowisko testowe: Python 3.12, Qiskit 2.1.2, Aer 0.17.2, IBM Runtime 0.44.0, zgodnie z zakresem zaleÄąÄ˝noÄąâ€şci projektu. Bez poÄąâ€šĂ„â€¦czeÄąâ€ž z backendami sprzĂ„â„˘towymi.

W peÄąâ€šnym przebiegu regresji ujawniono cztery istniejĂ„â€¦ce rozbieÄąÄ˝noÄąâ€şci notebooka GHZ i jego testÄ‚Ĺ‚w: nazwa kernela, `RUN_IQM=True`, `SHOTS=7000` i siedem instancji twirlingu. Te same cztery bÄąâ€šĂ„â„˘dy potwierdzono w niezmienionym gÄąâ€šÄ‚Ĺ‚wnym katalogu (12 pozostaÄąâ€šych testÄ‚Ĺ‚w tego pliku przeszÄąâ€šo, test hardware pominiĂ„â„˘to). Nie zmieniano tych ustawieÄąâ€ž uÄąÄ˝ytkownika.

## Granice pierwszej wersji

Wyniki sĂ„â€¦ diagnostykĂ„â€¦ na jednym procesorze, nie certyfikacjĂ„â€¦ nielokalnoÄąâ€şci. PrzedziaÄąâ€šy zakÄąâ€šadajĂ„â€¦ niezaleÄąÄ˝noÄąâ€şĂ„â€ˇ blokÄ‚Ĺ‚w przy zapisanym harmonogramie; nie obejmujĂ„â€¦ dowolnej pamiĂ„â„˘ci miĂ„â„˘dzy blokami ani zaleÄąÄ˝nych od kontekstu zmian urzĂ„â€¦dzenia.

Potwierdzone joby IBM/IQM/PIAST moÄąÄ˝na pobieraĂ„â€ˇ ponownie. RĂ„â„˘czne przypiĂ„â„˘cie nieznanego joba przez `recover_randomized_job` wymaga dowodu dostawcy o obwodach, backendzie, K i opcjach raw. IBM udostĂ„â„˘pnia takĂ„â€¦ weryfikacjĂ„â„˘; aktualne interfejsy IQM/PIAST nie udostĂ„â„˘pniajĂ„â€¦ kompletnego dowodu i odrzucajĂ„â€¦ to przypiĂ„â„˘cie. Nie zastĂ„â„˘pujemy dowodu parametrami podanymi przez klienta.

Utraconego uchwytu lokalnego joba Aer nie da siĂ„â„˘ przywrÄ‚Ĺ‚ciĂ„â€ˇ; zapisany harmonogram, seedy partii i istniejĂ„â€¦ce raw pozostajĂ„â€¦ zachowane. Jawne `replay_randomized_aer_batch` pozwala odtworzyĂ„â€ˇ lokalnĂ„â€¦ partiĂ„â„˘ z zapisanym seedem, zachowujĂ„â€¦c poprzedniĂ„â€¦ prÄ‚Ĺ‚bĂ„â„˘ i dodatkowy budÄąÄ˝et. Operacja odrzuca hardware oraz istniejĂ„â€¦ce raw. Automatyczne ponowienia sĂ„â€¦ wyÄąâ€šĂ„â€¦czone rÄ‚Ĺ‚wnieÄąÄ˝ dla Aer.

## PR validation

The PR branch is based on current origin/main and excludes unrelated local ZNE and notebook changes. Full regression on this branch: **1895 passed, 6 skipped, 4 deselected** in 523.79 s. The four exclusions are the previously reproduced GHZ notebook mismatches described above. Hardware/live execution was disabled. `git diff --cached --check` passed.
