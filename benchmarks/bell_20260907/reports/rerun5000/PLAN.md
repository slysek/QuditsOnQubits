# Powtórzenie Fez + IQM: 5000 shotów

Zakres zaakceptowany przez użytkownika: wszystkie 3 stany, baseline i 3 wcześniej wybrane bazy na stan, F3 standard/optimal, oba rzeczywiste procesory. Dokładnie 5000 shotów na każde ustawienie oraz obwód kalibracyjny. Użytkownik zgodził się pominąć ZNE, jeżeli pełne ZNE przy 5000 shotów nie mieści się w budżecie. Nowa seria obejmuje wyłącznie skalę 1.

- [x] Sprawdzić saldo IBM i dostępność Fez: 243 s na koncie; pierwotny limit kampanii pozostawia 229 s. Fez operational.
- [x] Zachować pierwotne limity całej kampanii: IBM 540 s, IQM 1000 s = 500 kredytów. Dotychczas IBM 311 s, IQM górne ograniczenie 218.937842 s.
- [x] Utrzymać bazy, obwody i definicje Bella z poprzedniej serii. Sprawdzić hashe zweryfikowanych QPY przed wykonaniem.
- [x] Przygotować exact 5000 dla IBM: 8 × 625 randomizacji, DD XY4, reset włączony, rep_delay 75 µs. Rezerwa i twardy limit zadania 220 s. Poprzednio 250 µs i 16 randomizacji; różnicę konfiguracji trzeba uwzględnić w interpretacji.
- [x] Sprawdzić podział IQM: 24 zadania, po jednym kompletnym ramieniu i dwóch kalibracjach. Rezerwy 172/211/224 s zależnie od stanu, zadania sekwencyjne. Nie sumować wszystkich hipotetycznych rezerw; przed każdym rzeczywistym zadaniem sprawdzić globalny rejestr i zatrzymać przy braku miejsca.
- [x] Zakończyć niezależny przegląd nowego wykonawcy i testy przed wysyłką.
- [x] Wysłać i odebrać Fez oraz 24 zadania IQM. Zachować potwierdzenia, job ID, surowe counts i zużycie.
- [x] Sprawdzić 48 kompletnych ramion, 272 ustawienia × 5000 shotów na dostawcę. Osobno kalibracje: IBM 2, IQM 48. Łącznie 2 970 000 shotów.
- [x] Analiza: cztery estymatory, SE i 95% CI, F3 opt − standard i kandydaci − baseline, wartości bezwzględne i procentowe. Bez łączenia starych counts z nową serią. Osobne porównanie z poprzednią serią Fez/IQM.
- [x] Raport po polsku, CSV/JSON, wykres i końcowy audyt zużycia oraz kompletności.

Estymacja z wcześniejszych jobów przy niezmienionych ustawieniach IBM: podstawowa seria około 415 s; trzy skale około 1333 s. Skrócenie rep_delay do 75 µs obniża ekstrapolację podstawowej serii do około 175 s; to estymacja, nie gwarantowany czas dostawcy. Limit 220 s pozostaje aktywny. Krótsza przerwa może zmienić jakość resetu. IQM zachowuje dotychczasowe ustawienia urządzenia.

Źródła: [IBM Sampler options](https://quantum.cloud.ibm.com/docs/en/guides/sampler-options), [IBM qubit initialization](https://quantum.cloud.ibm.com/docs/en/guides/repetition-rate-execution), lokalne potwierdzenia poprzednich jobów i aktualny odczyt API.

Wysłano Fez: `daf8ptm42tqs73avkj50`. IQM runner wykonuje `raw5000-iqm-00`…`23` sekwencyjnie. Wszystkie receipts pozostają w globalnym `weighted/<provider>/jobs/`; analizy nowej serii wyłącznie w `weighted/rerun5000/`. Review round1: CLEAN. Testy nowego zakresu13passed.

Ukończono: 25/25jobów,48/48ramion,544ustawienia po5000shotów. Nowa seria IBM127s,IQM526.660201s(górne); globalnie IBM438/540s,IQM745.598043/1000s.62testypassed. Końcowy raport: RAPORT.md.
