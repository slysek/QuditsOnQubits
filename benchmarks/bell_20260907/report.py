"""Generate Polish report, CSV tables and scientific figures from hardware evidence."""
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import numpy as np
from study import ROOT,STATES,VARIANTS,encodings,save_json
from execute import budget_charge

KEYS=("raw_unconditional","raw_conditional","readout_unconditional","readout_conditional")


def load_results():
    groups=defaultdict(list)
    for provider in ("iqm","ibm"):
        for path in (ROOT/provider/"jobs").glob("*/analysis.json"):
            rows=json.loads(path.read_text())
            with np.load(path.parent/"bootstrap.npz") as boot:
                for row in rows:
                    if not row["pilot"]:
                        groups[(provider,row["state"],row["name"],row["variant"],row["factor"])].append((row,boot[row["bootstrap_key"]].copy()))
    result={}
    for key,items in groups.items():
        weights=np.array([r["total_shots"] for r,b in items],float)
        weights/=weights.sum()
        draws=sum(w*b for w,(r,b) in zip(weights,items))
        row={"jobs":[r["job_id"] for r,b in items],"total_shots":sum(r["total_shots"] for r,b in items),
             "invalid_fraction":float(sum(w*r["invalid_fraction"] for w,(r,b) in zip(weights,items)))}
        for i,name in enumerate(KEYS):
            row[name]={"value":float(sum(w*r[name]["value"] for w,(r,b) in zip(weights,items))),
                       "se":float(np.std(draws[:,i],ddof=1)),"ci95":np.quantile(draws[:,i],[.025,.975]).tolist()}
        result[key]=(row,draws)
    return result


def difference(left,right,index):
    l,lb=left
    r,rb=right
    value=l[KEYS[index]]["value"]-r[KEYS[index]]["value"]
    draws=lb[:,index]-rb[:,index]
    return {"value":value,"se":float(np.std(draws,ddof=1)),"ci95":np.quantile(draws,[.025,.975]).tolist()}


def fmt(estimate):
    return f'{estimate["value"]:.4f} ± {estimate["se"]:.4f}'


def write_csv(path,rows):
    if rows:
        with path.open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def report():
    selection=json.loads((ROOT/"selection.json").read_text())
    bounds={r["state"]:r for r in json.loads((ROOT.parent/"bounds.json").read_text())}
    data=load_results()
    expected={(p,s,n,v,1) for p in ("iqm","ibm") for s in STATES for n in ["canonical_ez"]+selection[s]["selected"] for v in VARIANTS}
    missing=sorted(expected-set(data))
    flat=[]
    deltas=[]
    lines=["# Benchmark kodowań monomialnych na IQM i IBM", "",
           "Status: "+("kompletny pomiar podstawowy (48 ramion)." if not missing else f"NIEKOMPLETNY: brakuje {len(missing)} z 48 ramion podstawowych."), "",
           "## Protokół i zakres", "",
           "Badane stany i operatory Bella pochodzą z rejestru repo: `two_qutrit`, `ghz3` (stan grafowy lokalnie równoważny GHZ trzech qutrytów) i `ame43` (AME(4,3)). Kodowanie baseline to `canonical_ez`: 0→00, 1→01, 2→10; 11 jest poza kodem.", "",
           "Przeszukano pełną dyskretną klasę repo: 4 podpory × 6 permutacji × 27 kombinacji faz {1,ω,ω²} = 648 baz. Po usunięciu globalnej fazy pozostaje 216. Ta klasa nie obejmuje wszystkich ciągłych faz monomialnych. Dla każdego stanu wszystkie 216 baz oraz baseline skompilowano lokalnie na obu rzeczywistych targetach z ziarnem 907. Wynik selekcji to średnia, po dwóch dostawcach, stosunku kosztu błędu do baseline. Koszt = średnia po ustawieniach sumy −log(1−error) po bramkach i odczytach. To model selekcyjny, nie prognoza wartości Bella ani dowód globalnego optimum. Trzy wskazane bazy mają najniższy koszt w tym przeszukaniu; pomiary sprzętowe są późniejszą walidacją.", "",
           "F3 zwykłe ma fazę 0 na nieużywanym stanie. Wersja zoptymalizowana używa analitycznej fazy dopełnienia (π/2 albo 11π/6, zależnie od bazy). Działanie na kodzie qutrytu jest identyczne; dokładna synteza pojedynczego F3 redukuje zwykle CZ z 3 do 2. Nie jest to zmiana kodowania na bazę Fouriera. Oba ramiona korzystają z tej samej optymalizacji ważonych krawędzi grafu (w AME43 dwie CZ zastępuje CZ²).", "",
           "Każdą wybraną bazę i baseline dopracowano tą samą procedurą kompilacji z trzema ziarnami (907–909), wybierając minimalny modelowany koszt całego obwodu pomiarowego. Kolejność obwodów w jobie jest losowana deterministycznie. Wyniki i job ID pozostają w katalogach `weighted/<provider>/jobs/`.", "",
           "Każdy końcowy QPY sprawdzono po ponownym odczytaniu: idealna wartość Bella, brak niepoprawnych codewordów oraz zgodność pełnego rozkładu pomiarowego z obwodem wejściowym. Naprawiono dekodowanie pomijanego uczestnika AME43 dla niekanonicznych przestrzeni kodowych; dla monomialnych podpór wymaga tylko X, bez dodatkowej CZ.", "",
           "IBM: `ibm_kingston`, SamplerV2, DD XY4 oraz twirling bramek i pomiarów (16 randomizacji). IQM: realny Garnet. Dane do lokalnej korekcji odczytu pochodzą z dwóch rzeczywistych obwodów kalibracyjnych (wszystkie używane kubity w 0 oraz w 1) w danym jobie. Przyjęto tensorowy model niezależnych błędów odczytu; nie koryguje on dowolnych korelacji ani błędów bramek.", "",
           "## Wyniki sprzętowe", "",
           "± to 1 bootstrap SE. Tabele CSV/JSON zawierają 95% przedziały. Bootstrap obejmuje statystykę pomiarów oraz resampling kalibracji odczytu, wspólny dla ramion z tego samego joba. Nie obejmuje dryfu, błędu modelu odczytu ani modelu ZNE. Porównania i wybór największego zmierzonego wyniku są eksploracyjne; przedziały nie mają korekty wielokrotnych porównań.", "",
           "Bell po postselekcji oznacza odrzucenie shotów z co najmniej jednym wynikiem poza kodem. Bell bez postselekcji przypisuje takim shotom wkład zero zgodnie z repo. Wynik po postselekcji lub mitygacji nie jest bezlukowym dowodem nielokalności; kubity uczestników są na tym samym procesorze.", ""]
    for state in STATES:
        lines += [f"### {state}","",f'Ideał: {bounds[state]["ideal"]:.0f}; nominalna granica klasyczna: {bounds[state]["classical"]:.7f}.',"",
                  "| Baza | Logiczne 0,1,2: indeksy fizyczne | Faza F3 opt / π |","|---|---|---:|"]
        from study import optimal_f3_leakage_phase
        for name in ["canonical_ez"]+selection[state]["selected"]:
            e=encodings()[name]
            lines.append(f'| {name} | {tuple(int(np.argmax(abs(e[:,i]))) for i in range(3))} | {optimal_f3_leakage_phase(e).phase/np.pi:.6f} |')
        lines.append("")
        for provider in ("iqm","ibm"):
            lines += [f"**{provider.upper()}**", "", "| Baza | F3 | Bell po postselekcji | Bell bez postselekcji | Po korekcji odczytu i postselekcji | Poza kodem |",
                      "|---|---|---:|---:|---:|---:|"]
            for name in ["canonical_ez"]+selection[state]["selected"]:
                for variant in VARIANTS:
                    key=(provider,state,name,variant,1)
                    if key not in data:
                        lines.append(f"| {name} | {variant} | brak pomiaru | — | — | — |")
                        continue
                    row,boot=data[key]
                    lines.append(f'| {name} | {variant} | {fmt(row["raw_conditional"])} | {fmt(row["raw_unconditional"])} | {fmt(row["readout_conditional"])} | {100*row["invalid_fraction"]:.2f}% |')
                    item={"provider":provider,"state":state,"name":name,"variant":variant,"factor":1,"invalid_fraction":row["invalid_fraction"]}
                    for k in KEYS:
                        item[k]=row[k]["value"]
                        item[k+"_se"]=row[k]["se"]
                        item[k+"_ci95_low"],item[k+"_ci95_high"]=row[k]["ci95"]
                    flat.append(item)
            lines += ["", "| Baza | Δ Bell: F3 opt − zwykłe (po postselekcji) | Δ opt − baseline opt (po postselekcji) |", "|---|---:|---:|"]
            baseline=(provider,state,"canonical_ez","optimal",1)
            for name in ["canonical_ez"]+selection[state]["selected"]:
                a=(provider,state,name,"optimal",1)
                b=(provider,state,name,"standard",1)
                if a not in data or b not in data or baseline not in data:
                    continue
                f3=difference(data[a],data[b],1)
                gain=difference(data[a],data[baseline],1)
                lines.append(f'| {name} | {fmt(f3)} | {fmt(gain)} |')
                for index,k in enumerate(KEYS):
                    for kind,left,right in (("f3_opt_minus_standard",a,b),("candidate_opt_minus_baseline_opt",a,baseline),
                                             ("candidate_opt_minus_baseline_standard",a,(provider,state,"canonical_ez","standard",1))):
                        if right in data:
                            delta=difference(data[left],data[right],index)
                            deltas.append({"provider":provider,"state":state,"name":name,"comparison":kind,"estimator":k,
                                           "delta":delta["value"],"se":delta["se"],"ci95_low":delta["ci95"][0],"ci95_high":delta["ci95"][1],
                                           "relative_percent":100*delta["value"]/data[right][0][k]["value"] if abs(data[right][0][k]["value"])>1e-12 else None})
            lines.append("")
            available=[(provider,state,n,"optimal",1) for n in selection[state]["selected"] if (provider,state,n,"optimal",1) in data]
            if available and baseline in data:
                best=max(available,key=lambda key:data[key][0]["raw_conditional"]["value"])
                gain=difference(data[best],data[baseline],1)
                standard=(provider,state,best[2],"standard",1)
                lines.append(f'Najwyższy zmierzony wynik kandydata z F3 opt (po postselekcji, bez korekcji odczytu): **{best[2]}**, Bell **{fmt(data[best][0]["raw_conditional"])}**. Różnica względem baseline z F3 opt: **{fmt(gain)}**.')
                if standard in data:
                    f3=difference(data[best],data[standard],1)
                    lines.append(f'Dla tej bazy zmiana F3 zwykłe → opt daje **{fmt(f3)}**. Dodatnia różnica oznacza poprawę; ujemna pogorszenie. Wybór maksimum jest eksploracyjny.')
                lines.append("")
    zne_rows=[]
    for key,(row,boot) in data.items():
        provider,state,name,variant,factor=key
        if factor!=1:
            continue
        keys=[(provider,state,name,variant,f) for f in (1,3,5)]
        if not all(k in data for k in keys):
            continue
        weights=np.array([13/12,1/3,-5/12])
        combined=sum(w*data[k][1] for w,k in zip(weights,keys))
        for i,estimator in enumerate(KEYS):
            value=sum(w*data[k][0][estimator]["value"] for w,k in zip(weights,keys))
            zne_rows.append({"provider":provider,"state":state,"name":name,"variant":variant,"estimator":estimator,
                            "value":float(value),"se":float(np.std(combined[:,i],ddof=1)),
                            "ci95_low":float(np.quantile(combined[:,i],.025)),"ci95_high":float(np.quantile(combined[:,i],.975))})
    if zne_rows:
        lines += ["## ZNE (analiza dodatkowa)","","Liniowa ekstrapolacja ze skal CZ 1,3,5, z ustalonymi wagami 13/12, 1/3, −5/12. Wszystkie wyniki, również pogorszenia i niepewne ekstrapolacje, zapisano w `zne.csv`. Nie wybierano modelu pod najwyższy wynik. Bootstrap nie obejmuje błędu modelu ZNE.",""]
        lines += ["| Dostawca | Stan | Baza | F3 | ZNE + odczyt + postselekcja |","|---|---|---|---|---:|"]
        for r in zne_rows:
            if r["estimator"]=="readout_conditional":
                lines.append(f'| {r["provider"]} | {r["state"]} | {r["name"]} | {r["variant"]} | {r["value"]:.4f} ± {r["se"]:.4f} |')
        lines.append("")
    resources=[]
    lines += ["## Koszt końcowych obwodów", "", "Liczby CZ dotyczą całego zestawu ustawień Bella (9/12/13 obwodów), przed dodaniem DD/twirlingu przez IBM i przed ewentualnym foldingiem ZNE.", "",
              "| Dostawca | Stan | Baza | CZ zwykłe F3 | CZ opt F3 | Redukcja |", "|---|---|---|---:|---:|---:|"]
    for provider in ("iqm","ibm"):
        source=ROOT/provider/"final_compile.json"
        if not source.exists():
            continue
        records=json.loads(source.read_text())
        lookup={(r["state"],r["name"],r["variant"]):r for r in records}
        for state in STATES:
            for name in ["canonical_ez"]+selection[state]["selected"]:
                a=lookup.get((state,name,"standard"))
                b=lookup.get((state,name,"optimal"))
                if a is None or b is None:
                    continue
                delta=a["two_qubit_total"]-b["two_qubit_total"]
                lines.append(f'| {provider} | {state} | {name} | {a["two_qubit_total"]} | {b["two_qubit_total"]} | {delta} |')
                resources.append({"provider":provider,"state":state,"name":name,"cz_standard":a["two_qubit_total"],
                                  "cz_optimal":b["two_qubit_total"],"cz_reduction":delta,
                                  "depth_standard":a["depth_max"],"depth_optimal":b["depth_max"],
                                  "error_proxy_standard":a["error_proxy"],"error_proxy_optimal":b["error_proxy"]})
    write_csv(ROOT/"resources.csv",resources)
    lines += ["", "## Budżet i dowody wykonania", "", "| Dostawca | Job | Status | Czas [s] | Rodzaj |", "|---|---|---|---:|---|"]
    budgets={}
    for provider in ("iqm","ibm"):
        rows=[json.loads(p.read_text()) for p in (ROOT/provider/"jobs").glob("*/receipt.json")]
        for r in rows:
            kind="rozliczony" if "charged_seconds" in r else ("górne ograniczenie QPU z timeline" if "accounted_seconds_upper_bound" in r else "rezerwa")
            lines.append(f'| {provider} | {r.get("job_id",r["label"])} | {r["status"]} | {budget_charge(r):.3f} | {kind} |')
        budgets[provider]={"accounted_seconds":sum(budget_charge(r) for r in rows),
                           "all_accounted_as_actual":all("charged_seconds" in r for r in rows)}
    lines += ["", "Limit tej kampanii: 540 s IBM, 1000 s IQM (500 kredytów, stawka podana przez użytkownika: 1 kredyt / 2 s). Wpis bez potwierdzonego zużycia zachowuje pełną rezerwę. Czas oczekiwania w kolejce nie jest czasem QPU.","",
              "IBM ma twardy limit czasu każdego zadania. IQM nie udostępnia analogicznego limitu: dalsze rezerwacje wyznacza zmierzony koszt pilota lub górne ograniczenie z timeline, sumy kalibrowanych czasów bramek, mnożnik bezpieczeństwa 2 oraz 30 s zapasu. Górne ograniczenie IQM obejmuje cały przedział od przyjęcia przez station control do ukończenia przez server, włącznie z kompilacją i przetwarzaniem wyników; nie jest fakturą. Batch IQM ma rezerwę najwyżej 240 s. Osobny proces nadzoruje czas przetwarzania i zleca anulowanie 30 s przed końcem rezerwy. To kontrola klienta, nie gwarancja dostawcy; zależy od łączności i czasu anulowania.", "",
              "Dokumentacja dostawców: [IBM: limit czasu](https://quantum.cloud.ibm.com/docs/en/guides/max-execution-time), [IBM: zużycie](https://quantum.cloud.ibm.com/docs/en/guides/estimate-job-run-time), [IQM Resonance](https://iqm.tech/products/iqm-resonance/).", ""]
    write_csv(ROOT/"results.csv",flat)
    write_csv(ROOT/"differences.csv",deltas)
    write_csv(ROOT/"zne.csv",zne_rows)
    save_json(ROOT/"summary.json",{"complete":not missing,"missing":missing,"budgets":budgets,"results":flat,"differences":deltas,"zne":zne_rows})
    (ROOT/"RAPORT.md").write_text("\n".join(lines),encoding="utf-8")
    if flat:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
        for i,state in enumerate(STATES):
            for j,provider in enumerate(("iqm","ibm")):
                ax=axes[i,j]
                names=["canonical_ez"]+selection[state]["selected"]
                for variant,dx,color in (("standard",-.12,"#475569"),("optimal",.12,"#007f86")):
                    rows=[data.get((provider,state,n,variant,1)) for n in names]
                    for index,item in enumerate(rows):
                        if item:
                            estimate=item[0]["raw_conditional"]
                            ax.errorbar(index+dx,estimate["value"],yerr=1.96*estimate["se"],fmt="o",color=color,label=variant if index==0 else None,capsize=4)
                ax.axhline(bounds[state]["classical"],color="#b45309",linestyle="--",label="próg klasyczny")
                ax.axhline(bounds[state]["ideal"],color="#64748b",linestyle=":",label="ideał")
                ax.set_xticks(range(4),["baseline","kand. 1","kand. 2","kand. 3"])
                ax.set_title(f"{provider.upper()} · {state}")
                ax.set_ylabel("Bell po postselekcji; bez korekcji odczytu")
                ax.grid(axis="y",alpha=.2)
                if i==0 and j==0:
                    ax.legend(fontsize=8)
        fig.savefig(ROOT/"bell_comparison.png",dpi=180)
        plt.close(fig)
    print(json.dumps({"complete":not missing,"missing":len(missing),"report":str(ROOT/"RAPORT.md")}),flush=True)


if __name__=="__main__":
    report()
