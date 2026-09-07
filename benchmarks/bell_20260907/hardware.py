"""Read live targets, compile locally, and retain hardware execution receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time

from study import *


def backend_for(provider):
    if provider == "iqm":
        from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import load_iqm_backend
        return load_iqm_backend("garnet",use_metrics=True), None
    if provider == "ibm":
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qudits_on_qubits._ibm_runtime import runtime_account_options
        logging.getLogger("qiskit_ibm_runtime").setLevel(logging.ERROR)
        service=QiskitRuntimeService(**runtime_account_options())
        return service.backend("ibm_kingston"), service
    raise ValueError("unknown provider")


def target_compile(provider, seeds=3):
    backend,_ = backend_for(provider)
    shortlist=json.loads((ROOT/"shortlist.json").read_text())
    target_root=ROOT/provider
    target_root.mkdir(exist_ok=True)
    record_path=target_root/"final_compile.json"
    records=json.loads(record_path.read_text()) if record_path.exists() else []
    finished={(r["state"],r["name"],r["variant"]) for r in records}
    errors={name:{str(qubits):{"error":p.error,"duration":p.duration} for qubits,p in backend.target[name].items() if p is not None}
            for name in backend.target.operation_names}
    save_json(target_root/"calibration.json",{"backend":backend.name,"captured_at":datetime.now(timezone.utc).isoformat(),
              "calibration_id":str(getattr(backend,"_calibration_set_id","")),"operations":errors})
    for state in STATES:
        for name in shortlist[state]:
            for variant in VARIANTS:
                if (state,name,variant) in finished:
                    continue
                source,meta=workload(state,name,variant)
                best=None
                trials=[]
                for seed in range(seeds):
                    # Compile the complete measured circuit. Final RZ removal
                    # on IQM is valid here because computational readout follows.
                    compiled=transpile(source,backend=backend,optimization_level=3,
                                       seed_transpiler=907+seed,num_processes=1)
                    compiled=[normalize_circuit_bit_indices(c) for c in compiled]
                    m=metrics(compiled,backend)
                    key=(m["error_proxy"],m["two_qubit_total"],m["depth_max"])
                    trials.append({"seed":907+seed,**m})
                    if best is None or key<best[0]:
                        best=(key,compiled,m,907+seed)
                directory=target_root/"final"/state/name/variant
                directory.mkdir(parents=True,exist_ok=True)
                with (directory/"circuits.qpy").open("wb") as f:
                    qpy.dump(best[1],f)
                record={"provider":provider,"backend":backend.name,"state":state,"name":name,"variant":variant,
                        "seed":best[3],"path":str(directory.relative_to(ROOT)),"trials":trials,**best[2]}
                records.append(record)
                save_json(record_path,records)
                print(json.dumps({k:record[k] for k in ("provider","state","name","variant","two_qubit_total","error_proxy")}),flush=True)
    return records


def choose(states=STATES):
    data={p:json.loads((ROOT/p/"compile.json").read_text()) for p in ("iqm","ibm")}
    path=ROOT/"selection.json"
    selection=json.loads(path.read_text()) if path.exists() else {}
    for state in states:
        maps={p:{r["name"]:r for r in rows if r["state"]==state and r["variant"]=="optimal"} for p,rows in data.items()}
        ranked=[]
        for name in sorted(set(maps["iqm"]) & set(maps["ibm"])):
            if name in ("canonical_ez","sup012_P012_ph000"):
                continue
            ratios={p:maps[p][name]["error_proxy"]/maps[p]["canonical_ez"]["error_proxy"] for p in maps}
            ranked.append({"name":name,"relative_error_proxy_mean":float(np.mean(list(ratios.values()))),"ratios":ratios})
        ranked.sort(key=lambda r:(r["relative_error_proxy_mean"],r["name"]))
        if len(ranked)<3:
            raise ValueError("fewer than three nonbaseline candidates")
        selection[state]={"selected":[r["name"] for r in ranked[:3]],"ranking":ranked}
    save_json(ROOT/"selection.json",selection)
    print(json.dumps(selection),flush=True)


def verify_selected(provider):
    selection=json.loads((ROOT/"selection.json").read_text())
    evidence=[]
    for state in selection:
        for name in ["canonical_ez"]+selection[state]["selected"]:
            previous=None
            for variant in VARIANTS:
                source,meta=workload(state,name,variant)
                expected=np.asarray([exact_probabilities(c) for c in source])
                path=ROOT/provider/"final"/state/name/variant/"circuits.qpy"
                with path.open("rb") as f:
                    compiled=qpy.load(f)
                actual=np.asarray([exact_probabilities(c) for c in compiled])
                tvd=float(np.max(.5*np.sum(np.abs(expected-actual),axis=1)))
                scores,accepted=outcome_scores(meta,source[0].num_qubits)
                bell=float(np.sum(actual*scores))
                target=get_reference_experiment(state).expected.ideal_bell_value
                if tvd>1e-8 or abs(bell-target)>1e-8 or np.sum(actual*(~accepted))>1e-8:
                    raise ValueError(f"verification failed: {provider} {state} {name} {variant}: {tvd} {bell}")
                if previous is not None and np.max(np.abs(expected-previous))>1e-9:
                    raise ValueError("F3 arm probabilities differ")
                previous=expected
                evidence.append({"state":state,"name":name,"variant":variant,"max_tvd":tvd,"ideal_bell":bell,
                                 "qpy_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
                save_json(ROOT/provider/"verification.json",evidence)
                print(json.dumps(evidence[-1]),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["compile","choose","verify"])
    parser.add_argument("--provider",choices=["iqm","ibm"])
    parser.add_argument("--seeds",type=int,default=3)
    parser.add_argument("--states",nargs="+",choices=STATES,default=list(STATES))
    args=parser.parse_args()
    if args.action=="compile":
        target_compile(args.provider,args.seeds)
    elif args.action=="choose":
        choose(args.states)
    else:
        verify_selected(args.provider)
