"""Exhaustive real-target screening; workers only compile locally."""
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import argparse
import json
import time
from study import *
from hardware import backend_for

TARGET=None
PLUGIN_OPTIONS={}


def initialize_worker(target,options,attributes):
    global TARGET,PLUGIN_OPTIONS
    TARGET=target
    # Qiskit's Rust Target pickle does not retain IQM subclass attributes.
    # Restore the architecture/index metadata explicitly (no client/account).
    for key,value in attributes.items():
        setattr(TARGET,key,value)
    PLUGIN_OPTIONS=options


def compile_item(item):
    provider,state,name=item
    source,_=workload(state,name,"optimal")
    compiled=transpile(source,target=TARGET,optimization_level=3,seed_transpiler=907,num_processes=1,**PLUGIN_OPTIONS)
    compiled=[normalize_circuit_bit_indices(c) for c in compiled]
    directory=ROOT/provider/"exhaustive"/state/name/"optimal"
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"circuits.qpy").open("wb") as f:
        qpy.dump(compiled,f)
    return {"provider":provider,"state":state,"name":name,"variant":"optimal","seed":907,
            "path":str(directory.relative_to(ROOT)),**metrics(compiled,SimpleNamespace(target=TARGET))}


def run(provider,workers):
    backend,_=backend_for(provider)
    options={}
    for method,key in (("get_translation_stage_plugin","translation_method"),("get_scheduling_stage_plugin","scheduling_method")):
        if hasattr(backend,method):
            options[key]=getattr(backend,method)()
    path=ROOT/provider/"exhaustive_compile.json"
    rows=json.loads(path.read_text()) if path.exists() else []
    done={(r["state"],r["name"]) for r in rows}
    todo=[(provider,state,name) for state in STATES for name in encodings() if (state,name) not in done]
    start=time.time()
    attributes=dict(vars(backend.target)) if provider=="iqm" else {}
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(backend.target,options,attributes)) as pool:
        for i,r in enumerate(pool.map(compile_item,todo,chunksize=1)):
            rows.append(r)
            if i%6==0:
                save_json(path,rows)
                print(json.dumps({"provider":provider,"done":len(rows),"total":651,"seconds":round(time.time()-start)}),flush=True)
    save_json(path,rows)


def choose_exhaustive():
    data={p:json.loads((ROOT/p/"exhaustive_compile.json").read_text()) for p in ("iqm","ibm")}
    if any(len(r)!=651 for r in data.values()):
        raise ValueError("exhaustive search is incomplete")
    selection={}
    for state in STATES:
        lookup={p:{r["name"]:r for r in rows if r["state"]==state} for p,rows in data.items()}
        rank=[]
        for name in encodings():
            if name in ("canonical_ez","sup012_P012_ph000"):
                continue
            ratios={p:lookup[p][name]["error_proxy"]/lookup[p]["canonical_ez"]["error_proxy"] for p in lookup}
            rank.append({"name":name,"relative_error_proxy_mean":float(np.mean(list(ratios.values()))),"ratios":ratios})
        rank.sort(key=lambda r:(r["relative_error_proxy_mean"],r["name"]))
        selection[state]={"selected":[r["name"] for r in rank[:3]],"ranking":rank}
    save_json(ROOT/"selection.json",selection)
    # Refine both arms of the complete selected hardware workload with the same
    # three-seed search. Canonical receives identical optimization opportunities.
    save_json(ROOT/"shortlist.json",{s:["canonical_ez"]+v["selected"] for s,v in selection.items()})
    print(json.dumps({s:v["selected"] for s,v in selection.items()}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["screen","choose"])
    parser.add_argument("--provider",choices=["iqm","ibm"])
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args()
    if args.action=="screen":
        run(args.provider,args.workers)
    else:
        choose_exhaustive()
