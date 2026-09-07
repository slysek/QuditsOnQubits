"""Bell estimates, readout correction and uncertainty from saved QPU counts."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from study import *


def count_array(counts,width):
    output=np.zeros(2**width)
    for bits,n in counts.items():
        output[int(bits.replace(" ",""),2)]+=n
    return output


def inverse_readout_vectors(vector,inverses,physical):
    """Pull a true-outcome score back through a tensor-product readout channel."""
    out=vector.copy()
    for bit,q in enumerate(physical):
        view=out.reshape(-1,2,2**bit)
        zero=view[:,0,:].copy()
        one=view[:,1,:].copy()
        inv=inverses[q]
        view[:,0,:]=inv[0,0]*zero+inv[1,0]*one
        view[:,1,:]=inv[0,1]*zero+inv[1,1]*one
    return out


def calibration_models(receipt,counts,samples,rng):
    indices={m["calibration"]:i for i,m in enumerate(receipt["mapping"]) if "calibration" in m}
    if set(indices)!={0,1}:
        raise ValueError("requires both readout calibration states")
    physical=receipt["mapping"][indices[0]]["physical_by_classical"]
    observed=[]
    resampled=[]
    for value in (0,1):
        values=counts[indices[value]]
        strings=list(values)
        n=np.array([values[s] for s in strings])
        bits=np.array([[(int(s.replace(" ",""),2)>>i)&1 for i in range(len(physical))] for s in strings])
        observed.append(n @ bits/n.sum())
        draws=rng.multinomial(int(n.sum()),n/n.sum(),size=samples)
        resampled.append(draws @ bits/n.sum())
    def model(p0,p1):
        output={}
        for i,q in enumerate(physical):
            matrix=np.array([[1-p0[i],1-p1[i]],[p0[i],p1[i]]])
            if abs(np.linalg.det(matrix))<.2:
                raise ValueError("readout calibration is too ill-conditioned")
            output[q]=np.linalg.inv(matrix)
        return output
    return model(*observed),[model(resampled[0][i],resampled[1][i]) for i in range(samples)]


def estimate_group(group_counts,physical,meta,model,bootstrap_models,rng):
    width=2*len(meta["qutrit_qubits"])
    scores,accepted=outcome_scores(meta,width)
    arrays=np.asarray([count_array(c,width) for c in group_counts])
    total=arrays.sum(axis=1)
    if np.any(total<=0):
        raise ValueError("empty measurement setting")
    samples=len(bootstrap_models)
    points=np.zeros(4)
    draws=np.zeros((samples,4))
    for counts,sc,ok,qubits,n in zip(arrays,scores,accepted,physical,total):
        numerator=counts@sc
        denominator=counts@ok
        if denominator<=0:
            raise ValueError("no accepted outcomes")
        corrected_sc=inverse_readout_vectors(sc,model,qubits)
        corrected_ok=inverse_readout_vectors(ok.astype(float),model,qubits)
        cnum=counts@corrected_sc
        cden=counts@corrected_ok
        if cden<=0:
            raise ValueError("nonpositive corrected normalization")
        points+=np.array([numerator/n,numerator/denominator,cnum/n,cnum/cden])
        sampled=rng.multinomial(int(n),counts/n,size=samples)
        snum=sampled@sc
        sden=sampled@ok
        if np.any(sden<=0):
            raise ValueError("bootstrap produced no accepted outcomes")
        draws[:,0]+=snum/n
        draws[:,1]+=snum/sden
        for i,m in enumerate(bootstrap_models):
            sc_corr=inverse_readout_vectors(sc,m,qubits)
            ok_corr=inverse_readout_vectors(ok.astype(float),m,qubits)
            num=sampled[i]@sc_corr
            den=sampled[i]@ok_corr
            if den<=0:
                raise ValueError("bootstrap produced nonpositive corrected normalization")
            draws[i,2]+=num/n
            draws[i,3]+=num/den
    result={"total_shots":int(total.sum()),"shots_per_setting":total.tolist(),
            "invalid_fraction":float(1-np.sum(arrays*accepted)/total.sum())}
    keys=("raw_unconditional","raw_conditional","readout_unconditional","readout_conditional")
    for i,key in enumerate(keys):
        result[key]={"value":float(points[i]),"se":float(np.std(draws[:,i],ddof=1)),
                     "ci95":np.quantile(draws[:,i],[.025,.975]).tolist()}
    return result,draws


def analyze_job(path,samples=2000):
    receipt=json.loads(path.read_text())
    if receipt["status"]!="completed":
        return []
    counts=json.loads((path.parent/"counts.json").read_text())
    seed=int(hashlib.sha256(receipt["job_id"].encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(seed)
    model,models=calibration_models(receipt,counts,samples,rng)
    groups=defaultdict(dict)
    for i,meta in enumerate(receipt["mapping"]):
        if "calibration" not in meta:
            groups[(meta["state"],meta["name"],meta["variant"])][meta["setting_index"]]=i
    results=[]
    boot={}
    for index,(key,items) in enumerate(sorted(groups.items())):
        state,name,variant=key
        _,meta=workload(state,name,variant)
        expected=len(meta["setting_by_circuit_index"])
        if sorted(items)!=list(range(expected)):
            raise ValueError("missing measurement settings")
        indices=[items[i] for i in range(expected)]
        group_counts=[counts[i] for i in indices]
        physical=[receipt["mapping"][i]["physical_by_classical"] for i in indices]
        result,draws=estimate_group(group_counts,physical,meta,model,models,rng)
        result.update(state=state,name=name,variant=variant,provider=receipt["provider"],factor=receipt["factor"],
                      job_id=receipt["job_id"],label=receipt["label"],pilot=receipt.get("pilot",False),
                      bootstrap_key=str(index),bootstrap_samples=samples)
        results.append(result)
        boot[str(index)]=draws
        print(json.dumps({"provider":receipt["provider"],"state":state,"name":name,"variant":variant,
                          "factor":receipt["factor"],"raw":result["raw_conditional"],"readout":result["readout_conditional"]}),flush=True)
    save_json(path.parent/"analysis.json",results)
    np.savez_compressed(path.parent/"bootstrap.npz",**boot)
    return results


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--provider",choices=["iqm","ibm"],required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--samples",type=int,default=2000)
    args=parser.parse_args()
    analyze_job(ROOT/args.provider/"jobs"/args.label/"receipt.json",args.samples)
