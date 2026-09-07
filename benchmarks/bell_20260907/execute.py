"""Explicit single-submit execution, bounded reservations, durable job receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

from study import *
from hardware import backend_for
from qudits_on_qubits.experiments.mitigation.zne import fold_cz_batch

LIMITS={"ibm":540.0,"iqm":1000.0}
IQM_MAX_BATCH_RESERVATION=240.0


def validate_batch_size(provider,count):
    if provider=="iqm" and count>100:
        raise ValueError("IQM accepts at most 100 circuits per job; split complete measurement groups")


def budget_charge(receipt):
    charge=receipt["charged_seconds"] if "charged_seconds" in receipt else (receipt["accounted_seconds_upper_bound"] if "accounted_seconds_upper_bound" in receipt else receipt["reservation_seconds"])
    if type(charge) not in (float,int) or not math.isfinite(charge) or charge<0:
        raise ValueError("invalid saved budget accounting")
    return charge


def iqm_timeline_accounting(payload):
    """Bound QPU occupation by complete provider processing, not invoice cost."""
    if payload.get("status")!="completed":
        raise ValueError("requires completed IQM timeline")
    def stamp(source,status):
        entries=[e for e in payload["timeline"] if e["source"]==source and e["status"]==status]
        if len(entries)!=1:
            raise ValueError("ambiguous or missing provider timing evidence")
        return datetime.fromisoformat(entries[0]["timestamp"].replace("Z","+00:00"))
    received=stamp("iqm-station-control","received")
    started=stamp("iqm-station-control","execution_started")
    ended=stamp("iqm-station-control","execution_ended")
    completed=stamp("iqm-server","completed")
    if not received<=started<ended<=completed:
        raise ValueError("inconsistent provider timeline")
    return {"accounted_seconds_upper_bound":(completed-received).total_seconds(),
            "instrument_execution_wall_seconds":(ended-started).total_seconds(),
            "accounting_source":"timeline.json: station received to server completed; QPU upper bound, not invoice"}


def record_iqm_accounting(label,evidence_path):
    """Import verified provider billing evidence; never infer billing from queue time."""
    directory=job_directory("iqm",label)
    receipt=json.loads((directory/"receipt.json").read_text())
    evidence=json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    seconds=evidence.get("billed_seconds")
    if receipt["status"]!="completed" or evidence.get("job_id")!=receipt.get("job_id"):
        raise ValueError("billing evidence must match the completed job")
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError("billing seconds must be finite and positive")
    if not evidence.get("source") or not evidence.get("verified_by"):
        raise ValueError("billing evidence requires source and verification attribution")
    save_json(directory/"billing_evidence.json",evidence)
    receipt["charged_seconds"]=float(seconds)
    receipt["accounting_source"]="billing_evidence.json"
    save_json(directory/"receipt.json",receipt)


def serial_duration(circuit,backend):
    """Conservative sum of calibrated gate durations, ignoring parallelism."""
    value=0.0
    for inst in circuit.data:
        if inst.operation.name=="barrier":
            continue
        prop=backend.target[inst.operation.name].get(tuple(circuit.find_bit(q).index for q in inst.qubits))
        if prop is None or prop.duration is None or not math.isfinite(prop.duration) or prop.duration<0:
            raise ValueError("IQM runtime model requires calibrated durations for all operations")
        value+=prop.duration
    return value


def iqm_reservation(circuits,shots,backend):
    pilots=[]
    for path in (ROOT/"iqm/jobs").glob("*/receipt.json"):
        receipt=json.loads(path.read_text())
        if receipt.get("pilot") and receipt["status"]=="completed" and ("charged_seconds" in receipt or "accounted_seconds_upper_bound" in receipt):
            pilots.append(receipt)
    if not pilots:
        raise ValueError("IQM requires completed pilot and verified duration before full batch")
    # Charge ALL pilot overhead to every shot, then double that observed cost.
    # Add the sum of durations of every target gate (without parallelism),
    # again doubled, plus 30 s. This is a conservative estimate, not a server cap.
    per_shot=max(budget_charge(p)/(p["shots"]*p["circuit_count"]) for p in pilots)
    seconds=2*shots*(len(circuits)*per_shot+sum(serial_duration(c,backend) for c in circuits))+30
    return seconds,{"pilot_job_ids":[p["job_id"] for p in pilots],"observed_seconds_per_shot":per_shot,
                    "safety_factor":2,"overhead_seconds":30,"estimated_seconds":seconds,"hard_server_cap":False}


def circuit_signature(circuit):
    value=[circuit.num_qubits,circuit.num_clbits,str(circuit.global_phase),
           [[i.operation.name,[str(p) for p in i.operation.params],
             [circuit.find_bit(q).index for q in i.qubits],
             [circuit.find_bit(c).index for c in i.clbits]] for i in circuit.data]]
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def recover(provider,label,job_id=None):
    """Attach an existing remote job only after matching its submitted payload."""
    directory=job_directory(provider,label)
    receipt=json.loads((directory/"receipt.json").read_text())
    if receipt["status"]!="submitting" or receipt.get("job_id"):
        raise ValueError("recovery is only for uncertain submission without a job ID")
    backend,service=backend_for(provider)
    with (directory/"submitted.qpy").open("rb") as f:
        circuits=qpy.load(f)
    if hashlib.sha256((directory/"submitted.qpy").read_bytes()).hexdigest()!=receipt["payload_sha256"]:
        raise ValueError("saved payload changed")
    if provider=="ibm":
        if job_id is None:
            token=receipt.get("submission_token")
            if not token:
                raise ValueError("legacy uncertain job requires provider job ID")
            jobs=service.jobs(limit=2,job_tags=[token],backend_name=receipt["backend"])
            if len(jobs)!=1:
                raise ValueError("cannot uniquely identify remote job; no resubmission allowed")
            job=jobs[0]
        else:
            job=service.job(job_id)
        if job.backend().name!=receipt["backend"]:
            raise ValueError("remote backend mismatch")
        pubs=job.inputs["pubs"]
        remote=[p.circuit if hasattr(p,"circuit") else p[0] for p in pubs]
        remote_shots=[p.shots if hasattr(p,"shots") else p[2] for p in pubs]
        if any(n!=receipt["shots"] for n in remote_shots):
            raise ValueError("remote shots mismatch")
        if [circuit_signature(c) for c in remote]!=[circuit_signature(c) for c in circuits]:
            raise ValueError("remote circuit fingerprint mismatch")
        job_id=job.job_id()
    else:
        if job_id is None:
            raise ValueError("IQM SDK cannot list jobs: supply the job ID from provider dashboard; no resubmission allowed")
        actual=backend.client._iqm_server_client.get_submit_circuits_payload(job_id)
        expected=backend.create_run_request(circuits,shots=receipt["shots"])
        if actual.shots!=expected.shots or actual.circuits!=expected.circuits:
            raise ValueError("remote IQM payload mismatch")
    receipt.update(status="submitted",job_id=job_id,recovered_at=datetime.now(timezone.utc).isoformat())
    save_json(directory/"receipt.json",receipt)
    return job_id


def job_directory(provider,label):
    if provider not in LIMITS or not re.fullmatch(r"[a-z0-9_-]{1,48}",label):
        raise ValueError("invalid provider or job label")
    return ROOT/provider/"jobs"/label


def used_budget(provider):
    committed=0.0
    for path in (ROOT/provider/"jobs").glob("*/receipt.json"):
        r=json.loads(path.read_text())
        committed+=budget_charge(r)
    return committed


def check_budget(provider,reservation):
    if not math.isfinite(reservation) or reservation<=0:
        raise ValueError("reservation must be positive and finite")
    if used_budget(provider)+reservation>LIMITS[provider]:
        raise ValueError("campaign QPU budget exceeded")
    if provider=="iqm" and reservation>IQM_MAX_BATCH_RESERVATION:
        raise ValueError("split IQM workload: at most 240 seconds may be reserved per batch")


def guard_decision(status,timeline,now,reservation,first_processing=None):
    """Conservative processing wall-time guard; queued time is never charged."""
    if status!="processing":
        return False,first_processing,0.0
    # The earliest provider event after creation bounds all subsequent QPU work.
    candidates=[datetime.fromisoformat(e["timestamp"].replace("Z","+00:00"))
                for e in timeline if e["status"]!="created"]
    observed=now.timestamp()-10  # Allow the preceding polling interval.
    start=min([observed]+[t.timestamp() for t in candidates])
    if first_processing is not None:
        start=min(start,first_processing)
    elapsed=max(0.0,now.timestamp()-start)
    return elapsed>=max(5.0,reservation-30),start,elapsed


def guard_iqm(label):
    """Monitor this one authorized job and cancel before its reserved time expires.

    IQM has no hard server runtime cap. Network/cancellation latency cannot be
    guaranteed, so reserve 30 s for cancellation and keep batches <=240 s.
    """
    directory=job_directory("iqm",label)
    receipt=json.loads((directory/"receipt.json").read_text())
    backend,_=backend_for("iqm")
    first_processing=None
    failures=0
    while True:
        try:
            remote=backend.client.get_job(UUID(receipt["job_id"])).data
            payload=remote.model_dump(mode="json")
            now=datetime.now(timezone.utc)
            status=str(remote.status).lower()
            cancel,first_processing,elapsed=guard_decision(status,payload.get("timeline",[]),now,
                                                          receipt["reservation_seconds"],first_processing)
            if cancel:
                backend.client._iqm_server_client.cancel_job(receipt["job_id"])
            save_json(directory/"guard.json",{"observed_at":now.isoformat(),"status":status,
                       "processing_elapsed_upper_estimate":elapsed,"cancel_requested":cancel})
            if cancel or status in ("completed","failed","cancelled"):
                return
            failures=0
        except Exception as error:
            failures+=1
            print(type(error).__name__,flush=True)
            if failures>=3:
                # Fail closed when we can no longer track processing time.
                backend.client._iqm_server_client.cancel_job(receipt["job_id"])
                save_json(directory/"guard.json",{"cancel_requested":True,"reason":"monitor_unavailable"})
                return
        time.sleep(2 if first_processing is not None else 10)


def start_iqm_guard(label):
    directory=job_directory("iqm",label)
    with (directory/"guard.log").open("a") as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"guard","--provider","iqm","--label",label],
                                 stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,"IQM_CLIENT_REQUESTS_TIMEOUT":"5"},
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    save_json(directory/"guard_process.json",{"pid":process.pid,"started_at":datetime.now(timezone.utc).isoformat()})


def make_batch(provider,backend,factor,seed,states,pilot=False):
    if factor not in (1,3,5):
        raise ValueError("only preregistered ZNE factors 1,3,5 allowed")
    selection=json.loads((ROOT/"selection.json").read_text())
    verified=json.loads((ROOT/provider/"verification.json").read_text())
    hashes={(r["state"],r["name"],r["variant"]):r["qpy_sha256"] for r in verified}
    circuits=[]
    mapping=[]
    for state in states:
        for name in ["canonical_ez"]+([] if pilot else selection[state]["selected"]):
            for variant in VARIANTS:
                key=(state,name,variant)
                path=ROOT/provider/"final"/state/name/variant/"circuits.qpy"
                if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[key]:
                    raise ValueError("QPY differs from verified circuit")
                with path.open("rb") as f:
                    loaded=qpy.load(f)
                for index,c in enumerate(fold_cz_batch(loaded,factor)):
                    measured={c.find_bit(inst.clbits[0]).index:c.find_bit(inst.qubits[0]).index
                              for inst in c.data if inst.operation.name=="measure"}
                    circuits.append(c)
                    mapping.append({"state":state,"name":name,"variant":variant,"setting_index":index,
                                    "physical_by_classical":[measured[i] for i in range(c.num_clbits)]})
    used=sorted({q for m in mapping for q in m["physical_by_classical"]})
    for value in (0,1):
        qc=QuantumCircuit(backend.num_qubits,len(used),name=f"readout_cal_{value}")
        if value:
            qc.x(used)
        for i,q in enumerate(used):
            qc.measure(q,i)
        qc=transpile(qc,backend=backend,initial_layout=list(range(backend.num_qubits)),optimization_level=0,num_processes=1)
        circuits.append(normalize_circuit_bit_indices(qc))
        mapping.append({"calibration":value,"physical_by_classical":used})
    order=np.random.default_rng(seed).permutation(len(circuits))
    circuits=[circuits[i] for i in order]
    mapping=[mapping[i] for i in order]
    # Unique names simplify remote provenance and avoid IQM name constraints.
    for i,c in enumerate(circuits):
        c.name=f"bell_{i:04d}"
    return circuits,mapping


def ibm_runtime_options(receipt,backend):
    shots=receipt["shots"]
    if type(shots) is not int or not 128<=shots<=8192:
        raise ValueError("shots must be in 128..8192")
    randomizations=math.gcd(16,shots)
    options={"max_execution_time":int(receipt["reservation_seconds"]),
             "dynamical_decoupling":{"enable":True,"sequence_type":"XY4"},
             "twirling":{"enable_gates":True,"enable_measure":True,
                         "num_randomizations":randomizations,"shots_per_randomization":shots//randomizations}}
    delay=receipt.get("rep_delay")
    if delay is not None:
        bounds=backend.configuration().rep_delay_range
        if type(delay) not in (int,float) or not math.isfinite(delay) or not bounds[0]<=delay<=bounds[1]:
            raise ValueError("rep_delay must be finite and within the live backend range")
        options["execution"]={"init_qubits":True,"rep_delay":delay}
    if receipt.get("submission_token"):
        options["environment"]={"job_tags":[receipt["submission_token"]]}
    return options


def prepare(provider,label,shots,factor,reservation,states,pilot=False,rep_delay=None):
    if not re.fullmatch(r"[a-z0-9_-]{1,48}",label):
        raise ValueError("invalid job label")
    if type(shots) is not int or shots<128 or shots>8192:
        raise ValueError("shots must be in 128..8192")
    if provider!="ibm" and rep_delay is not None:
        raise ValueError("rep_delay is supported only for IBM")
    if not states or any(s not in STATES for s in states) or len(states)!=len(set(states)):
        raise ValueError("states must be a nonempty distinct supported list")
    check_budget(provider,reservation)
    backend,service=backend_for(provider)
    if provider=="ibm":
        remaining=service.usage().get("usage_remaining_seconds")
        if remaining is None or remaining<reservation:
            raise ValueError("insufficient verified IBM account budget")
        if backend.configuration().simulator or not backend.status().operational:
            raise ValueError("requires operational real IBM backend")
        ibm_runtime_options({"shots":shots,"reservation_seconds":reservation,"rep_delay":rep_delay},backend)
    if pilot and (states != ["two_qutrit"] or shots>256 or factor!=1):
        raise ValueError("pilot is limited to two-qutrit baseline at <=256 shots, factor 1")
    if provider=="iqm" and not pilot:
        pilot_paths=list((ROOT/provider/"jobs").glob("*/receipt.json"))
        if not any((r:=json.loads(p.read_text())).get("pilot") and r["status"]=="completed" and ("charged_seconds" in r or "accounted_seconds_upper_bound" in r) for p in pilot_paths):
            raise ValueError("IQM requires completed pilot and verified duration before full batch")
    circuits,mapping=make_batch(provider,backend,factor,907+factor+shots,states,pilot)
    validate_batch_size(provider,len(circuits))
    model=None
    estimated=2e-3*shots*len(circuits)+30  # Small initial pilot only.
    if provider=="iqm" and not pilot:
        estimated,model=iqm_reservation(circuits,shots,backend)
    if provider=="iqm" and reservation<estimated:
        raise ValueError(f"IQM conservative reservation requires {estimated:.1f} seconds")
    directory=job_directory(provider,label)
    directory.mkdir(parents=True,exist_ok=False)
    token="bell-"+uuid4().hex
    for i,c in enumerate(circuits):
        c.name=f"{token}_{i:04d}"
    with (directory/"submitted.qpy").open("wb") as f:
        qpy.dump(circuits,f)
    receipt={"provider":provider,"backend":backend.name,"label":label,"status":"prepared",
             "shots":shots,"factor":factor,"states":list(states),"reservation_seconds":reservation,"pilot":pilot,
             "mapping":mapping,"circuit_count":len(circuits),
             "payload_sha256":hashlib.sha256((directory/"submitted.qpy").read_bytes()).hexdigest(),
             "created_at":datetime.now(timezone.utc).isoformat()}
    receipt.update(submission_token=token,runtime_estimate=model)
    if rep_delay is not None:
        receipt["rep_delay"]=rep_delay
    save_json(directory/"receipt.json",receipt)
    return directory


def submit(provider,label):
    directory=job_directory(provider,label)
    path=directory/"receipt.json"
    receipt=json.loads(path.read_text())
    if receipt["status"]!="prepared":
        raise ValueError("refuse duplicate or uncertain submission; retrieve existing job instead")
    if used_budget(provider)>LIMITS[provider]:
        raise ValueError("budget exceeded")
    payload=directory/"submitted.qpy"
    if hashlib.sha256(payload.read_bytes()).hexdigest()!=receipt["payload_sha256"]:
        raise ValueError("payload changed")
    with payload.open("rb") as f:
        circuits=qpy.load(f)
    validate_batch_size(provider,len(circuits))
    backend,service=backend_for(provider)
    if backend.name != receipt["backend"]:
        raise ValueError("backend identity changed")
    if provider=="ibm":
        if service.usage().get("usage_remaining_seconds",0)<receipt["reservation_seconds"]:
            raise ValueError("account budget decreased")
        from qiskit_ibm_runtime import SamplerV2
        options=ibm_runtime_options(receipt,backend)
        receipt["runtime_options"]=options
        sampler=SamplerV2(mode=backend,options=options)
    # An atomic exclusive claim prevents concurrent duplicate submissions.
    with (directory/"submission.claim").open("x") as f:
        f.write(datetime.now(timezone.utc).isoformat())
    receipt["status"]="submitting"
    save_json(path,receipt)
    if provider=="ibm":
        job=sampler.run(circuits,shots=receipt["shots"])
    else:
        job=backend.run(circuits,shots=receipt["shots"])
    receipt.update(status="submitted",job_id=job.job_id(),submitted_at=datetime.now(timezone.utc).isoformat())
    save_json(path,receipt)
    if provider=="iqm":
        start_iqm_guard(label)
    print(json.dumps({"provider":provider,"label":label,"job_id":receipt["job_id"],"status":"submitted"}),flush=True)


def retrieve(provider,label):
    directory=job_directory(provider,label)
    path=directory/"receipt.json"
    receipt=json.loads(path.read_text())
    if "job_id" not in receipt:
        raise ValueError("uncertain submission: use recover to identify and verify existing remote job; never resubmit")
    if receipt["status"]=="completed":
        print(json.dumps({"status":"completed","label":label,"charged_seconds":receipt.get("charged_seconds")}),flush=True)
        return
    backend,service=backend_for(provider)
    if provider=="ibm":
        job=service.job(receipt["job_id"])
        status=str(job.status()).lower()
    else:
        job=backend.retrieve_job(receipt["job_id"])
        status=str(job.status()).lower()
    receipt["remote_status"]=status
    save_json(path,receipt)
    print(json.dumps({"label":label,"remote_status":status}),flush=True)
    if "done" not in status and "completed" not in status:
        return
    result=job.result()
    if provider=="ibm":
        counts=[dict(pub.join_data().get_counts()) for pub in result]
        metric=job.metrics()
        save_json(directory/"metrics.json",metric)
        receipt["charged_seconds"]=float(metric["usage"]["quantum_seconds"])
    else:
        counts=[dict(c) for c in result.get_counts()]
        remote=backend.client.get_job(UUID(receipt["job_id"]))
        save_json(directory/"timeline.json",remote.data.model_dump(mode="json"))
        try:
            receipt.update(iqm_timeline_accounting(remote.data.model_dump(mode="json")))
        except ValueError:
            pass  # Retain the entire reservation when timing is incomplete.
    if len(counts)!=receipt["circuit_count"] or any(sum(c.values())!=receipt["shots"] for c in counts):
        raise ValueError("incomplete or wrong-shot hardware result")
    save_json(directory/"counts.json",counts)
    receipt.update(status="completed",completed_at=datetime.now(timezone.utc).isoformat())
    save_json(path,receipt)
    print(json.dumps({"label":label,"status":"completed","charged_seconds":receipt.get("charged_seconds"),"reserved_seconds":used_budget(provider)}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["prepare","submit","retrieve","recover","account","guard"])
    parser.add_argument("--provider",choices=["ibm","iqm"],required=True)
    parser.add_argument("--label",required=True)
    parser.add_argument("--shots",type=int,default=1024)
    parser.add_argument("--factor",type=int,default=1)
    parser.add_argument("--reservation",type=float,default=150)
    parser.add_argument("--states",nargs="+",choices=STATES,default=list(STATES))
    parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--rep-delay",type=float)
    parser.add_argument("--job-id")
    parser.add_argument("--evidence")
    args=parser.parse_args()
    if args.action=="prepare":
        print(prepare(args.provider,args.label,args.shots,args.factor,args.reservation,args.states,args.pilot,args.rep_delay))
    elif args.action=="submit":
        submit(args.provider,args.label)
    elif args.action=="recover":
        print(recover(args.provider,args.label,args.job_id))
    elif args.action=="account":
        if args.provider!="iqm" or not args.evidence:
            parser.error("account requires IQM and --evidence")
        record_iqm_accounting(args.label,args.evidence)
    elif args.action=="guard":
        if args.provider!="iqm":
            parser.error("guard is IQM only")
        guard_iqm(args.label)
    else:
        retrieve(args.provider,args.label)
