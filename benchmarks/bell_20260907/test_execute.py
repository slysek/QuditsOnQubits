import json
import math
import pytest
import execute


def test_provider_confirmed_iqm_batch_limit_is_checked_locally():
    execute.validate_batch_size("iqm",100)
    execute.validate_batch_size("ibm",274)
    with pytest.raises(ValueError,match="at most 100"):
        execute.validate_batch_size("iqm",101)


def test_budget_includes_completed_pending_and_uncertain_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    for i,row in enumerate([
        {"status":"completed","charged_seconds":30,"reservation_seconds":150},
        {"status":"submitted","reservation_seconds":150},
        {"status":"submitting","reservation_seconds":150},
    ]):
        directory=tmp_path/"ibm"/"jobs"/str(i)
        directory.mkdir(parents=True)
        (directory/"receipt.json").write_text(json.dumps(row))
    assert execute.used_budget("ibm")==330
    execute.check_budget("ibm",210)
    with pytest.raises(ValueError,match="budget"):
        execute.check_budget("ibm",211)


@pytest.mark.parametrize("value",[-1,0,float("inf"),float("nan")])
def test_invalid_reservation_rejected(value,tmp_path,monkeypatch):
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    with pytest.raises(ValueError):
        execute.check_budget("iqm",value)


@pytest.mark.parametrize("status",["submitting","submitted","completed"])
def test_no_resubmission_after_submission_starts(status,tmp_path,monkeypatch):
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    directory=tmp_path/"ibm"/"jobs"/"raw"
    directory.mkdir(parents=True)
    (directory/"receipt.json").write_text(json.dumps({"status":status}))
    monkeypatch.setattr(execute,"backend_for",lambda _:pytest.fail("must not contact provider"))
    with pytest.raises(ValueError,match="duplicate"):
        execute.submit("ibm","raw")


@pytest.mark.parametrize("label",["../other","A", "", "x/y"])
def test_unsafe_job_labels_rejected(label):
    with pytest.raises(ValueError):
        execute.job_directory("ibm",label)


def test_iqm_accounting_requires_completed_matching_job(tmp_path,monkeypatch):
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    d=tmp_path/"iqm/jobs/pilot"
    d.mkdir(parents=True)
    receipt={"status":"completed","job_id":"j1","reservation_seconds":45}
    (d/"receipt.json").write_text(json.dumps(receipt))
    evidence=tmp_path/"evidence.json"
    evidence.write_text(json.dumps({"job_id":"other","billed_seconds":2,"source":"provider dashboard","verified_by":"operator"}))
    with pytest.raises(ValueError,match="match"):
        execute.record_iqm_accounting("pilot",evidence)
    evidence.write_text(json.dumps({"job_id":"j1","billed_seconds":2,"source":"provider dashboard","verified_by":"operator"}))
    execute.record_iqm_accounting("pilot",evidence)
    assert execute.used_budget("iqm")==2
    assert json.loads((d/"receipt.json").read_text())["accounting_source"]=="billing_evidence.json"


def test_iqm_estimator_uses_measured_pilot_and_gate_duration(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from qiskit import QuantumCircuit
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    d=tmp_path/"iqm/jobs/pilot"
    d.mkdir(parents=True)
    receipt={"pilot":True,"status":"completed","job_id":"j1","charged_seconds":2,
             "shots":100,"circuit_count":20}
    (d/"receipt.json").write_text(json.dumps(receipt))
    qc=QuantumCircuit(1,1)
    qc.measure(0,0)
    backend=SimpleNamespace(target={"measure":{(0,):SimpleNamespace(duration=1e-6)}})
    seconds,model=execute.iqm_reservation([qc]*10,100,backend)
    assert seconds==pytest.approx(32.002)
    assert model["observed_seconds_per_shot"]==.001
    receipt["charged_seconds"]=4
    (d/"receipt.json").write_text(json.dumps(receipt))
    seconds,_=execute.iqm_reservation([qc]*10,100,backend)
    assert seconds==pytest.approx(34.002)


def test_recovery_fingerprint_distinguishes_wires_and_angles():
    from qiskit import QuantumCircuit
    a=QuantumCircuit(2,2)
    a.rz(.3,0)
    a.measure([0,1],[0,1])
    b=a.copy()
    b.x(1)
    assert execute.circuit_signature(a)==execute.circuit_signature(a.copy())
    assert execute.circuit_signature(a)!=execute.circuit_signature(b)


def test_iqm_guard_excludes_queue_and_cancels_with_margin():
    from datetime import datetime,timezone
    now=datetime(2026,9,7,3,0,0,tzinfo=timezone.utc)
    timeline=[{"status":"created","timestamp":"2026-09-06T00:00:00Z"}]
    assert execute.guard_decision("waiting",timeline,now,100)==(False,None,0.0)
    timeline.append({"status":"processing_started","timestamp":"2026-09-07T02:58:40Z"})
    cancel,start,elapsed=execute.guard_decision("processing",timeline,now,100)
    assert cancel and elapsed==80
    cancel,start,elapsed=execute.guard_decision("processing",[],now,100)
    assert not cancel and elapsed==10


def test_iqm_batches_are_bounded_before_contacting_provider(tmp_path,monkeypatch):
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    with pytest.raises(ValueError,match="split IQM"):
        execute.check_budget("iqm",241)


def test_iqm_timeline_accounts_processing_upper_bound_not_invoice():
    payload={"status":"completed","timeline":[
        {"source":"iqm-server","status":"created","timestamp":"2026-09-06T00:00:00Z"},
        {"source":"iqm-station-control","status":"received","timestamp":"2026-09-07T00:00:01Z"},
        {"source":"iqm-station-control","status":"execution_started","timestamp":"2026-09-07T00:00:02Z"},
        {"source":"iqm-station-control","status":"execution_ended","timestamp":"2026-09-07T00:00:03Z"},
        {"source":"iqm-server","status":"completed","timestamp":"2026-09-07T00:00:05Z"}]}
    result=execute.iqm_timeline_accounting(payload)
    assert result["accounted_seconds_upper_bound"]==4
    assert result["instrument_execution_wall_seconds"]==1
    assert "charged_seconds" not in result
    assert execute.budget_charge({**result,"reservation_seconds":50})==4
    payload["timeline"].pop()
    with pytest.raises(ValueError,match="missing"):
        execute.iqm_timeline_accounting(payload)


@pytest.mark.parametrize("remote_shots",[128,256])
def test_recover_ibm_matches_tag_payload_and_shots(tmp_path,monkeypatch,remote_shots):
    from types import SimpleNamespace
    from qiskit import QuantumCircuit,qpy
    monkeypatch.setattr(execute,"ROOT",tmp_path)
    d=tmp_path/"ibm/jobs/uncertain"
    d.mkdir(parents=True)
    qc=QuantumCircuit(1,1)
    qc.measure(0,0)
    with (d/"submitted.qpy").open("wb") as f:
        qpy.dump([qc],f)
    r={"status":"submitting","shots":128,"backend":"ibm_test","submission_token":"token",
       "payload_sha256":execute.hashlib.sha256((d/"submitted.qpy").read_bytes()).hexdigest()}
    (d/"receipt.json").write_text(json.dumps(r))
    job=SimpleNamespace(inputs={"pubs":[[qc,None,remote_shots]]},job_id=lambda:"remote-id",
                        backend=lambda:SimpleNamespace(name="ibm_test"))
    def jobs(**kwargs):
        assert kwargs["job_tags"]==["token"]
        return [job]
    monkeypatch.setattr(execute,"backend_for",lambda _: (None,SimpleNamespace(jobs=jobs)))
    if remote_shots==128:
        assert execute.recover("ibm","uncertain")=="remote-id"
        assert json.loads((d/"receipt.json").read_text())["status"]=="submitted"
    else:
        with pytest.raises(ValueError,match="shots mismatch"):
            execute.recover("ibm","uncertain")
        assert json.loads((d/"receipt.json").read_text())["status"]=="submitting"
