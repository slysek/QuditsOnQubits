"""Check the discovery pipeline: ideation, one campaign per family, migration and the global best."""

import json

import pytest

from qudits_on_qubits.evolution import discovery
from qudits_on_qubits.evolution.oe_candidates import load_skeleton, skeleton_program
from qudits_on_qubits.evolution.oe_run import INVALID_BEST, MAX_NOTES_BYTES
from qudits_on_qubits.evolution.state_workloads import evaluate_workload
from test_evolution_oe_run import _baseline, _socketpair_works

HUB = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 5]]
CHAIN = [[q, q + 1] for q in range(5)]


def _answer(*items):
    return "Here are the families.\n```json\n" + json.dumps(list(items)) + "\n```\n"


def test_parse_keeps_valid_families_and_records_rejections():
    answer = _answer({"name": "Hub / fan-out!", "idea": "copy", "cz": HUB},
                     {"name": "chain", "idea": "too few CZ across cuts", "cz": CHAIN},
                     {"name": "bad", "idea": "x", "cz": [[0, 9], [1, 2]]},
                     "not a family",
                     {"name": "extra", "idea": "beyond the limit", "cz": HUB})
    accepted, rejected = discovery.parse_families(answer, "ghz3", 4)
    assert [f["name"] for f in accepted] == ["0_hub_fan_out"]
    assert accepted[0]["cz"] == HUB
    assert len(rejected) == 3
    assert "cut" in rejected[0]["reason"]


@pytest.mark.parametrize("answer", ["no json here", "```json\n{\"name\": 1}\n```", _answer({"name": "c", "idea": "x", "cz": CHAIN}),
                                    "[" * 100_000])
def test_parse_rejects_unusable_answers(answer):
    with pytest.raises(ValueError):
        discovery.parse_families(answer, "ghz3", 3)


@pytest.mark.parametrize("answer", ["", " \n"])
def test_parse_names_an_empty_answer(answer):
    with pytest.raises(ValueError, match="no text.*effort"):
        discovery.parse_families(answer, "ghz3", 3)


def test_offline_families_satisfy_cuts_or_are_rejected():
    accepted, rejected = discovery.select_families(discovery.offline_families("ghz3", [(0, 2), (1, 3)]), "ghz3", 3)
    assert [f["name"] for f in accepted] == ["0_hub"]
    assert accepted[0]["cz"] == HUB
    assert len(rejected) == 2


def test_notes_show_other_families_and_respect_the_byte_limit():
    family = {"title": "hub", "idea": "copy"}
    other = {"title": "source", "cz_count": 9, "two_qubit_depth": 7, "cz": HUB, "is_valid": True, "F": 1.0}
    near = {"title": "near", "cz_count": 4, "two_qubit_depth": 3, "cz": HUB[:4], "is_valid": False, "F": 0.61}
    notes = discovery.family_notes(family, [other, near], "user hint")
    assert "copy" in notes and "source: 9 CZ" in notes and notes.endswith("user hint")
    assert "two-qubit depth 7, valid:" in notes and "near: 4 CZ, two-qubit depth 3, INVALID with fidelity F=0.6100" in notes
    assert len(discovery.family_notes(family, [other], "ą" * 5000).encode()) <= MAX_NOTES_BYTES


def test_invalid_options_create_no_output_and_send_no_request(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "_ask_model", lambda *_: pytest.fail("no request expected"))
    model = tmp_path / "model.yaml"
    model.write_text("llm: {}\n")
    for options in ({"families": 0}, {"rounds": 11}, {"iterations": 0}, {"encoding_mode": "other"},
                    {"discovery_timeout_seconds": 0}, {"model_config": tmp_path / "missing.yaml"}):
        with pytest.raises(ValueError):
            discovery.run_discovery(tmp_path / "base", tmp_path / "out", **{"model_config": model, **options})
        assert not (tmp_path / "out").exists()


def _fake_campaign(log):
    def run(baseline, output, **kwargs):
        seed = load_skeleton(kwargs["initial_program"], expected_qubits=6)
        log.append({"output": output.name, "seed": [list(p) for p in seed],
                    "notes": kwargs["prompt_notes"].read_text(), "representation": kwargs["representation"],
                    "min_seed_fidelity": kwargs["min_seed_fidelity"]})
        if "1_bad" in output.name:
            raise RuntimeError("The best candidate did not pass the state check")
        output.mkdir()
        (output / "best_program.py").write_text(skeleton_program(6, seed[1:]))
        cz = len(seed) - 1
        return {"saved": {"is_valid": True, "cz_count": cz, "two_qubit_depth": cz, "depth": 2 * cz,
                          "one_qubit_gate_count": 3 * cz, "F": 1.0, "L": 0.0}}
    return run


def test_model_ideation_rounds_migration_and_global_best(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "model.yaml"
    model.write_text("llm: {}\n")
    seven = HUB + [[2, 4], [3, 5]]
    answer = _answer({"name": "hub", "idea": "fan out", "cz": seven},
                     {"name": "bad", "idea": "fails", "cz": HUB},
                     {"name": "copy", "idea": "copy", "cz": HUB + [[2, 4]]})
    prompts = []
    monkeypatch.setattr(discovery, "_ask_model", lambda _cfg, system, prompt: prompts.append(prompt) or answer)
    log = []
    monkeypatch.setattr(discovery, "run_campaign", _fake_campaign(log))
    report = discovery.run_discovery(baseline, tmp_path / "out", state="ghz3", method="canonical_direct",
                                     families=3, rounds=2, model_config=model, iterations=5)
    assert "at least 2 CZ across" in prompts[0] and "ghz3" in prompts[0]
    # Seeds stay near the known structure; a seed that does not fit is replaced in its campaign.
    assert "small change of it" in prompts[0] and "fewer CZ than" not in prompts[0]
    assert f"below {discovery.MIN_SEED_FIDELITY}" in prompts[0]
    assert all(entry["min_seed_fidelity"] == discovery.MIN_SEED_FIDELITY for entry in log)
    assert [r["status"] for r in report["runs"]] == ["completed", "failed", "completed"] * 2
    assert all(entry["representation"] == "skeleton" for entry in log)
    # Round 2 starts each family from its own best and shows the other families.
    round2_hub = next(e for e in log if e["output"] == "round_2_0_hub")
    assert round2_hub["seed"] == seven[1:]
    assert "copy: 5 CZ" in round2_hub["notes"]
    assert report["best"]["family"] == "2_copy" and report["best"]["cz_count"] == 4
    assert report["status"] == "completed"
    saved = json.loads((tmp_path / "out" / "discovery.json").read_text())
    assert saved["best"] == report["best"]
    assert json.loads((tmp_path / "out" / "ideation.json").read_text())["answer"] == answer


def test_blank_discovery_uses_only_new_family_seeds(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "model.yaml"
    model.write_text("llm: {}\n")
    prompts, options = [], []
    answer = _answer({"name": "new", "idea": "new structure", "cz": HUB})
    monkeypatch.setattr(discovery, "_ask_model", lambda _cfg, system, prompt: prompts.append(prompt) or answer)
    monkeypatch.setattr(discovery, "seed_records", lambda *args: pytest.fail("No record can seed this run"))

    def campaign(base, output, **kwargs):
        options.append(kwargs)
        output.mkdir()
        (output / "best_program.py").write_text(skeleton_program(6, HUB))
        return {"saved": {"is_valid": True, "cz_count": 5, "two_qubit_depth": 3,
                          "depth": 7, "one_qubit_gate_count": 16, "F": 1.0, "L": 0.0}}

    monkeypatch.setattr(discovery, "run_campaign", campaign)
    report = discovery.run_discovery(baseline, tmp_path / "out", state="ghz3", method="canonical_direct",
                                     model_config=model, families=1, rounds=2, iterations=2,
                                     blank_start=True, record_store=object())
    assert "known valid structure" not in prompts[0].lower()
    assert "0.5" not in prompts[0]
    assert all(item["blank_start"] and item["min_seed_fidelity"] == 0 for item in options)
    assert all(item["initial_program"] is not None for item in options)
    assert options[0]["warm_starts"] == [] and len(options[1]["warm_starts"]) == 1
    assert report["seed_record"] is None and report["blank_start"] and "records" not in report


def test_invalid_best_seeds_the_next_round_and_reaches_the_other_families(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "model.yaml"
    model.write_text("llm: {}\n")
    near, hub = HUB + [[2, 4]], HUB + [[3, 5]]
    answer = _answer({"name": "near", "idea": "x", "cz": near}, {"name": "hub", "idea": "y", "cz": hub})
    monkeypatch.setattr(discovery, "_ask_model", lambda *_: answer)
    log = []

    def run(baseline, output, **kwargs):
        seed = load_skeleton(kwargs["initial_program"], expected_qubits=6)
        log.append({"output": output.name, "seed": [list(p) for p in seed], "notes": kwargs["prompt_notes"].read_text()})
        output.mkdir()
        (output / "best_program.py").write_text(skeleton_program(6, seed[1:]))
        fidelity = 0.6 if output.name.startswith("round_1") else 0.4  # round 2 is worse: keep round 1
        (output / INVALID_BEST).write_text(json.dumps({
            "is_valid": False, "cz_count": len(seed) - 1, "two_qubit_depth": 3, "depth": 6,
            "one_qubit_gate_count": 9, "F": fidelity, "L": 0.1}))
        raise RuntimeError("The best candidate did not pass the state check")

    monkeypatch.setattr(discovery, "run_campaign", run)
    report = discovery.run_discovery(baseline, tmp_path / "out", state="ghz3", method="canonical_direct",
                                     families=2, rounds=2, model_config=model, iterations=5)
    assert [r["status"] for r in report["runs"]] == ["invalid"] * 4
    assert report["runs"][0]["F"] == 0.6 and "did not pass" in report["runs"][0]["error"]
    round2 = {e["output"]: e for e in log if e["output"].startswith("round_2")}
    assert round2["round_2_0_near"]["seed"] == near[1:]
    assert "INVALID with fidelity F=0.6000" in round2["round_2_0_near"]["notes"]
    assert json.dumps(hub[1:]) in round2["round_2_0_near"]["notes"]
    assert report["family_best"]["0_near"]["directory"] == "round_1_0_near"
    assert report["best"] is None and report["status"] == "failed"


def test_orchestrator_model_with_offline_loop(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    ideas = tmp_path / "ideas.yaml"
    ideas.write_text("llm: {}\n")
    asked = []
    monkeypatch.setattr(discovery, "_ask_model",
                        lambda cfg, *_: asked.append(cfg) or _answer({"name": "hub", "idea": "x", "cz": HUB}))
    log = []
    campaigns = []
    fake = _fake_campaign(log)
    monkeypatch.setattr(discovery, "run_campaign", lambda *a, **k: campaigns.append(k["model_config"]) or fake(*a, **k))
    report = discovery.run_discovery(baseline, tmp_path / "out", state="ghz3", method="canonical_direct",
                                     ideation_model_config=ideas, offline_model="mutations", iterations=3)
    assert asked == [ideas] and campaigns == [None]
    assert report["ideation_mode"] == "configured_model" and report["model_mode"] == "offline_mutations"
    assert json.loads((tmp_path / "out" / "ideation.json").read_text())["model_config"] == str(ideas.resolve())


def test_unusable_ideation_is_saved_and_raises(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "model.yaml"
    model.write_text("llm: {}\n")
    monkeypatch.setattr(discovery, "_ask_model", lambda *_: "I cannot help")
    with pytest.raises(ValueError, match="JSON"):
        discovery.run_discovery(baseline, tmp_path / "out", state="ghz3", method="canonical_direct",
                                model_config=model, iterations=2)
    assert json.loads((tmp_path / "out" / "ideation.json").read_text())["answer"] == "I cannot help"


def test_ideation_time_does_not_use_the_search_budget_after_resume(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import checkpoints
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "ideas.yaml"
    model.write_text("llm: {}\n")
    clock = [0]
    monkeypatch.setattr(discovery, "monotonic", lambda: clock[0])
    monkeypatch.setattr(checkpoints, "monotonic", lambda: clock[0])
    asked, budgets = [], []

    def ideation(*args):
        asked.append(True)
        clock[0] += 10000
        return _answer({"name": "hub", "idea": "copy", "cz": HUB})

    def campaign(base, output, **options):
        budgets.append(options["campaign_timeout_seconds"])
        if len(budgets) == 1:
            raise KeyboardInterrupt
        return _fake_campaign([])(base, output, **options)

    monkeypatch.setattr(discovery, "_ask_model", ideation)
    monkeypatch.setattr(discovery, "run_campaign", campaign)
    output = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        discovery.run_discovery(baseline, output, method="canonical_direct", families=1,
                                 ideation_model_config=model, discovery_timeout_seconds=5)
    clock[0] += 1000000
    assert checkpoints.resume_operation(output)["status"] == "completed"
    assert asked == [True]
    assert budgets == [5, 5]
    assert json.loads((output / "resume.json").read_text())["ideation_seconds"] == 10000


def test_resume_excludes_the_time_of_legacy_failed_ideation(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution import checkpoints
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    model = tmp_path / "ideas.yaml"
    model.write_text("llm: {}\n")
    clock, budgets = [0], []
    monkeypatch.setattr(discovery, "monotonic", lambda: clock[0])
    monkeypatch.setattr(checkpoints, "monotonic", lambda: clock[0])
    answers = iter(("", _answer({"name": "hub", "idea": "copy", "cz": HUB})))

    def ideation(*args):
        clock[0] += 10000
        return next(answers)

    def campaign(base, output, **options):
        budgets.append(options["campaign_timeout_seconds"])
        return _fake_campaign([])(base, output, **options)

    monkeypatch.setattr(discovery, "_ask_model", ideation)
    monkeypatch.setattr(discovery, "run_campaign", campaign)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="no text"):
        discovery.run_discovery(baseline, output, method="canonical_direct", families=1,
                                 ideation_model_config=model, discovery_timeout_seconds=5)
    document = json.loads((output / "resume.json").read_text())
    document.pop("ideation_seconds")
    checkpoints.save(output / "resume.json", document)
    assert checkpoints.resume_operation(output)["status"] == "completed"
    assert budgets == [5]
    assert json.loads((output / "resume.json").read_text())["ideation_seconds"] == 20000


@pytest.mark.skipif(not _socketpair_works(), reason="The sandbox blocks the asyncio socket pair")
def test_offline_discovery_with_openevolve(tmp_path):
    pytest.importorskip("openevolve")
    baseline = tmp_path / "baseline"
    _baseline(baseline)
    output = tmp_path / "out"
    report = discovery.run_discovery(baseline, output, state="ghz3", method="canonical_direct",
                                     families=3, rounds=2, timeout_seconds=60, cpu_seconds=60)
    assert report["families"] == ["0_source", "1_hub"]
    source = next(r for r in report["runs"] if r["directory"] == "round_1_0_source")
    assert source["status"] == "completed"
    assert "source:" in (output / "inputs" / "round_2_1_hub" / "notes.txt").read_text()
    best = report["best"]
    run = output / best["directory"]
    from qiskit import qpy
    import numpy as np
    with (run / "best.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    encodings = np.load(run / "best_encodings.npy", allow_pickle=False)
    measured = evaluate_workload("ghz3", circuit, encodings, tolerance=1e-10)
    assert measured["is_valid"] and measured["cz_count"] == best["cz_count"] <= source["cz_count"]
