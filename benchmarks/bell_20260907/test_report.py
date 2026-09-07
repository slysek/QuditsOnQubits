"""Reporting regression checks use synthetic fixtures in pytest temp directories."""
import json
import numpy as np
import report


def test_report_preserves_negative_gains_and_marks_missing_hardware(tmp_path,monkeypatch):
    root=tmp_path/"weighted"
    root.mkdir()
    monkeypatch.setattr(report,"ROOT",root)
    selection={s:{"selected":["sup023_P021_ph000","sup023_P021_ph012","sup023_P021_ph020"]} for s in report.STATES}
    (root/"selection.json").write_text(json.dumps(selection))
    (tmp_path/"bounds.json").write_text(json.dumps([
        {"state":s,"classical":5.6 if s!="ame43" else 7.6,"ideal":6 if s!="ame43" else 8}
        for s in report.STATES]))
    data={}
    for name,variant,value in [("canonical_ez","standard",5.0),("canonical_ez","optimal",5.2),
                                ("sup023_P021_ph000","standard",5.1),("sup023_P021_ph000","optimal",4.9)]:
        row={k:{"value":value,"se":.1,"ci95":[value-.2,value+.2]} for k in report.KEYS}
        row["invalid_fraction"]=.03
        draws=np.tile(np.linspace(value-.1,value+.1,20)[:,None],(1,4))
        data[("ibm","two_qutrit",name,variant,1)]=(row,draws)
    monkeypatch.setattr(report,"load_results",lambda:data)
    report.report()
    result=json.loads((root/"summary.json").read_text())
    assert result["complete"] is False
    assert len(result["missing"])==44
    delta=next(r for r in result["differences"] if r["name"]=="sup023_P021_ph000" and
               r["comparison"]=="candidate_opt_minus_baseline_opt" and r["estimator"]=="raw_conditional")
    assert abs(delta["delta"]+.3)<1e-10
    assert delta["relative_percent"]<0
    assert "NIEKOMPLETNY" in (root/"RAPORT.md").read_text(encoding="utf-8")
    assert (root/"bell_comparison.png").stat().st_size>1000
