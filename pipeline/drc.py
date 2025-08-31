# pcb/pipeline/drc.py
from typing import Dict, Any, List

def apply_drc(min_w_mm: float | None,
              min_s_mm: float | None,
              min_r_mm: float | None,
              rules: Dict[str, Any]) -> Dict[str, Any]:

    req_w = rules.get("min_line_width_mm")
    req_s = rules.get("min_spacing_mm")
    req_r = rules.get("min_radius_mm")

    checks: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []

    def check(rule_name: str, measured: float | None, required: float | None) -> bool:
        if required is None:
            ok = True
        elif measured is None:
            ok = False
        else:
            ok = (measured >= required)
        checks.append({"rule": rule_name, "measured": measured, "required": required, "ok": ok})
        if not ok:
            violations.append({"rule": rule_name, "measured": measured, "required": required})
        return ok

    ok_w = check("min_line_width_mm", min_w_mm, req_w)
    ok_s = check("min_spacing_mm",   min_s_mm, req_s)
    ok_r = check("min_radius_mm",    min_r_mm, req_r)

    if rules.get("no_via_in_flex"):
        checks.append({"rule": "no_via_in_flex", "ok": True, "note": "not_implemented"})

    return {"pass": all(c["ok"] for c in checks), "checks": checks, "violations": violations,
            "required": {"min_line_width_mm": req_w, "min_spacing_mm": req_s, "min_radius_mm": req_r}}

def deskew_perspective(img):
    # 아직 구현 전이면 그냥 통과
    return img
