# cli/analyze_file.py
import json, argparse
from pathlib import Path
import numpy as np
import cv2
import math

from pipeline.preprocess import to_gray, clahe_gray, binarize, ensure_foreground_white, deskew_perspective
from pipeline.skeleton   import make_skeleton, extract_polyline_points
from pipeline.measure    import min_line_width_px, min_spacing_px
from pipeline.curvature  import min_curvature_radius_px
from pipeline.scale      import estimate_scale_mm_per_px
from pipeline.drc        import apply_drc
from pipeline.overlay    import make_overlay
from pipeline.report     import build_pdf_report

# ---------- 호환 래퍼들 ----------
def try_make_overlay(make_overlay_fn, img, bin_img, poly, violations, out_path):
    try:
        return make_overlay_fn(img, bin_img, poly, violations=violations, out_path=out_path)
    except TypeError:
        pass
    for attempt in (
        lambda: make_overlay_fn(img, bin_img, poly, out_path=out_path),
        lambda: make_overlay_fn(img, bin_img, poly, violations, out_path),
        lambda: make_overlay_fn(img, bin_img, out_path=out_path),
        lambda: make_overlay_fn(img, bin_img, poly),
    ):
        try:
            return attempt()
        except TypeError:
            pass
    # 폴백: 간단 오버레이 생성
    overlay = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 1)
    if poly is not None and len(poly) > 1:
        pts = np.array(poly, dtype=np.int32)
        cv2.polylines(overlay, [pts], False, (0, 0, 255), 1)
    cv2.imwrite(str(out_path), overlay)
    return out_path

def try_build_pdf_report(build_pdf_report_fn, res, rules, overlay_path, out_path):
    try:
        return build_pdf_report_fn(metrics=res, rules=rules, overlay_path=overlay_path, out_path=out_path)
    except TypeError:
        pass
    try:
        return build_pdf_report_fn(res, rules, overlay_path, out_path)
    except TypeError:
        pass
    for kws in [
        dict(result=res, rules=rules, overlay_path=overlay_path, out_path=out_path),
        dict(data=res,   rules=rules, overlay_path=overlay_path, out_path=out_path),
        dict(metrics=res, rules=rules, overlay=overlay_path, out_path=out_path),
        dict(metrics=res, rules=rules, overlay_path=overlay_path, pdf_path=out_path),
        dict(res=res, rules=rules, overlay_path=overlay_path, out_path=out_path),
    ]:
        try:
            return build_pdf_report_fn(**kws)
        except TypeError:
            continue
    return None

# ---------- 보조 유틸 ----------
def load_mm_per_px_from_json(path=Path("calibration/mm_per_px.json")) -> float:
    try:
        return float(json.loads(Path(path).read_text(encoding="utf-8")).get("mm_per_px") or 0.0)
    except Exception:
        return 0.0

def load_rules(path=Path("rules.json")) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    # 기본 규칙(없으면)
    return {"min_line_width_mm":0.15,"min_spacing_mm":0.15,"min_radius_mm":0.30,"no_via_in_flex":True}

def post_morph(bin_img, k=3, it=1):
    if k <= 1 or it <= 0: return bin_img
    kernel = np.ones((k,k), np.uint8)
    out = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=it)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=it)
    return out

def moving_average(x, w=5):
    if w <= 1: return x
    w = int(max(1, w | 1))  # 홀수로
    pad = w//2
    xp = np.pad(x, (pad,pad), mode="edge")
    c = np.convolve(xp, np.ones(w)/w, mode="valid")
    return c

def min_width_with_coord(bin_img, skel):
    # 전경(흰색) 거리변환 → 스켈레톤 위치에서 2*dist = local width
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return float("nan"), None
    vals = dist[ys, xs] * 2.0
    i = int(np.argmin(vals))
    return float(vals[i]), (int(xs[i]), int(ys[i]))

def min_spacing_with_coord(bin_img):
    # 배경(검정) 거리변환 → 배경 스켈레톤에서 2*dist = local spacing
    inv = cv2.bitwise_not(bin_img)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    skel_bg = make_skeleton(inv)
    ys, xs = np.where(skel_bg > 0)
    if len(xs) == 0:
        return float("nan"), None
    vals = dist[ys, xs] * 2.0
    i = int(np.argmin(vals))
    return float(vals[i]), (int(xs[i]), int(ys[i]))

def min_radius_with_coord(poly, smooth_w=5):
    if poly is None or len(poly) < 5:
        return float("nan"), None
    P = np.array(poly, dtype=np.float32)
    x = moving_average(P[:,0], smooth_w)
    y = moving_average(P[:,1], smooth_w)
    # 중앙차분
    dx  = (x[2:] - x[:-2]) * 0.5
    dy  = (y[2:] - y[:-2]) * 0.5
    ddx =  x[2:] - 2*x[1:-1] + x[:-2]
    ddy =  y[2:] - 2*y[1:-1] + y[:-2]
    denom = (dx*dx + dy*dy)**1.5 + 1e-6
    kappa = np.abs(dx*ddy - dy*ddx) / denom  # 곡률
    with np.errstate(divide="ignore", invalid="ignore"):
        R = 1.0 / kappa
    if not np.any(np.isfinite(R)):
        return float("nan"), None
    j = int(np.nanargmin(R))
    # j는 1..N-2 기준이므로 원 poly 좌표로 보정
    idx = j + 1
    pt = (int(P[idx,0]), int(P[idx,1]))
    return float(R[j]), pt

def annotate_overlay(path: Path, res: dict, pts: dict | None = None):
    img = cv2.imread(str(path))
    if img is None: return
    # 텍스트 박스
    txt = (f"mm/px={res['mm_per_px']:.5f} | "
           f"w={res['min_line_width_mm']:.3f}mm, "
           f"s={res['min_spacing_mm']:.3f}mm, "
           f"Rmin={res['min_radius_mm']:.3f}mm | PASS={res['pass']}")
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (10, 10), (20 + tw, 20 + th + 6), (0, 0, 0), -1)
    cv2.putText(img, txt, (15, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # 포인트/ROI/VIA 마킹
    if pts:
        if pts.get("roi_poly"):
            cv2.polylines(img, [np.array(pts["roi_poly"], dtype=np.int32)], True, (255, 255, 0), 1)
        if pts.get("w_px"):
            cv2.circle(img, pts["w_px"], 5, (0,0,255), 2)  # 빨강
            cv2.putText(img, "Wmin", (pts["w_px"][0]+6, pts["w_px"][1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        if pts.get("s_px"):
            cv2.circle(img, pts["s_px"], 5, (0,255,255), 2)  # 노랑
            cv2.putText(img, "Smin", (pts["s_px"][0]+6, pts["s_px"][1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if pts.get("r_px"):
            cv2.circle(img, pts["r_px"], 5, (255,0,0), 2)  # 파랑
            cv2.putText(img, "Rmin", (pts["r_px"][0]+6, pts["r_px"][1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
        if pts.get("vias"):
            for v in pts["vias"]:
                cv2.drawMarker(img, v, (0,255,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
                cv2.putText(img, "VIA", (v[0]+6, v[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    cv2.imwrite(str(path), img)

# ---------- 기하/ROI/VIA 도우미 ----------
def rdp(points, eps=1.5):
    if not points or len(points) < 3: return points
    P = np.asarray(points, dtype=float)
    def _rdp(s,e):
        if e <= s+1: return [s,e]
        A,B = P[s], P[e]
        AB = B - A; L2 = np.dot(AB,AB) + 1e-9
        dmax, idx = -1, s+1
        for i in range(s+1, e):
            AP = P[i] - A
            t = np.clip(np.dot(AP,AB)/L2, 0, 1)
            proj = A + t*AB
            d = np.linalg.norm(P[i]-proj)
            if d > dmax: dmax, idx = d, i
        if dmax > eps:
            left  = _rdp(s, idx)
            right = _rdp(idx, e)
            return left[:-1] + right
        else:
            return [s,e]
    idxs = _rdp(0, len(P)-1)
    return [tuple(map(int, P[i])) for i in idxs]

def load_flex_roi(path):
    p = Path(path)
    if not p.exists(): return None
    return json.loads(p.read_text(encoding="utf-8")).get("polygon")

def point_in_poly(pt, poly):
    # ray-casting
    x,y = pt; inside=False
    for i in range(len(poly)):
        x1,y1 = poly[i]; x2,y2 = poly[(i+1)%len(poly)]
        if ((y1>y) != (y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9) + x1):
            inside = not inside
    return inside

def detect_vias_in_roi(bin_img, poly, mm_per_px, dmm_min, dmm_max, circ_min=0.7):
    if poly is None: return []
    mask = np.zeros(bin_img.shape, np.uint8)
    cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    roi = cv2.bitwise_and(bin_img, mask)
    contours,_ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hits=[]
    for c in contours:
        A = cv2.contourArea(c)
        if A < 4: continue
        P = cv2.arcLength(c, True)
        circ = 4*math.pi*A/(P*P + 1e-9)
        if circ < circ_min: continue
        (x,y), r = cv2.minEnclosingCircle(c)
        d_mm = 2*r*mm_per_px
        if dmm_min <= d_mm <= dmm_max and point_in_poly((int(x),int(y)), poly):
            hits.append({"center_px": (int(x),int(y)), "diameter_mm": float(d_mm), "circularity": float(circ)})
    return hits

# ---------- 메인 분석 ----------
def analyze(in_path: Path, outdir: Path, mm_per_px: float|None,
            make_pdf: bool=True, morph_k=3, morph_it=1, rules_path: Path|None=None):
    outdir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"❌ 이미지 로드 실패: {in_path}")

    rules = load_rules(rules_path or Path("rules.json"))

    # 전처리
    gray = clahe_gray(to_gray(img))
    warped = deskew_perspective(gray)
    bin_img = ensure_foreground_white(binarize(warped))
    bin_img = post_morph(bin_img, k=morph_k, it=morph_it)

    # 스켈레톤 / 폴리라인
    skel = make_skeleton(bin_img)
    poly = extract_polyline_points(skel)
    if poly and len(poly) > 8:
        poly = rdp(poly, eps=2.0)                  # 모서리 단순화
        P = np.array(poly, dtype=np.float32)       # 간단 스무딩
        def _movavg(a, w=5):
            if w <= 1: return a
            pad = w // 2
            aa = np.pad(a, (pad, pad), mode="edge")
            return np.convolve(aa, np.ones(w)/w, mode="valid")
        poly = list(map(tuple, np.stack([_movavg(P[:,0],5), _movavg(P[:,1],5)], axis=1).astype(int)))

    # 스케일(mm/px)
    auto_scale = estimate_scale_mm_per_px(warped)
    calib = load_mm_per_px_from_json()
    s = (mm_per_px or auto_scale or calib or 0.06)  # 합리적 기본
    if not (0.01 <= s <= 0.3):
        print(f"⚠️ Suspicious mm_per_px={s:.5f} (check calibration/ROI).", flush=True)

    # 플렉스 ROI에서 VIA 검출
    flex_poly = load_flex_roi(rules.get("flex_roi_path", ""))
    via_hits = []
    if rules.get("no_via_in_flex") and flex_poly:
        via_hits = detect_vias_in_roi(
            bin_img, flex_poly, s,
            rules.get("via_diameter_mm_min", 0.10),
            rules.get("via_diameter_mm_max", 0.60),
            rules.get("via_circularity_min", 0.70)
        )

    # 기존 측정치(스칼라)
    w_px = float(min_line_width_px(bin_img))
    g_px = float(min_spacing_px(bin_img))
    r_px = float(min_curvature_radius_px(poly))

    # 좌표 포함 최소치(보조 계산)
    w_px_c, w_xy = min_width_with_coord(bin_img, skel)
    g_px_c, g_xy = min_spacing_with_coord(bin_img)
    r_px_c, r_xy = min_radius_with_coord(poly, smooth_w=5)

    # 좌표 계산 성공 시 그 값을 채택
    if np.isfinite(w_px_c): w_px = float(w_px_c)
    if np.isfinite(g_px_c): g_px = float(g_px_c)
    if np.isfinite(r_px_c): r_px = float(r_px_c)

    # 결과 dict
    res = {
        "mm_per_px": float(s),
        "min_line_width_px": w_px, "min_spacing_px": g_px, "min_radius_px": r_px,
        "min_line_width_mm": w_px*s, "min_spacing_mm": g_px*s, "min_radius_mm": r_px*s,
        "min_line_width_at_px": w_xy, "min_spacing_at_px": g_xy, "min_radius_at_px": r_xy,
        "min_line_width_at_mm": (tuple(np.array(w_xy)*s) if w_xy else None),
        "min_spacing_at_mm":   (tuple(np.array(g_xy)*s) if g_xy else None),
        "min_radius_at_mm":    (tuple(np.array(r_xy)*s) if r_xy else None),
    }

    # DRC 판정
    drc = apply_drc(res["min_line_width_mm"], res["min_spacing_mm"], res["min_radius_mm"], rules)

    # VIA 위반 반영
    if via_hits:
        res_violation = {"rule": "no_via_in_flex", "count": len(via_hits), "samples": via_hits[:5]}
        drc["violations"] = drc.get("violations", []) + [res_violation]
        drc["checks"] = drc.get("checks", []) + [{"rule":"no_via_in_flex","ok":False,"count":len(via_hits)}]
        drc["pass"] = False

    # 최종 DRC 결과를 res에 동기화
    res.update(drc)

    # 산출물 저장
    bin_path = outdir / "binary.jpg"; cv2.imwrite(str(bin_path), bin_img)
    ov_path  = outdir / "overlay.jpg"; try_make_overlay(make_overlay, img, bin_img, poly, res.get("violations", []), ov_path)

    # 마커+수치 각인(한 번만 호출)
    via_pts = [tuple(h["center_px"]) for h in via_hits]
    annotate_overlay(ov_path, res, {
        "roi_poly": flex_poly, "w_px": w_xy, "s_px": g_xy, "r_px": r_xy, "vias": via_pts
    })

    res.update({"binary_path": str(bin_path), "overlay_path": str(ov_path)})

    if make_pdf:
        pdf_path = outdir / "report.pdf"
        try_build_pdf_report(build_pdf_report, res, rules, ov_path, pdf_path)
        res["report_path"] = str(pdf_path)

    with open(outdir / "result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results_cli"))
    ap.add_argument("--mm-per-px", type=float, default=None)
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--morph-k", type=int, default=3, help="morph kernel size (odd)")
    ap.add_argument("--morph-it", type=int, default=1, help="morph iterations")
    ap.add_argument("--rules", type=Path, default=Path("rules.json"))
    args = ap.parse_args()
    analyze(args.image, args.out, args.mm_per_px, make_pdf=not args.no_pdf,
            morph_k=args.morph_k, morph_it=args.morph_it, rules_path=args.rules)
