import cv2
import numpy as np
from pathlib import Path

def make_overlay(bgr, binimg, min_w_px, min_gap_px, drc_dict, out_path="results/overlay.jpg"):
    """
    간단 오버레이:
    - binimg 외곽선(흰색)을 원본에 얹고
    - 좌상단에 측정치/DRC 텍스트 박스
    """
    vis = bgr.copy()
    if len(bgr.shape) == 2:
        vis = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    # 전경 윤곽선 -> 녹색
    cnts, _ = cv2.findContours((binimg>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0,255,0), 1)

    # 텍스트 패널
    panel = np.full((110, 360, 3), (32,32,32), np.uint8)
    lines = [
        f"min line width: {min_w_px:.3f}px",
        f"min spacing  : {min_gap_px:.3f}px",
    ]
    if drc_dict.get("scale_mm_per_px") is not None:
        lines += [
            f"scale       : {drc_dict['scale_mm_per_px']} mm/px",
            f"min width   : {drc_dict['min_line_width_mm']} mm",
            f"min spacing : {drc_dict['min_spacing_mm']} mm",
        ]
    status = drc_dict.get("pass")
    if status is True:  status_text, color = "DRC: PASS", (0,200,0)
    elif status is False: status_text, color = "DRC: FAIL", (0,0,255)
    else: status_text, color = "DRC: N/A", (200,200,0)
    lines.append(status_text)

    y = 22
    for t in lines:
        cv2.putText(panel, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240,240,240), 1, cv2.LINE_AA)
        y += 22
    cv2.rectangle(panel, (0,0), (panel.shape[1]-1, panel.shape[0]-1), color, 2)

    # 좌상단에 붙이기
    H, W = vis.shape[:2]
    ph, pw = panel.shape[:2]
    ph = min(ph, H); pw = min(pw, W)
    vis[0:ph, 0:pw] = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_AREA)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)
    return out_path
