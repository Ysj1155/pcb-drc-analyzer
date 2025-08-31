# tools/calibrate_two_point.py
import json, argparse, math
from pathlib import Path
import cv2

HELP = """
[사용법]
  1) 이미지 창 뜨면 키보드 'r'을 눌러 ROI(관심영역) 선택 → Enter로 확정
  2) ROI가 표시되면 그 화면에서 두 지점을 클릭
  3) 창을 닫으면 mm/px가 계산되어 JSON으로 저장됩니다.

[단축키]
  r : ROI 선택 시작 (여러 번 가능)
  q : 바로 종료
"""

def pick_roi(img):
    # ROI 선택 (여러 번 시도 가능)
    while True:
        cv2.imshow("calib", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            try:
                x,y,w,h = cv2.selectROI("calib", img, showCrosshair=True, fromCenter=False)
                cv2.destroyWindow("ROI selector")
                if w>0 and h>0:
                    return img[int(y):int(y+h), int(x):int(x+w)]
            except Exception:
                pass
        elif key == ord('q'):
            return img  # ROI 건너뛰기
        else:
            # 안내 텍스트 보여주기
            print(HELP)

def main(img_path: Path, out_json: Path, real_mm: float):
    img = cv2.imread(str(img_path))
    if img is None: raise SystemExit("이미지 로드 실패")

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.imshow("calib", img)
    print(HELP)

    # 1) ROI 선택(선택 사항)
    roi = pick_roi(img)
    canvas = roi.copy()

    points = []
    def on_mouse(event, x, y, *_):
        nonlocal canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x,y))
            cv2.circle(canvas,(x,y),4,(0,255,0),-1)
            cv2.imshow("calib", canvas)

    cv2.setMouseCallback("calib", on_mouse)
    cv2.imshow("calib", canvas)
    cv2.waitKey(0); cv2.destroyAllWindows()

    if len(points) != 2:
        raise SystemExit("두 점을 정확히 클릭해야 합니다.")

    (x1,y1),(x2,y2) = points
    px_dist = math.hypot(x2-x1, y2-y1)
    mm_per_px = float(real_mm) / float(px_dist)

    out = {"mm_per_px": mm_per_px, "px_dist": px_dist, "real_mm": real_mm, "points": points}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--mm", type=float, required=True, help="두 점 사이 실제 거리(mm)")
    ap.add_argument("--out", type=Path, default=Path("calibration/mm_per_px.json"))
    args = ap.parse_args()
    main(args.image, args.out, args.mm)
