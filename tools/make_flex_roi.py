import json, argparse
from pathlib import Path
import cv2, numpy as np

def main(img_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None: raise SystemExit("이미지 로드 실패")
    disp = img.copy()
    pts = []

    def on_mouse(e,x,y,flags,param):
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x),int(y)])
            cv2.circle(disp,(x,y),4,(0,255,0),-1)
            if len(pts)>1: cv2.line(disp,tuple(pts[-2]),(x,y),(0,255,0),2)
            cv2.imshow("ROI", disp)

    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI", disp)
    cv2.setMouseCallback("ROI", on_mouse)
    print("좌클릭으로 점 추가, Enter로 저장, ESC로 취소.")
    while True:
        k = cv2.waitKey(10) & 0xFF
        if k in (13,10):  # Enter
            break
        if k == 27:       # ESC
            cv2.destroyAllWindows()
            return
    cv2.destroyAllWindows()

    if len(pts) < 3: raise SystemExit("최소 3점 필요")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"polygon": pts}, indent=2), encoding="utf-8")
    print("saved:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("flex_roi.json"))
    args = ap.parse_args()
    main(args.image, args.out)
