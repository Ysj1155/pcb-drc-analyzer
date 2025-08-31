from pathlib import Path
import cv2, numpy as np, json, math

def draw_lines(img, widths_px, start=(50,50), gap=40):
    x0,y0 = start
    gt = []
    for w in widths_px:
        cv2.rectangle(img, (x0, y0), (x0+300, y0+int(w)), 255, -1)
        gt.append({"type":"line", "width_px": float(w)})
        y0 += int(w)+gap
    return gt

def draw_gaps(img, gaps_px, start=(400,50), bar_w=40, gap=40):
    x0,y0 = start
    gt=[]
    for g in gaps_px:
        cv2.rectangle(img,(x0,y0),(x0+bar_w,y0+200),255,-1)
        cv2.rectangle(img,(x0+bar_w+int(g),y0),(x0+2*bar_w+int(g),y0+200),255,-1)
        gt.append({"type":"gap","gap_px": float(g)})
        y0 += 200+gap
    return gt

def draw_arc(img, radius_px, center, ang=(0,180)):
    c = center; r = int(radius_px)
    cv2.ellipse(img, c, (r,r), 0, ang[0], ang[1], 255, 4)

def main(out_dir=Path("data/patterns")):
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas = np.zeros((1200,1600), np.uint8)

    gt=[]
    gt += draw_lines(canvas, widths_px=[6,10,15,20,30])
    gt += draw_gaps(canvas,  gaps_px=[6,10,15,20,30])

    # 곡률 반경 ground-truth (R ≈ 60, 100, 150px)
    draw_arc(canvas, 60,  (1100, 200))
    draw_arc(canvas, 100, (1100, 500))
    draw_arc(canvas, 150, (1100, 850))
    gt += [{"type":"radius","radius_px":v} for v in [60,100,150]]

    out_img = out_dir/"synthetic_pattern.png"
    cv2.imwrite(str(out_img), canvas)
    (out_dir/"synthetic_gt.json").write_text(json.dumps(gt,indent=2),encoding="utf-8")
    print(f"saved: {out_img}")

if __name__ == "__main__":
    main()
