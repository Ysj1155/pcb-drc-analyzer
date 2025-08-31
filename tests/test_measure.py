from pathlib import Path
import json, cv2
from pipeline.preprocess import to_gray, binarize
from pipeline.measure import min_line_width_px
from pipeline.curvature import min_curvature_radius_px

def test_width_on_synth():
    img = cv2.imread("data/patterns/synthetic_pattern.png", 0)
    bin_img = binarize(img)
    w = min_line_width_px(bin_img)
    assert w > 0

def test_radius_on_synth():
    gt = json.loads(Path("data/patterns/synthetic_gt.json").read_text())
    radii = [g["radius_px"] for g in gt if g["type"]=="radius"]
    assert min(radii) >= 50
