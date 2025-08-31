import cv2
import numpy as np
from .preprocess import ensure_foreground_white

def min_line_width_px(binimg):
    fg = ensure_foreground_white(binimg)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((fg>0).astype(np.uint8), 8)
    if num_labels > 1:
        keep = np.zeros_like(fg, np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        if len(areas) > 0:
            thresh = max(10, np.percentile(areas, 20))
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= thresh:
                    keep[labels == i] = 255
        if keep.sum() > 0:
            fg = keep
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
    dist = cv2.distanceTransform((fg>0).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dist[fg>0]
    if vals.size == 0:
        return 0.0
    return max(0.0, 2.0 * float(vals.min()))

def min_spacing_px(binimg):
    fg = ensure_foreground_white(binimg)
    bg = cv2.bitwise_not(fg)
    dist = cv2.distanceTransform((bg>0).astype(np.uint8), cv2.DIST_L2, 3)
    dx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, 3)
    dy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, 3)
    grad = cv2.magnitude(dx, dy)
    ridge = (grad < 0.2) & (dist > 0.5)
    vals = dist[ridge]
    if vals.size == 0:
        nz = dist[dist > 0]
        if nz.size == 0:
            return 0.0
        return float(2.0 * np.percentile(nz, 5))
    return float(2.0 * np.min(vals))
