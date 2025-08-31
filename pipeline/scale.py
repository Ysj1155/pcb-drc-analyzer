import cv2
import numpy as np

def estimate_scale_mm_per_px(gray, pattern_size=(7,7), square_size_mm=1.0):
    """
    체커보드 내부 코너 (w,h) 개수. 예: 8x8 보드는 pattern_size=(7,7)
    square_size_mm: 한 칸의 실제 한 변(mm)
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    ok, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        return None

    corners = cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ).reshape(-1,2)

    w, h = pattern_size
    dists = []
    for r in range(h):
        row = corners[r*w:(r+1)*w]
        dists += list(np.linalg.norm(row[1:] - row[:-1], axis=1))
    for c in range(w):
        col = corners[c::w][:h]
        dists += list(np.linalg.norm(col[1:] - col[:-1], axis=1))

    if not dists: return None
    px_per_square = float(np.mean(dists))
    if px_per_square <= 0: return None
    return square_size_mm / px_per_square  # mm/px
