import numpy as np

def _circle_radius_from_three(p1, p2, p3):
    """
    세 점으로 원 반지름 계산(세 점이 일직선이면 np.inf).
    p*: (x, y)
    """
    (x1,y1),(x2,y2),(x3,y3) = p1,p2,p3
    a = np.hypot(x2-x1, y2-y1)
    b = np.hypot(x3-x2, y3-y2)
    c = np.hypot(x1-x3, y1-y3)
    s = (a+b+c)/2.0
    area = max(s*(s-a)*(s-b)*(s-c), 0.0)
    if area == 0:
        return np.inf
    R = (a*b*c)/(4.0*np.sqrt(area))
    return R

def min_curvature_radius_px(poly_pts, win=9, step=2):
    """
    폴리라인(점열)에서 길이 win의 구간마다 세 점(양끝+중앙)으로 반지름 근사.
    - 너무 노이즈 많은 경우를 줄이기 위해 step 간격 샘플링
    """
    n = len(poly_pts)
    if n < win:
        return np.inf
    # 균일간격 샘플링
    pts = poly_pts[::max(1, step)]
    m = len(pts)
    if m < win:
        return np.inf

    half = win//2
    radii = []
    for i in range(half, m-half):
        p1 = tuple(pts[i-half])
        p2 = tuple(pts[i])
        p3 = tuple(pts[i+half])
        R = _circle_radius_from_three(p1,p2,p3)
        if np.isfinite(R) and R > 0:
            radii.append(R)
    if not radii:
        return np.inf
    return float(np.min(radii))
