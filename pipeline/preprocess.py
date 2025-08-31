import cv2
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def to_gray(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def clahe_gray(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def deskew_by_rotation(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    if lines is None:
        return gray
    angles = []
    for rho_theta in lines[:100]:
        rho, theta = rho_theta[0]
        angle = (theta * 180 / np.pi) - 90
        if angle > 45: angle -= 90
        if angle < -45: angle += 90
        angles.append(angle)
    if not angles:
        return gray
    rot = np.median(angles)
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def binarize(gray, method="otsu", bias=0):
    if method == "adaptive":
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=31, C=10 + int(bias)
        )
    else:
        ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th = int(np.clip(ret + int(bias), 0, 255))
        _, bin_img = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    white_ratio = (bin_img == 255).mean()
    if white_ratio < 0.5:
        bin_img = cv2.bitwise_not(bin_img)
    return bin_img

def ensure_foreground_white(binimg):
    return binimg if (binimg == 255).mean() >= 0.5 else cv2.bitwise_not(binimg)

def preprocess_and_binarize(bgr_img, use_deskew=True, bin_method="otsu", bias=0, save_prefix="binary"):
    gray = to_gray(bgr_img)
    if use_deskew:
        gray = deskew_by_rotation(gray)
    gray_eq = clahe_gray(gray)
    bin_img = binarize(gray_eq, method=bin_method, bias=bias)
    out_path = RESULTS_DIR / f"{save_prefix}.jpg"
    cv2.imwrite(str(out_path), bin_img)
    return bin_img, str(out_path)

def _order_points(pts: np.ndarray) -> np.ndarray:
    # 4점 (x, y) 정렬: tl, tr, br, bl
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # tl
    rect[2] = pts[np.argmax(s)]  # br
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # tr
    rect[3] = pts[np.argmax(diff)]  # bl
    return rect

def deskew_perspective(gray: np.ndarray) -> np.ndarray:
    """
    1) 가장 큰 외곽 컨투어에서 4각형 근사 → 투시 보정
    2) 실패 시 minAreaRect 각도로 회전 보정
    실패해도 입력을 그대로 반환(파이프라인이 죽지 않도록)
    """
    try:
        if gray is None:
            return gray
        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return gray

        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # (1) 투시 보정
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = _order_points(pts)
            (tl, tr, br, bl) = rect

            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)

            maxW = int(max(widthA, widthB))
            maxH = int(max(heightA, heightB))
            if maxW >= 10 and maxH >= 10:
                dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(gray, M, (maxW, maxH))
                return warped

        # (2) 회전 보정 (사각형 못 찾았을 때)
        rect = cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = gray.shape[:2]
        R = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(gray, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    except Exception:
        # 어떤 예외든 파이프라인 끊지 않기
        return gray
