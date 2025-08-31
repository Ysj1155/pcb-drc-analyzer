import numpy as np
import cv2
from skimage.morphology import skeletonize

def make_skeleton(binimg):
    """
    binimg: 전경(트레이스) 흰색(255) 이진영상
    return: uint8 스켈레톤(0/255)
    """
    sk = skeletonize((binimg > 0).astype(np.uint8)).astype(np.uint8) * 255
    return sk

def extract_polyline_points(skel, max_points=10000):
    """
    간단한 방법: 스켈레톤의 흰 픽셀 좌표를 (y,x) 리스트로 반환
    (정교한 그래프 추출은 다음 단계에서 개선)
    """
    ys, xs = np.where(skel > 0)
    pts = np.stack([xs, ys], axis=1)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts)-1, max_points).astype(int)
        pts = pts[idx]
    return pts  # shape (N, 2) with columns (x, y)
