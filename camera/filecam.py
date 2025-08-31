import cv2
from pathlib import Path

class FileCamera:
    def __init__(self, device=0, fallback_path="data/pcb1.jpg"):
        self.cap = None
        try:
            self.cap = cv2.VideoCapture(device, cv2.CAP_DSHOW)
            ok, _ = self.cap.read()
            if not ok:
                self.cap.release(); self.cap=None
        except Exception:
            self.cap=None
        self.fallback = Path(fallback_path)

    def capture(self):
        if self.cap:
            ok, frame = self.cap.read()
            if ok: return frame
        # 파일 대체
        import cv2
        img = cv2.imread(str(self.fallback))
        if img is None:
            raise RuntimeError(f"fallback image not found: {self.fallback}")
        return img

    def close(self):
        if self.cap: self.cap.release()
