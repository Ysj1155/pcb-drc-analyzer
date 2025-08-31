from picamera2 import Picamera2
import cv2

class PiCamera:
    def __init__(self, size=(1920,1080)):
        self.cam = Picamera2()
        cfg = self.cam.create_still_configuration(main={"size": size}, buffer_count=1)
        self.cam.configure(cfg)
        # 노출/화이트밸런스 고정 예시 (현장에서 조정)
        self.cam.set_controls({"AwbMode": 0, "ExposureTime": 6000, "AnalogueGain": 1.0})
        self.cam.start()

    def capture(self):
        arr = self.cam.capture_array()
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def close(self):
        self.cam.stop()
