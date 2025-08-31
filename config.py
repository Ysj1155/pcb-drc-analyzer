TARGET = "PC"  # "PI" 로 바꾸면 라즈베리파이 드라이버 사용

# 카메라
CAMERA_RES = (1920, 1080)
CAMERA_USE_LIBCAMERA = True  # Pi에서 picamera2(libcamera) 사용

# GPIO 핀맵 (BCM 번호)
BTN_PIN  = 17
LED_PASS = 27
LED_FAIL = 22
BUZZ_PIN = 5

# 처리 파라미터(필요시)
BIN_METHOD = "otsu"
THRESH_BIAS = 0
