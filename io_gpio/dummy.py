# pcb/io_gpio/dummy.py
import time, sys

class DummyGpio:
    def wait_trigger(self, timeout=None):
        # timeout 동안 대기 후 자동 트리거 (테스트용)
        if timeout is None: timeout = 1.0
        time.sleep(min(timeout, 1.0))
        return True
    def indicate_pass(self):
        print("[GPIO] PASS")
    def indicate_fail(self):
        print("[GPIO] FAIL")
    def idle(self):
        print("[GPIO] IDLE")
    def close(self):
        sys.stdout.flush()
