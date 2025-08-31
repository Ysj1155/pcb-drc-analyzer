# pcb/io_gpio/pigpio.py
from gpiozero import Button, LED, Buzzer
import time

class PiGpio:
    def __init__(self, btn_pin, led_pass, led_fail, buzz_pin):
        self.btn   = Button(btn_pin, pull_up=True, bounce_time=0.02)
        self.led_p = LED(led_pass)
        self.led_f = LED(led_fail)
        self.buzz  = Buzzer(buzz_pin)

    def wait_trigger(self, timeout=None):
        if timeout is None:
            self.btn.wait_for_press()
            return True
        else:
            start = time.time()
            while time.time()-start < timeout:
                if self.btn.is_pressed:
                    while self.btn.is_pressed: time.sleep(0.01)
                    return True
                time.sleep(0.01)
            return False

    def indicate_pass(self):
        self.led_p.on(); self.led_f.off()
        self.buzz.beep(on_time=0.05, off_time=0, n=1)

    def indicate_fail(self):
        self.led_p.off(); self.led_f.on()
        # 짧게 3번
        for _ in range(3):
            self.buzz.on(); time.sleep(0.05)
            self.buzz.off(); time.sleep(0.05)

    def idle(self):
        self.led_p.off(); self.led_f.off(); self.buzz.off()

    def close(self):
        self.idle()
