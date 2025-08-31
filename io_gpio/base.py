class Gpio:
    def wait_trigger(self, timeout=None):  # True면 트리거 감지
        raise NotImplementedError
    def indicate_pass(self):  # LED/부저 등
        pass
    def indicate_fail(self):
        pass
    def idle(self):
        pass
    def close(self):
        pass
