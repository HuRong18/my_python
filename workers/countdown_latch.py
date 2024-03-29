from threading import Condition

class CountDownLatch:

    def __init__(self,count):
        self.count = count
        self.condition = Condition()

    def wait(self):
        try:
            self.condition.acquire()
            while self.count > 0:
                self.condition.wait()
        finally:
            self.condition.release()

    def countDown(self):
        try:
            self.condition.acquire()
            self.count -= 1
            self.condition.notify_all()
        finally:
            self.condition.release()

    def getCount(self):
        return self.count
