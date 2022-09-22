import time
# import multiprocessing
from threading import Thread
import logging
from multiprocessing import Queue
import sys
logger = logging.getLogger(__name__)

def parameterize(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                    args=(), kwargs={}):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._result = None
        self._exec_duration = None

    def run(self):
        # logger.info(type(self._target))
        if self._target is not None:
            start = time.time()
            self._result = self._target(*self._args, **self._kwargs)
            self._exec_duration = round(time.time() - start, 2)

    def result(self):
        return self._result, self._exec_duration
        
class fed:
    @parameterize
    def exec_control(func, timeout):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"Started: ({str(func.__name__)})")
            p = CustomThread(target=func, name=func.__name__, args=args, kwargs=kwargs)
            p.start()
            p.join(timeout)

            duration = round(time.time() - start, 3) if p.result()[1] is None else p.result()[1]

            print(f"({str(p.name)}) Execution took : {duration} second(s)")
            if p.is_alive():
                raise TimeoutError(f"Timed out and Terminated: ({str(p.name)})")
            
            return p.result()[0]

        return wrapper


# @execution.control(timeout=5)
# def some_serializer(c, a, b=-2):
#     time.sleep(a + b + c)
#     k = 50
#     pl = 80
#     asd = k + pl
#     print("In Serializer")
#     return f"slept for c {round(a + b + c, 2)} seconds"

# @execution.control(timeout=5)
# def some_deserializer(c, a, b=-2):
#     time.sleep(a + b + c)
#     print("In De-Serializer")
#     return "Return from Deserializer"

# print(some_serializer(3, 3, b=-2))
# print(some_deserializer(2, 1, b=-2))