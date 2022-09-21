import time
# import multiprocessing
from threading import Thread
import logging
from multiprocessing import Queue
import sys
logger = logging.getLogger(__name__)

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

class CustomThread(Thread):
    def __init__(self, group=None, target=None, name=None,
                    args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._result = None
        self._exec_duration = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            start = time.time()
            self._result = self._target(*self._args, **self._kwargs)
            end = time.time()
            self._exec_duration = round(end - start, 2)

    def result(self):
        return self._result, self._exec_duration
        
class fedcomponent:
    @parametrized
    def timeit(func, timeout):
        def wrapper(*args, **kwargs):
            start = time.time()
            print(f"Started: ({str(func.__name__)})")
            p = CustomThread(target=func, name=func.__name__, args=args, kwargs=kwargs)
            p.start()
            p.join(timeout)
            duration = str(time.time() - start) if p.result()[1] is None else p.result()[1]

            print(f"({str(p.name)}) Execution took : {duration} second(s)")

            if p.is_alive():
                raise TimeoutError(f"Timed out and Terminated: ({str(p.name)})")
            
            return p.result()[0]

        return wrapper


@fedcomponent.timeit(timeout=5)
def some_serializer(c, a, b=-2):
    time.sleep(a + b + c)
    k = 50
    pl = 80
    asd = k + pl
    print("In Serializer")
    return f"slept for c {round(a + b + c, 2)} seconds"

@fedcomponent.timeit(timeout=5)
def some_deserializer(c, a, b=-2):
    time.sleep(a + b + c)
    print("In De-Serializer")
    return "Return from Deserializer"

print(some_serializer(3, 3, b=-2))
print(some_deserializer(2, 1, b=-2))