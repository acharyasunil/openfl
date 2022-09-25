import os
import sys
import time
from threading import Thread
import logging
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
        
class federation:
    @parameterize
    def exec(func, timeout):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"Started: ({str(func.__name__)})")
            p = CustomThread(target=func, name=func.__name__, args=args, kwargs=kwargs)
            p.start()
            p.join(timeout)

            duration = round(time.time() - start, 3) if p.result()[1] is None else p.result()[1]

            print(f"({str(p.name)}) Execution took : {duration} second(s)")
            if p.is_alive():
                logger.info(f"Logger Terminating: {str(p.name)}")
                print(f"Timeout Terminating: {str(p.name)}", file=sys.stderr)
                os._exit(status=os.EX_TEMPFAIL)
            return p.result()[0]
        return wrapper