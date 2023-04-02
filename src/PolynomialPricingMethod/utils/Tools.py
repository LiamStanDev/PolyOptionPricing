import time


def timeit(f):
    start = time.perf_counter()
    result = f()
    end = time.perf_counter()
    return result, end - start
