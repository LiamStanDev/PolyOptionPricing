import time


def timeit(f):
    start = time.perf_counter_ns()
    result = f()
    end = time.perf_counter_ns()
    return result, (end - start) / (1000 * 1000)  # convert to millisecond
