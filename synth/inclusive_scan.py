import random


# Treat generators as threads to be executed in parallel.
def run_parallel(thread, thread_count, handler, *args):
    indices = list(range(thread_count))
    threads = [thread(i, thread_count, *args) for i in range(thread_count)]
    alive = [True for _ in range(thread_count)]

    while any(alive):
        handler(*args)

        # Simulate non-deterministic thread execution order.
        random.shuffle(indices)

        for i in indices:
            if alive[i]:
                try:
                    next(threads[i])
                except StopIteration:
                    alive[i] = False


# A thread that performs inclusive scan with raking.
def scan_thread(thread_x, thread_count, mem):
    step = 1
    idx = thread_x * 2 + 1
    while step < len(mem):
        if idx < len(mem):
            mem[idx] += mem[idx - step]
            idx = idx * 2 + 1
        step *= 2
        # Threads synchronize with each other by calling yield.
        yield

    step = len(mem) // 4
    idx = thread_x * (2 * step) + (3 * step) - 1
    while step > 0:
        if idx < len(mem):
            mem[idx] += mem[idx - step]
        idx //= 2
        step //= 2
        yield


def test_scan(mem):
    run_parallel(scan_thread, len(mem), print, mem)


if __name__ == "__main__":
    test_scan([2 ** i for i in range(8)])
