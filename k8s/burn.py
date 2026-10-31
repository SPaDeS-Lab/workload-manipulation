"""CPU/memory burner used as the Job payload.

Fixed *useful work* = `--work` CPU-seconds in total, split over `--workers` processes, so a Job
with more workers (a higher CPU limit) finishes sooner but does the same work.
Prints one line:  RESULT {json}  with precise epoch start/end and measured CPU seconds.
"""
import argparse, json, multiprocessing as mp, os, time


def worker(share, mem_mb, q):
    buf = bytearray(mem_mb * 1024 * 1024) if mem_mb else None
    if buf:
        for i in range(0, len(buf), 4096):
            buf[i] = 1
    x = 0
    c0 = time.process_time()
    while time.process_time() - c0 < share:
        for _ in range(20000):
            x = (x * 1103515245 + 12345) & 0xFFFFFFFF
    q.put(time.process_time() - c0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mem-mb", type=int, default=256)
    a = ap.parse_args()
    t0 = time.time()
    q = mp.Queue()
    ps = [mp.Process(target=worker, args=(a.work / a.workers, a.mem_mb // a.workers, q)) for _ in range(a.workers)]
    [p.start() for p in ps]
    cpu = sum(q.get() for _ in ps)
    [p.join() for p in ps]
    print("RESULT " + json.dumps(dict(start=t0, end=time.time(), cpu_seconds=cpu,
                                      workers=a.workers, node=os.environ.get("NODE_NAME", ""))), flush=True)
