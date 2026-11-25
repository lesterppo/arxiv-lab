#!/usr/bin/env python3
"""Replicate the Rust soak (tests/cape_tests.rs soak_many_small_worlds)
in Python, world for world, to check whether the 0.9 optimal-rate
threshold is a property of CAPE itself or of the Rust port.

Replicates Rust's xorshift64star Rng and make_world's partial
Fisher-Yates sampling exactly (64-bit wrapping arithmetic), then runs
compile_policy(cap=2, budget=200) + exact_optimum on each world.
Prints the optimal rate; also dumps per-world (value == opt) to a file
for comparison.
"""
import itertools, sys

sys.path.insert(0, "/home/hatch/workspace/arxiv-lab-build/src")
from arxiv_lab.agent.cape import (
    DEFAULT_ASPECTS, World, compile_policy, exact_optimum,
)

MASK = (1 << 64) - 1

class Rng:
    def __init__(self, seed):
        self.s = 0x9E3779B97F4A7C15 if seed == 0 else seed
    def next_u64(self):
        x = self.s
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        x &= MASK
        self.s = x
        return (x * 0x2545F4914F6CDD1D) & MASK
    def below(self, n):
        return self.next_u64() % n

def make_world_rs(n_configs, seed):
    aspects = DEFAULT_ASPECTS
    prod = list(itertools.product(*[a.values for a in aspects]))
    rng = Rng((seed + 0xC0FFEE) & MASK)
    if n_configs >= len(prod):
        configs = list(prod)
    else:
        for i in range(n_configs):
            j = i + rng.below(len(prod) - i)
            prod[i], prod[j] = prod[j], prod[i]
        configs = sorted(prod[:n_configs])
    probes = []
    for pi in range(len(aspects)):
        for ri in range(len(aspects)):
            if ri == pi:
                continue
            for v in aspects[pi].values:
                probes.append((pi, v, ri))
    return World(aspects, configs, list(itertools.product(*[a.values for a in aspects])), probes)

def main():
    rng = Rng(777)
    n = 500
    n_opt = 0
    misses = []
    for k in range(n):
        ncfg = 6 + rng.below(5)
        seed = rng.next_u64()
        w = make_world_rs(ncfg, seed)
        r = compile_policy(w, cap=2, budget=200)
        opt, _ = exact_optimum(w, cap=2)
        if r.value == opt:
            n_opt += 1
        else:
            misses.append((k, ncfg, seed, r.value, opt, r.queries, r.certified))
        if (k + 1) % 50 == 0:
            print("done %d/500, optimal so far %d" % (k + 1, n_opt), flush=True)
    print("PYTHON soak: optimal %d/%d = %.3f" % (n_opt, n, n_opt / n))
    print("misses: %d" % len(misses))
    for m in misses[:20]:
        print("  world %d ncfg=%d seed=%d py=%s opt=%s q=%d cert=%s" % m)

if __name__ == "__main__":
    main()
