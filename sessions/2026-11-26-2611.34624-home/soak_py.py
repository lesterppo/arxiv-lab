#!/usr/bin/env python3
"""Soak cross-check (arXiv 2611.34624): replicate the Rust xorshift64
world generator call-for-call in Python and measure the Python loop's
oracle-reach rate on the SAME 500 worlds the Rust soak uses
(seeds 5000..5499, 48 cases each).

If the replica is exact, the rate must equal the Rust soak rate
(500/500). A mismatch would show up first as a world-generation
divergence -- verified separately by `dump_check.py` in this session
(see the Rust `dump_soak_world` test, seed 5000, compared field by
field below via --check).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arxiv_lab.agent.home import (  # noqa: E402
    ENTITIES, ATTRIBUTES, VALUES, Case, Fact, Harness, N_FORMAL,
    home_loop, oracle_best,
)

MASK = (1 << 64) - 1
MUL = 0x2545F4914F6CDD1D


class XorShift:
    def __init__(self, seed):
        self.s = 0x9E3779B97F4A7C15 if seed == 0 else seed

    def next_u64(self):
        x = self.s
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        x &= MASK
        self.s = x
        return (x * MUL) & MASK

    def below(self, n):
        return self.next_u64() % n

    def shuffle(self, xs):
        xs = list(xs)
        for i in range(len(xs) - 1, 0, -1):
            j = self.below(i + 1)
            xs[i], xs[j] = xs[j], xs[i]
        return xs

    def sample(self, xs, k):
        return self.shuffle(xs)[:k]


def make_world_rust(seed, n_cases=48):
    """Call-for-call replica of Rust home::make_world."""
    rng = XorShift(seed)
    cases = []
    for _ in range(n_cases):
        entity = ENTITIES[rng.below(8)]
        attr = ATTRIBUTES[rng.below(5)]
        n_rev = [1, 1, 2][rng.below(3)]
        sessions = sorted(rng.sample([0, 1, 2, 3], n_rev + 1))
        vals = rng.sample(list(range(12)), n_rev + 1)
        facts = [Fact(entity, attr, VALUES[v], s)
                 for s, v in zip(sessions, vals)]
        n_dist = 1 + rng.below(3)
        for _ in range(n_dist):
            others = [e for e in ENTITIES if e != entity]
            other = others[rng.below(7)]
            facts.append(Fact(other, attr, VALUES[rng.below(12)],
                              rng.below(4)))
        other_attrs = [a for a in ATTRIBUTES if a != attr]
        other_attr = other_attrs[rng.below(4)]
        facts.append(Fact(entity, other_attr, VALUES[rng.below(12)],
                          rng.below(4)))
        historical = rng.below(100) < 25
        if historical:
            ti = rng.below(len(sessions))
            target, gold = sessions[ti], VALUES[vals[ti]]
        else:
            target, gold = 3, VALUES[vals[-1]]
        query_synonym = rng.below(100) < 30
        query_attr = ({"office": "workplace", "phone": "telephone",
                       "project": "assignment", "manager": "supervisor",
                       "city": "town"}[attr] if query_synonym else attr)
        cases.append(Case(entity, attr, query_attr, historical, target,
                          gold, tuple(facts)))
    return cases


START = Harness(top_k=3, recency=0.0, revision="all", fuzzy=False,
                answer="majority")


def main():
    check_only = "--check" in sys.argv
    n = 500
    reached = 0
    total_formal = 0
    for s in range(n):
        cases = make_world_rust(5000 + s)
        lp = home_loop(cases, START, seed=s)
        _, ores = oracle_best(cases)
        if abs(lp.best_result.accuracy - ores.accuracy) < 1e-12:
            reached += 1
        total_formal += lp.formal_evals
    print(f"soak_py: loop reaches oracle accuracy in {reached}/{n} "
          f"worlds ({reached / n:.3f}), mean formal evals "
          f"{total_formal / n:.2f}")
    if check_only:
        # Field-by-field check of the seed-5000 world against the Rust
        # dump (first line of rust_dump.txt, written by the Rust
        # dump_soak_world test).
        dump = Path(__file__).with_name("rust_dump.txt").read_text().strip()
        py_cases = make_world_rust(5000)
        parts = dump.split(" ")
        assert parts[0] == "5000" and parts[1] == "48", parts[:2]
        ok = True
        for f, c in zip(parts[2:2 + 48], py_cases):
            head, facts_s = f.split(":")
            hp = head.split(",")
            if not (ENTITIES[int(hp[0])] == c.entity
                    and ATTRIBUTES[int(hp[1])] == c.attribute
                    and int(hp[3]) == (1 if c.query_attribute != c.attribute else 0)
                    and int(hp[4]) == (1 if c.historical else 0)
                    and int(hp[5]) == c.target_session
                    and VALUES[int(hp[6])] == c.gold):
                ok = False
                print("HEAD MISMATCH", f, c)
            py_facts = ";".join(
                f"{ENTITIES.index(fa.entity)},{ATTRIBUTES.index(fa.attribute)},"
                f"{VALUES.index(fa.value)},{fa.session}" for fa in c.history)
            if facts_s != py_facts:
                ok = False
                print("FACTS MISMATCH", facts_s, "vs", py_facts)
        print("world-replica check:", "EXACT MATCH" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
