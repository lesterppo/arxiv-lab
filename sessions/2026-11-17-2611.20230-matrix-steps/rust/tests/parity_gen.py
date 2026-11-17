"""Python side of the matrix-steps Python-vs-Rust parity sweep.

Mirrors the `matrix-steps parity <kind>` subcommands exactly (same JSON
keys, same draw order, same seed handling) so a parsed-JSON comparison
is exact for discrete outputs. Usage:

    python3 tests/parity_gen.py <kind> <args...>
    # kinds: rng <seed> <n> | embed <text> | sign <id> <width>
    #        matrix <seed> <n_procs> <passes> <beta>
    #        consolidate <seed> <n> | promotion <seed> <p0> <p1> <n>
    #        hidden <seed> | revisit <seed> <n> <repeats> | demotion <seed>

Not part of the Rust build; kept under tests/ as the parity oracle.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                "..", "src"))

from arxiv_lab.memory import matrix_steps as ms


def rng_kind(seed, n):
    r = ms.Rng(seed)
    u32s = [r.uint32() for _ in range(n)]
    r = ms.Rng(seed)
    uniform = [r.uniform() for _ in range(n)]
    r = ms.Rng(seed)
    randint6 = [r.randint(0, 5) for _ in range(n)]
    r = ms.Rng(seed)
    bern = [r.bernoulli(0.5) for _ in range(n)]
    print(json.dumps({"kind": "rng", "seed": seed, "u32": u32s,
                      "uniform": uniform, "randint6": randint6,
                      "bernoulli": bern}))


def embed_kind(text):
    e = ms.HashEmbedder()
    v = e.embed(text)
    import numpy as np
    print(json.dumps({"kind": "embed", "text": text,
                      "vec": [float(x) for x in v],
                      "norm": float(np.linalg.norm(v))}))


def sign_kind(pid, width):
    print(json.dumps({"kind": "sign", "id": pid, "width": width,
                      "signs": ms.procedure_sign_vector(pid, width)}))


def matrix_kind(seed, n_procs, passes, beta):
    import numpy as np
    mem = ms.EpisodicMemory("parity", rebuild_beta=beta)
    procs = ms.generate_synthetic_procedures(n_procs, seed=seed)
    for p in procs:
        mem.add_record(p)
    pairs = mem.write_pairs()
    mem.matrix.rebuild(pairs, passes=passes, beta=beta)
    found, nxt = [], []
    for p in procs:
        rec, _ = mem.find(p.goal, method="matrix", min_cosine=0.0)
        found.append(rec.id if rec else "")
        for i in range(len(p.steps) - 1):
            best, _ = mem.next_step_via_matrix(p.id, i)
            nxt.append(best.text() if best else "")
    checksum = float(np.sum(mem.matrix.M.astype(np.float64)))
    print(json.dumps({"kind": "matrix", "seed": seed, "n_procs": n_procs,
                      "passes": passes, "beta": beta, "found": found,
                      "next": nxt, "matrix_checksum": checksum,
                      "matrix_bytes": mem.matrix.M.size * 4}))


def consolidate_kind(seed, n):
    rng = ms.Rng(seed)
    seq = ("deflate", "lift tyre", "fit tube")
    other = ("deflate", "lift tyre", "fit tube", "extra step")
    eps = []
    for i in range(n):
        success = rng.bernoulli(0.7)
        steps = seq if rng.bernoulli(0.8) else other
        e = ms.Episode("e%d" % i, "alice", "tyre", "change tyre", steps,
                       "success" if success else "failure",
                       "external_check" if rng.bernoulli(0.9)
                       else "model_claim",
                       args={"torque": "40"}, condition="dry",
                       failure_mode=None if success else "slipped")
        eps.append(e)
    rec = ms.consolidate_episodes(eps)
    if rec is None:
        print(json.dumps({"kind": "consolidate", "seed": seed, "n": n,
                          "record": None}))
    else:
        print(json.dumps({"kind": "consolidate", "seed": seed, "n": n,
                          "record": {"confidence": rec["confidence"],
                                     "n_episodes": rec["n_episodes"],
                                     "n_successes": rec["n_successes"],
                                     "top_share": rec["top_sequence_share"],
                                     "steps": rec["steps"],
                                     "episode_ids": rec["episode_ids"]}}))


def promotion_kind(seed, p0, p1, n_windows):
    res = ms.run_promotion_windows(seed, p0, p1, n_windows=n_windows)
    print(json.dumps({"kind": "promotion", "seed": seed, "p_default": p0,
                      "p_candidate": p1, "n_windows": n_windows,
                      "promoted": res["promoted"],
                      "naive_kept": res["naive_kept"]}))


def hidden_kind(seed):
    r = ms.simulate_hidden_criterion(seed=seed)
    def counts(d):
        return {"reported": round(d["reported_done"] * 200),
                "solved": round(d["solved"] * 200),
                "attempts": round(d["attempts_per_task"] * 200),
                "damaging": d["damaging"]}
    print(json.dumps({"kind": "hidden", "seed": seed,
                      "no_memory": counts(r["no_memory"]),
                      "self_judged": counts(r["self_judged_stops"]),
                      "verified": counts(r["verified"]),
                      "ceiling": r["ceiling"]}))


def revisit_kind(seed, n, repeats):
    r = ms.simulate_revisit_stream(seed=seed, n_situations=n,
                                   repeats=repeats)
    print(json.dumps({"kind": "revisit", "seed": seed, "n": n,
                      "repeats": repeats,
                      "solved_situations": r["solved_situations"],
                      "revisits": r["revisits"],
                      "revisit_first": round(
                          r["revisit_first_attempt"] * r["revisits"]),
                      "first_encounter_attempts":
                          r["first_encounter_attempts"]}))


def demotion_kind(seed):
    r = ms.simulate_demotion(seed=seed)
    print(json.dumps({"kind": "demotion", "seed": seed,
                      "solved_without": round(
                          r["solved_without_rule"] * 200),
                      "solved_with": round(r["solved_with_rule"] * 200),
                      "trips_without": r["trips_without_rule"],
                      "trips_with": r["trips_with_rule"],
                      "demotions": r["demotions"],
                      "first_trip": r["first_trip_task"]}))


KINDS = {
    "rng": lambda a: rng_kind(int(a[0]), int(a[1])),
    "embed": lambda a: embed_kind(a[0]),
    "sign": lambda a: sign_kind(a[0], int(a[1])),
    "matrix": lambda a: matrix_kind(int(a[0]), int(a[1]), int(a[2]),
                                    float(a[3])),
    "consolidate": lambda a: consolidate_kind(int(a[0]), int(a[1])),
    "promotion": lambda a: promotion_kind(int(a[0]), float(a[1]),
                                          float(a[2]), int(a[3])),
    "hidden": lambda a: hidden_kind(int(a[0])),
    "revisit": lambda a: revisit_kind(int(a[0]), int(a[1]), int(a[2])),
    "demotion": lambda a: demotion_kind(int(a[0])),
}


if __name__ == "__main__":
    KINDS[sys.argv[1]](sys.argv[2:])
