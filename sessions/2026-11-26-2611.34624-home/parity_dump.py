#!/usr/bin/env python3
"""Generate tests/parity_cases.txt: worlds dumped from Python plus the
Python loop + oracle results the Rust port must match exactly.

One line per world (seed, n_cases=48):
  <seed> <n_cases> <case ...> <loop ...> <oracle ...>
Case field: e,a,qa,qs,h,t,g:fe,fa,fv,fs;fe,fa,fv,fs;...
  (e=entity idx, a=attribute idx, qa=query attribute idx, qs=1 if the
  query names the synonym, h=1 if historical, t=target session,
  g=gold value idx; each fact fe,fa,fv,fs)
Loop field: topk,recency_pct,rev(0=all,1=latest),fuzzy,answer(0=top1,
  1=majority),correct,n,tokens,formal,smoke,rejects,tried
Oracle field: topk,recency_pct,rev,fuzzy,answer,correct,n,tokens
Indices follow home.py's ENTITIES / ATTRIBUTES / VALUES order.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arxiv_lab.agent.home import (  # noqa: E402
    ENTITIES, ATTRIBUTES, VALUES, Harness, N_FORMAL, home_loop,
    make_world, oracle_best,
)

START = Harness(top_k=3, recency=0.0, revision="all", fuzzy=False,
                answer="majority")


def enc_harness(h):
    return ",".join([
        str(h.top_k), str(int(round(h.recency * 100))),
        "1" if h.revision == "latest" else "0",
        "1" if h.fuzzy else "0",
        "1" if h.answer == "majority" else "0",
    ])


def main():
    n_worlds = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out = Path(__file__).with_name("rust").joinpath("tests",
                                                    "parity_cases.txt")
    lines = []
    for seed in range(n_worlds):
        cases = make_world(seed=seed, n_cases=48)
        loop = home_loop(cases, START, seed=seed)
        oh, ores = oracle_best(cases)
        case_fields = []
        for c in cases:
            facts = ";".join(
                f"{ENTITIES.index(f.entity)},{ATTRIBUTES.index(f.attribute)},"
                f"{VALUES.index(f.value)},{f.session}" for f in c.history)
            case_fields.append(
                f"{ENTITIES.index(c.entity)},{ATTRIBUTES.index(c.attribute)},"
                f"{ATTRIBUTES.index(c.attribute)},"
                f"{1 if c.query_attribute != c.attribute else 0},"
                f"{1 if c.historical else 0},{c.target_session},"
                f"{VALUES.index(c.gold)}:{facts}")
        n = len(cases)
        loop_correct = round(loop.best_result.accuracy * n)
        oracle_correct = round(ores.accuracy * n)
        loop_f = (f"{enc_harness(loop.best)},{loop_correct},{n},"
                  f"{int(loop.best_result.mean_tokens)},"
                  f"{loop.formal_evals},{loop.smoke_evals},"
                  f"{loop.smoke_rejects},{loop.candidates_tried}")
        oracle_f = (f"{enc_harness(oh)},{oracle_correct},{n},"
                    f"{int(ores.mean_tokens)}")
        lines.append(" ".join([str(seed), str(n)] + case_fields
                              + [loop_f, oracle_f]))
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} worlds -> {out}")


if __name__ == "__main__":
    main()
