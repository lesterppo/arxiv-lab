"""
Simplified MCMC data-reshaping for arXiv:2610.02140
"Finetuning with Sampling: SFT Learns Better Than You Think".

The paper introduces an MCMC sampler that progressively transforms
off-policy expert traces into traces that are more on-policy for a
reference model, then runs plain SFT on the reshaped data.

Simplified version here (pure stdlib, no model needed):
  * For each prompt we have a small candidate POOL of traces:
    pool[0] is the off-policy expert trace (correct by construction),
    the rest are reference-model samples.
  * The caller supplies per-candidate scores: mean token log-probability
    under the reference model, and a correctness flag (our verifier,
    standing in for the paper's task verifier).
  * A Metropolis chain targets  pi(x) ~ exp(logp(x)/T) * 1[correct(x)]
    i.e. correct traces weighted toward high reference-model likelihood.
    Proposal: uniform over the pool. Acceptance: min(1, pi(y)/pi(x)).
  * The chain's final state is the reshaped trace for that prompt.

Honest simplification vs the paper: the real algorithm runs the chain
over full trace space with local edit proposals; we run it over a
discrete candidate pool (expert + reference samples). The essential
mechanism — Metropolis acceptance steering data toward the reference
distribution subject to a correctness gate — is preserved.
"""
import math
import random


def metropolis_reshape(logps, correct, steps=30, temperature=1.0, rng=None,
                       start=0):
    """Run one Metropolis chain over a candidate pool.

    logps   : list[float], mean token logp under the reference model
    correct : list[bool], verifier outcome per candidate
    steps   : number of MCMC steps
    temperature : T in pi(x) ~ exp(logp(x)/T); T=1 targets p_ref itself
    start   : index of the initial state (default 0 = the expert trace)

    Returns (final_index, stats dict).
    """
    rng = rng or random.Random()
    n = len(logps)
    assert n == len(correct) and n > 0
    assert any(correct), "pool must contain at least one correct trace"

    def log_target(i):
        if not correct[i]:
            return float("-inf")
        return logps[i] / temperature

    cur = start
    if not correct[cur]:  # never start on an incorrect trace
        cur = next(i for i, c in enumerate(correct) if c)
    accepts = 0
    for _ in range(steps):
        prop = rng.randrange(n)
        if prop == cur:
            continue
        log_alpha = log_target(prop) - log_target(cur)
        if log_alpha >= 0 or rng.random() < math.exp(log_alpha):
            cur = prop
            accepts += 1
    return cur, {"accept_rate": accepts / steps, "steps": steps,
                 "temperature": temperature, "start": start}


def reshape_dataset(pools, steps=30, temperature=1.0, seed=0):
    """Apply metropolis_reshape to every prompt's pool.

    pools: list of dicts with keys 'traces', 'logps', 'correct'.
    Returns (chosen_indices, aggregate stats).
    """
    rng = random.Random(seed)
    chosen, acc_rates, moved = [], [], 0
    for pool in pools:
        idx, st = metropolis_reshape(pool["logps"], pool["correct"],
                                     steps=steps, temperature=temperature,
                                     rng=rng, start=0)
        chosen.append(idx)
        acc_rates.append(st["accept_rate"])
        if idx != 0:
            moved += 1
    return chosen, {
        "n": len(pools),
        "moved_off_expert": moved,
        "moved_fraction": moved / len(pools),
        "mean_accept_rate": sum(acc_rates) / len(acc_rates),
    }


if __name__ == "__main__":
    # ---- synthetic sanity checks (no model needed) ----
    rng = random.Random(0)

    # 1) chain prefers higher-logp correct traces, rejects wrong ones
    logps = [-8.0, -2.0, -1.0, -0.5]   # 0 = off-policy expert
    correct = [True, True, True, False]
    idx, st = metropolis_reshape(logps, correct, steps=200, temperature=1.0,
                                 rng=random.Random(1))
    assert correct[idx], "chain ended on an incorrect trace"
    print(f"test1: final={idx} logp={logps[idx]} accept={st['accept_rate']:.2f}")
    assert idx in (1, 2), f"expected chain near best correct, got {idx}"

    # 2) wrong high-logp candidate is never accepted
    logps = [-5.0, -0.1]
    correct = [True, False]
    idx, _ = metropolis_reshape(logps, correct, steps=200,
                                rng=random.Random(2))
    assert idx == 0, f"chain accepted an incorrect trace: {idx}"
    print("test2: incorrect high-logp candidate correctly rejected")

    # 3) low temperature -> greedy toward best; high temperature -> wanders
    logps = [-4.0, -1.0, -0.2]
    correct = [True, True, True]
    counts = {0: 0, 1: 0, 2: 0}
    for s in range(60):
        i, _ = metropolis_reshape(logps, correct, steps=40, temperature=0.05,
                                  rng=random.Random(100 + s))
        counts[i] += 1
    print(f"test3 low-T visits: {counts}")
    assert counts[2] > 40, "low-T chain should concentrate on best trace"

    # 4) dataset-level: most prompts move off the expert trace
    pools = [{"logps": [-8.0, -1.5, -2.5], "correct": [True, True, True]}
             for _ in range(20)]
    chosen, agg = reshape_dataset(pools, steps=50, seed=0)
    print(f"test4: moved_fraction={agg['moved_fraction']:.2f}")
    assert agg["moved_fraction"] > 0.8

    # 5) no correct candidate except expert -> stays at expert (fallback)
    pools = [{"logps": [-8.0, -1.0], "correct": [True, False]}]
    chosen, agg = reshape_dataset(pools, steps=50, seed=0)
    assert chosen == [0], "should fall back to expert trace"
    print("test5: fallback to expert when nothing else is correct")

    print("mcmc OK")
