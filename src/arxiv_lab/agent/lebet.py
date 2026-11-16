"""LeBet: learning interpretable rules for reliable agent intervention
(arXiv:2611.17662, "Learn then Bet: Learning Interpretable Rules for
Reliable Agent Intervention").

Paper claim: practitioners stop runaway agents with turn limits and
keyword filters, which stop runs that would succeed and let failing runs
continue. LeBet learns an interpretable OR-of-AND monitor over trajectory
features by stochastic local search (objective derived from sufficient
conditions for valid and powerful intervention), then accumulates evidence
through a fixed betting statistic and intervenes when the evidence
reaches a prescribed threshold. Under finite within-class exchangeability
and leave-one-out stability it gives finite-sample guarantees for
false-positive control (validity) and detection power conditional on the
training data, using the same trajectories for learning and calibration
(no iid assumption). On SWE-bench Lite with Ministral 3 8B the learned
monitor saves 34.46% of input+output tokens and raises output precision
from 18.33% to 21.59%, at an empirical false-positive rate of 13.64%.

This module implements the mechanism, not the paper's benchmark:
  * ``Trajectory`` -- a simulated agent run: a list of steps, each with
    an action, an outcome (ok/error), a state hash, an observation
    fingerprint, and tokens spent; plus a ground-truth label (SUCCESS
    = would eventually succeed if left alone, FAIL = will fail).
  * ``PREDICATES`` -- 10 boolean predicates over a trajectory prefix
    (repeats, error streaks, stalls, loops, length, token spend, ...).
  * ``Monitor`` -- an OR-of-ANDs over predicates; fires on a prefix iff
    any conjunction's literals are all true.
  * ``learn_monitor`` -- stochastic local search over rule edits
    (add/remove/replace literal, add/remove conjunction), maximising the
    betting score on training trajectories: betting power penalised by
    betting FPR and rule size, so the monitor is learned for the
    intervention rule it will actually drive.
  * ``BettingMonitor`` -- fixed betting statistic: a test martingale
    W_t = prod (1 + lam*(x_t - q)/(1-q)) where x_t is the monitor's
    per-step fire signal and q its baseline fire rate on successful
    training trajectories; intervene at the first t with W_t >= 1/alpha.
    By Ville's inequality P(ever cross | success) <= alpha, so the
    false-positive rate is controlled at alpha.
  * ``evaluate`` -- held-out FPR, detection power, token savings, and
    completed-run precision, against turn-limit and error-filter
    baselines, plus a fire-immediately ablation (monitor without the
    betting statistic) and a label-shuffled negative control.

RNG: SplitMix64, implemented here in pure Python so the Rust second stage
can reproduce the exact same streams from the same seeds (parity sweep).

Stdlib only. No network, no model API.
"""

import math

# ---------------------------------------------------------------- RNG
_MASK64 = (1 << 64) - 1


def splitmix64(state):
    """One SplitMix64 step; returns (value, new_state)."""
    state = (state + 0x9E3779B97F4A7C15) & _MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z = z ^ (z >> 31)
    return z, state


class Rng:
    """Deterministic RNG (SplitMix64); uniform() in [0, 1), randint, choice."""

    def __init__(self, seed):
        self.state = seed & _MASK64

    def next_u64(self):
        v, self.state = splitmix64(self.state)
        return v

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def randint(self, n):
        """Uniform int in [0, n)."""
        return self.next_u64() % n

    def choice(self, seq):
        return seq[self.randint(len(seq))]

    def gauss(self, mu=0.0, sigma=1.0):
        """Box-Muller normal."""
        u1 = max(self.uniform(), 1e-12)
        u2 = self.uniform()
        return mu + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------- trajectories
# A step: (action, ok, state_hash, obs_fp, tokens)
#   action: int tool id 0..5; ok: bool; state_hash: int; obs_fp: int; tokens: int
N_ACTIONS = 6


def gen_trajectory(rng, label):
    """Simulate one agent run. label True = SUCCESS (would finish), False = FAIL."""
    if label:  # success: mostly clean, but with a long tail and occasional noise
        n_steps = max(5, int(rng.gauss(16, 6)))
        p_repeat = 0.12
        p_err = 0.08
        p_stall = 0.06
    elif rng.uniform() < 0.6:
        # fail, subtype A (error-prone): long, repetitive, error-prone, stalls
        n_steps = max(8, int(rng.gauss(28, 5)))
        p_repeat = 0.38
        p_err = 0.30
        p_stall = 0.30
    else:
        # fail, subtype B (silent looper): fails by looping/stalling with
        # almost no errors -- no single error predicate can catch it, so a
        # learned OR-of-ANDs must cover both failure modes.
        n_steps = max(8, int(rng.gauss(30, 5)))
        p_repeat = 0.55
        p_err = 0.06
        p_stall = 0.45
    steps = []
    state = rng.randint(1 << 30)
    prev_action = rng.randint(N_ACTIONS)
    prev_obs = rng.randint(1 << 30)
    for _ in range(n_steps):
        if steps and rng.uniform() < p_repeat:
            action = prev_action
        else:
            action = rng.randint(N_ACTIONS)
        ok = rng.uniform() >= p_err
        if rng.uniform() < p_stall:
            pass  # state unchanged: a stall step
        else:
            state = (state * 2654435761 + 97) & ((1 << 30) - 1)
        if not ok and steps and steps[-1][1] is False and rng.uniform() < 0.5:
            obs = prev_obs  # repeated failed observation
        else:
            obs = rng.randint(1 << 30)
        tokens = 250 + rng.randint(150)
        steps.append((action, ok, state, obs, tokens))
        prev_action, prev_obs = action, obs
    return {"steps": steps, "label": label}


def gen_dataset(rng, n_success, n_fail):
    data = [gen_trajectory(rng, True) for _ in range(n_success)]
    data += [gen_trajectory(rng, False) for _ in range(n_fail)]
    # deterministic shuffle (Fisher-Yates with our RNG)
    for i in range(len(data) - 1, 0, -1):
        j = rng.randint(i + 1)
        data[i], data[j] = data[j], data[i]
    return data


# ---------------------------------------------------------------- predicates
# Each predicate maps a prefix (list of steps) to bool. Index order is the
# canonical predicate order used by monitors and the Rust port.
def _p_repeat3(prefix):
    return len(prefix) >= 3 and prefix[-1][0] == prefix[-2][0] == prefix[-3][0]


def _p_err2(prefix):
    return len(prefix) >= 2 and not prefix[-1][1] and not prefix[-2][1]


def _p_err_rate(prefix):
    w = prefix[-10:]
    return len(w) >= 5 and sum(1 for s in w if not s[1]) / len(w) > 0.3


def _p_stalled(prefix):
    return len(prefix) >= 3 and prefix[-1][2] == prefix[-2][2] == prefix[-3][2]


def _p_loop(prefix):
    seen = {}
    for s in prefix:
        seen[s[2]] = seen.get(s[2], 0) + 1
        if seen[s[2]] >= 3:
            return True
    return False


def _p_long(prefix):
    return len(prefix) > 20


def _p_repeat_tool_fail(prefix):
    if len(prefix) < 2 or prefix[-1][1]:
        return False
    a = prefix[-1][0]
    return sum(1 for s in prefix if s[0] == a and not s[1]) >= 2


def _p_no_new_obs(prefix):
    return len(prefix) >= 2 and prefix[-1][3] == prefix[-2][3]


def _p_tok_high(prefix):
    return sum(s[4] for s in prefix) > 6000


def _p_same_action_5(prefix):
    w = prefix[-8:]
    if len(w) < 5:
        return False
    a = prefix[-1][0]
    return sum(1 for s in w if s[0] == a) >= 5


PREDICATES = [
    ("repeat3", _p_repeat3),
    ("err2", _p_err2),
    ("err_rate", _p_err_rate),
    ("stalled", _p_stalled),
    ("loop", _p_loop),
    ("long", _p_long),
    ("repeat_tool_fail", _p_repeat_tool_fail),
    ("no_new_obs", _p_no_new_obs),
    ("tok_high", _p_tok_high),
    ("same_action_5", _p_same_action_5),
]
N_PRED = len(PREDICATES)


def prefix_matrix(traj):
    """Precompute the [n_steps x N_PRED] boolean matrix for a trajectory."""
    steps = traj["steps"]
    mat = []
    for t in range(1, len(steps) + 1):
        prefix = steps[:t]
        mat.append(tuple(fn(prefix) for _, fn in PREDICATES))
    return mat


# ---------------------------------------------------------------- monitor (OR of ANDs)
# A monitor is a tuple of conjunctions; a conjunction is a frozenset of
# predicate indices. Fires on a prefix-row iff any conjunction is a subset
# of the true predicates in that row.

def monitor_fires_row(monitor, row):
    for conj in monitor:
        if conj and all(row[i] for i in conj):
            return True
    return False


def monitor_fires_traj(monitor, mat):
    """First firing step (1-based) or None if it never fires."""
    for t, row in enumerate(mat):
        if monitor_fires_row(monitor, row):
            return t + 1
    return None


def monitor_tpr_fpr(monitor, mats):
    """mats: list of (matrix, label). Returns (tpr, fpr) at trajectory level."""
    tp = fp = n_pos = n_neg = 0
    for mat, label in mats:
        fired = monitor_fires_traj(monitor, mat) is not None
        if label:
            n_neg += 1
            fp += 1 if fired else 0
        else:
            n_pos += 1
            tp += 1 if fired else 0
    return (tp / n_pos if n_pos else 0.0, fp / n_neg if n_neg else 0.0)


def monitor_complexity(monitor):
    return sum(len(c) for c in monitor)


def monitor_score(monitor, mats, lam=2.0, complexity=0.005):
    tpr, fpr = monitor_tpr_fpr(monitor, mats)
    return tpr - lam * fpr - complexity * monitor_complexity(monitor)


def _random_conjunction(rng, max_lits=3):
    k = 1 + rng.randint(max_lits)
    lits = set()
    while len(lits) < k:
        lits.add(rng.randint(N_PRED))
    return frozenset(lits)


def _random_monitor(rng, max_conj=3, max_lits=3):
    k = 1 + rng.randint(max_conj)
    return tuple(_random_conjunction(rng, max_lits) for _ in range(k))


def _mutate(rng, monitor, max_conj=3, max_lits=3):
    """One stochastic-local-search edit, exploiting the Boolean structure."""
    monitor = [set(c) for c in monitor]
    move = rng.randint(6)
    if move == 0 and monitor:  # add a literal to a random conjunction
        ci = rng.randint(len(monitor))
        if len(monitor[ci]) < max_lits:
            monitor[ci].add(rng.randint(N_PRED))
    elif move == 1 and monitor:  # remove a literal
        ci = rng.randint(len(monitor))
        if len(monitor[ci]) > 1:
            monitor[ci].discard(rng.choice(sorted(monitor[ci])))
        else:
            monitor.pop(ci)
    elif move == 2 and len(monitor) < max_conj:  # add a conjunction
        monitor.append(set(_random_conjunction(rng, max_lits)))
    elif move == 3 and len(monitor) > 1:  # remove a conjunction
        monitor.pop(rng.randint(len(monitor)))
    elif move == 4 and monitor:  # replace a literal
        ci = rng.randint(len(monitor))
        if monitor[ci]:
            monitor[ci].discard(rng.choice(sorted(monitor[ci])))
            monitor[ci].add(rng.randint(N_PRED))
    else:  # fresh random conjunction in place of one
        if monitor:
            monitor[rng.randint(len(monitor))] = set(_random_conjunction(rng, max_lits))
    out = tuple(frozenset(c) for c in monitor if c)
    return out if out else (frozenset([rng.randint(N_PRED)]),)


def learn_monitor(rng, mats, restarts=6, iters=200, alpha=0.15, bet_lam=0.75):
    """Stochastic local search for the OR-of-AND monitor.

    The objective is the betting score on the training trajectories:
    betting power penalised by betting FPR and rule size -- i.e. the
    monitor is learned for the intervention rule it will actually drive,
    not for trajectory-level firing alone.
    """
    best, best_score = None, -1e18
    for _ in range(restarts):
        cur = _random_monitor(rng)
        cur_score = monitor_betting_score(cur, mats, alpha, lam=bet_lam)
        for _ in range(iters):
            cand = _mutate(rng, cur)
            s = monitor_betting_score(cand, mats, alpha, lam=bet_lam)
            if s > cur_score or (s == cur_score and rng.uniform() < 0.1):
                cur, cur_score = cand, s
        if cur_score > best_score:
            best, best_score = cur, cur_score
    return best, best_score


# ---------------------------------------------------------------- betting statistic
def baseline_fire_rate(monitor, mats):
    """Per-step fire rate on SUCCESS trajectories (the H0 rate q)."""
    fires = steps = 0
    for mat, label in mats:
        if not label:
            continue
        for row in mat:
            steps += 1
            if monitor_fires_row(monitor, row):
                fires += 1
    return fires / steps if steps else 0.0


def betting_wealth_series(monitor, mat, q, lam=0.5):
    """Wealth W_t under the fixed betting rule; returns the series."""
    w = 1.0
    series = []
    for row in mat:
        x = 1.0 if monitor_fires_row(monitor, row) else 0.0
        w *= (1.0 + lam * (x - q) / (1.0 - q))
        series.append(w)
    return series


def betting_intervene_step(monitor, mat, q, alpha, lam=0.5):
    """First step with wealth >= 1/alpha, or None."""
    threshold = 1.0 / alpha
    w = 1.0
    for t, row in enumerate(mat):
        x = 1.0 if monitor_fires_row(monitor, row) else 0.0
        w *= (1.0 + lam * (x - q) / (1.0 - q))
        if w >= threshold:
            return t + 1
    return None


def monitor_betting_eval(monitor, mats, alpha, lam=0.75):
    """Betting power/FPR of a monitor on a set of (matrix, label) pairs,
    with q calibrated on the successes of the same set (the paper uses the
    same trajectories for learning and calibration)."""
    q = baseline_fire_rate(monitor, mats)
    fp = tp = n_pos = n_neg = 0
    for mat, label in mats:
        step = betting_intervene_step(monitor, mat, q, alpha, lam)
        if label:
            n_neg += 1
            fp += 1 if step is not None else 0
        else:
            n_pos += 1
            tp += 1 if step is not None else 0
    power = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    return power, fpr, q


def monitor_betting_score(monitor, mats, alpha, lam=0.75, fpr_weight=2.0,
                          complexity=0.005):
    """SLS objective: betting power, penalised by betting FPR and rule size.
    A monitor only scores well if it fires early and repeatedly on failing
    runs (so wealth accumulates) and rarely on successful ones."""
    power, fpr, _ = monitor_betting_eval(monitor, mats, alpha, lam)
    return power - fpr_weight * fpr - complexity * monitor_complexity(monitor)


# ---------------------------------------------------------------- evaluation
def traj_tokens(traj):
    return sum(s[4] for s in traj["steps"])


def tune_turn_limit(train_mats, alpha):
    """Smallest step cap K whose training FPR (success runs longer than K)
    is <= alpha; the most aggressive valid turn limit. None = never stop."""
    succ_lens = sorted(len(m) for m, lab in train_mats if lab)
    if not succ_lens:
        return None
    n = len(succ_lens)
    for k in range(5, 61):
        fpr = sum(1 for L in succ_lens if L > k) / n
        if fpr <= alpha:
            return k
    return None


def tune_error_filter(train_mats, alpha):
    """Best single-predicate immediate-stop rule on train, subject to
    training FPR <= alpha. Candidates: err2, err_rate, repeat_tool_fail."""
    best = None
    for idx in (1, 2, 6):
        tp = fp = n_pos = n_neg = 0
        for mat, label in train_mats:
            fired = any(row[idx] for row in mat)
            if label:
                n_neg += 1
                fp += 1 if fired else 0
            else:
                n_pos += 1
                tp += 1 if fired else 0
        tpr = tp / n_pos if n_pos else 0.0
        fpr = fp / n_neg if n_neg else 0.0
        if fpr <= alpha and (best is None or tpr > best[1]):
            best = (idx, tpr, fpr)
    return best  # (predicate index, train TPR, train FPR) or None


def evaluate(monitor, q, alpha, train_data, test_data, lam=0.5):
    """Held-out evaluation of the betting monitor.

    Baselines are TUNED ON TRAIN to the same validity budget (FPR <= alpha),
    so the comparison is at matched validity -- the paper's framing
    (valid first, then powerful). Returns FPR, power, token savings and
    completed-run precision for the betting monitor, the fire-immediately
    ablation, and the tuned turn-limit / error-filter baselines.
    """
    train_mats = [(prefix_matrix(t), t["label"]) for t in train_data]
    test_mats = [(prefix_matrix(t), t["label"]) for t in test_data]
    k_star = tune_turn_limit(train_mats, alpha)
    err_star = tune_error_filter(train_mats, alpha)
    n_succ = sum(1 for t in test_data if t["label"])
    n_fail = len(test_data) - n_succ
    total_tokens = sum(traj_tokens(t) for t in test_data)

    def summarise(intervene_fn):
        fp = tp = 0
        spent = 0
        completed_ok = completed = 0
        intervene_steps_fail = []
        for traj, (mat, label) in zip(test_data, test_mats):
            step = intervene_fn(mat)
            if step is None:
                spent += traj_tokens(traj)
                completed += 1
                completed_ok += 1 if label else 0
            else:
                spent += sum(s[4] for s in traj["steps"][:step])
                if label:
                    fp += 1
                else:
                    tp += 1
                    intervene_steps_fail.append(step)
        return {
            "fpr": fp / n_succ if n_succ else 0.0,
            "power": tp / n_fail if n_fail else 0.0,
            "token_savings": 1.0 - spent / total_tokens if total_tokens else 0.0,
            "precision_completed": completed_ok / completed if completed else 0.0,
            "median_intervene_step_fail": (
                sorted(intervene_steps_fail)[len(intervene_steps_fail) // 2]
                if intervene_steps_fail else None),
            "n_intervened_fail": tp,
            "n_false_positive": fp,
        }

    betting = summarise(lambda mat: betting_intervene_step(monitor, mat, q, alpha, lam))
    immediate = summarise(lambda mat: monitor_fires_traj(monitor, mat))
    if k_star is None:
        turn_limit = summarise(lambda mat: None)
    else:
        turn_limit = summarise(lambda mat: k_star if len(mat) > k_star else None)
    turn_limit["k_star"] = k_star
    if err_star is None:
        err_filter = summarise(lambda mat: None)
        err_filter["predicate"] = None
    else:
        idx = err_star[0]
        err_filter = summarise(
            lambda mat, idx=idx: next((t + 1 for t, row in enumerate(mat)
                                       if row[idx]), None))
        err_filter["predicate"] = PREDICATES[idx][0]
        err_filter["train_tpr"] = err_star[1]
        err_filter["train_fpr"] = err_star[2]
    base_precision = n_succ / len(test_data) if test_data else 0.0
    return {
        "alpha": alpha, "q": q, "lam": lam,
        "n_success": n_succ, "n_fail": n_fail,
        "base_precision_no_monitor": base_precision,
        "betting": betting,
        "immediate": immediate,
        "turn_limit_tuned": turn_limit,
        "error_filter_tuned": err_filter,
    }


def monitor_to_text(monitor):
    parts = []
    for conj in monitor:
        lits = " AND ".join(sorted(PREDICATES[i][0] for i in conj))
        parts.append(f"({lits})")
    return " OR ".join(parts)


def run_experiment(seed=20261116, n_train_succ=200, n_train_fail=200,
                   n_test_succ=200, n_test_fail=200, alpha=0.15,
                   restarts=6, iters=200, lam=2.0, bet_lam=0.75):
    """Full LeBet experiment: generate, learn, calibrate, evaluate."""
    rng = Rng(seed)
    train = gen_dataset(rng, n_train_succ, n_train_fail)
    test = gen_dataset(rng, n_test_succ, n_test_fail)
    train_mats = [(prefix_matrix(t), t["label"]) for t in train]
    monitor, train_score = learn_monitor(rng, train_mats, restarts=restarts,
                                         iters=iters, alpha=alpha,
                                         bet_lam=bet_lam)
    train_tpr, train_fpr = monitor_tpr_fpr(monitor, train_mats)
    train_power, train_bet_fpr, _ = monitor_betting_eval(
        monitor, train_mats, alpha, lam=bet_lam)
    q = baseline_fire_rate(monitor, train_mats)
    result = evaluate(monitor, q, alpha, train, test, lam=bet_lam)
    # negative control: same learner on label-permuted training data
    rng2 = Rng(seed + 999)
    labels = [lab for _, lab in train_mats]
    for i in range(len(labels) - 1, 0, -1):
        j = rng2.randint(i + 1)
        labels[i], labels[j] = labels[j], labels[i]
    shuffled = [(m, lab) for (m, _), lab in zip(train_mats, labels)]
    monitor_nc, _ = learn_monitor(rng2, shuffled, restarts=3, iters=100,
                                  alpha=alpha, bet_lam=bet_lam)
    nc_tpr, nc_fpr = monitor_tpr_fpr(monitor_nc, shuffled)
    result["monitor"] = monitor_to_text(monitor)
    result["monitor_raw"] = [[PREDICATES[i][0] for i in sorted(c)] for c in monitor]
    result["train_tpr"] = train_tpr
    result["train_fpr"] = train_fpr
    result["train_score"] = train_score
    result["train_betting_power"] = train_power
    result["train_betting_fpr"] = train_bet_fpr
    result["negative_control"] = {
        "monitor": monitor_to_text(monitor_nc),
        "train_tpr_minus_fpr": nc_tpr - nc_fpr,
        "train_tpr": nc_tpr, "train_fpr": nc_fpr,
    }
    result["seed"] = seed
    return result
