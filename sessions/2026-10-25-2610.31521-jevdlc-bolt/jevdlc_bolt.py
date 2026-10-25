#!/usr/bin/env python3
"""Jev-DLC (arXiv 2610.31521) tested on Nemotron 3.5 Bolt E4B, decision level.

Paper claim (abstract + §4-6, read from the PDF): typed corrections become
compact causal state above a FROZEN scorer. Two bounded channels:
  - conservative: base logits + Dirichlet-smoothed count prior over the
    semantic identity of the preferred action (Eq. 5-6, alpha=0.5, weight 2.0)
  - responsive: base logits + a correction vector updated by minimum-change
    PA-I repair on the directed (preferred, rejected) contrast (Eq. 7-9,
    gamma=C=1), with no count prior
Combined by geometric mixing with reliability weights updated by zero-one
loss after each revealed feedback (Eq. 10-11, eta=0.25, sigma=0.1).
No raw history, no extra scorer call, no backward pass. Baselines in the
paper: frozen scorer (39.44% on TCPA-Repeat) and Recent-4 history replay
(42.40%), Jev-DLC 44.83% overall and 56.07% at feedback depth >= 2.

RECONSTRUCTION, stated honestly: the paper's scorer is JEV-SFT (option
tokens scored at the first output position) with a learned 32-dim
projection of hidden states, which Ollama does not expose. What we CAN
reproduce faithfully on Bolt:
  - frozen scorer = Bolt's first-token logprobs for the candidate letters
    (verified: /api/chat with logprobs returns clean single-token A-D
    logprobs; /api/generate silently drops logprobs, so chat is used).
    top_logprobs=20 with a -20.0 floor for any letter below the top 20:
    an earlier run with top_logprobs=5 dropped every event where Bolt
    was confident enough that a letter fell out of the top 5, which
    selected only low-confidence events and invalidated its verdict;
    that run was discarded and the test re-run with this fix.
  - conservative channel = base logprobs + the paper's count prior
  - responsive channel = base logprobs + a bounded directed-pair score
    (decision-level analogue of mu.d: pair counts, tanh-bounded, so the
    state stays bounded like the paper's capped PA-I step)
  - geometric mixing + reliability-weight update exactly as Eq. 10-11
  - Recent-4 = the same scorer re-prompted with the last 4 typed
    interactions (the paper's own definition, §7.1)
  - predict-then-update protocol (§6.1): every method commits before the
    target is revealed; a correction only helps LATER events.
This is therefore a mechanism test (does typed-feedback state above a
frozen Bolt scorer beat history replay, and does the gain grow with
feedback depth?), NOT a replication of TCPA-Bench numbers.

Synthetic stream (no TCPA-Bench release to reuse): 8 users, each with a
hidden fixed ranking over 6 action identities. Each event shows a slate
of 4 identities under a neutral context; the target is the user's
top-ranked identity present in the slate. Contexts are unrelated to the
ranking, so the frozen scorer cannot know the user: any gain over it
must come from the feedback state. Identities are generic actions so
Bolt's generic preference is a real, non-degenerate prior.

Pre-registered verdict rule:
  SUPPORTED  if Jev-DLC > Recent-4 overall AND depth>=2 margin > 0
             AND Jev-DLC > Frozen
  NEGATIVE   if Jev-DLC < Frozen overall
  otherwise  INCONCLUSIVE
Temperature 0, think off, for reproducibility (vendor default is 1.0).
"""
import json, math, os, random, sys, urllib.request

MODEL = "nemotron-3.5-bolt-e4b-q4-ctx8192"
URL = "http://127.0.0.1:11434/api/chat"
OUT = os.environ.get("JEVDLC_OUT",
                     os.path.dirname(os.path.abspath(__file__)))
SEED = 20261025
N_USERS, EVENTS_PER_USER = 8, 24

IDENTITIES = [
    "Send a summary email",
    "Schedule a 15-minute call",
    "Post an update in the team chat",
    "Create a task ticket",
    "Book a meeting room",
    "Share a document link",
]
CONTEXTS = [
    "The team needs a follow-up on the project.",
    "A decision is needed before the end of the day.",
    "The weekly review is coming up.",
    "A colleague asked for help with a task.",
    "The project plan needs an update.",
    "A deadline is approaching.",
    "The team is coordinating next steps.",
    "A status update is due.",
    "Work needs to be handed over to the next shift.",
    "A request came in from another team.",
    "The sprint board needs tidying.",
    "A meeting just ended and actions are open.",
]

# paper constants
ALPHA, COUNT_W, PAIR_W, ETA, SIGMA = 0.5, 2.0, 2.0, 0.25, 0.1


def scorer(prompt):
    """Frozen scorer: Bolt first-token logprobs for letters A-D."""
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "think": False, "options": {"temperature": 0},
            "logprobs": True, "top_logprobs": 20}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    lp = out["logprobs"][0]["top_logprobs"]
    scores = {c: -20.0 for c in "ABCD"}  # floor for any letter below top-20
    for entry in lp:
        tok = entry["token"].strip()
        if tok in "ABCD":
            scores[tok] = entry["logprob"]
    return scores


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def main():
    rng = random.Random(SEED)
    rows = []
    # hidden rankings, one permutation per user
    rankings = []
    for u in range(N_USERS):
        perm = list(range(len(IDENTITIES)))
        rng.shuffle(perm)
        rankings.append(perm)
    for u in range(N_USERS):
        rank = {ident: r for r, ident in enumerate(rankings[u])}
        counts = {i: 0 for i in range(len(IDENTITIES))}
        pairs = {}  # (preferred, rejected) -> count
        n_corr = 0
        g = {"c": 0.0, "r": 0.0}
        recent = []  # typed feedback lines, max 4
        for t in range(EVENTS_PER_USER):
            slate = rng.sample(range(len(IDENTITIES)), 4)
            ctx = CONTEXTS[(u * EVENTS_PER_USER + t) % len(CONTEXTS)]
            letters = "ABCD"
            slate_txt = "\n".join(
                f"{letters[i]}. {IDENTITIES[s]}" for i, s in enumerate(slate))
            prompt = (f"{ctx}\nWhich action is best? Reply with only one "
                      f"letter.\n{slate_txt}")
            try:
                base_lp = scorer(prompt)
            except Exception as e:
                print(f"SCORER ERROR user {u} event {t}: {e}", flush=True)
                continue
            base = [base_lp[letters[i]] for i in range(4)]
            target_pos = min(range(4), key=lambda i: rank[slate[i]])
            target = slate[target_pos]
            # --- frozen ---
            frozen_pred = slate[max(range(4), key=lambda i: base[i])]
            # --- Jev-DLC channels ---
            K = 4
            q = [COUNT_W * math.log((counts[slate[i]] + ALPHA) /
                                    (n_corr + ALPHA * K)) for i in range(4)]
            pair_net = []
            for i in range(4):
                net = 0
                for j in range(4):
                    net += pairs.get((slate[i], slate[j]), 0)
                    net -= pairs.get((slate[j], slate[i]), 0)
                pair_net.append(net)
            s_c = [base[i] + q[i] for i in range(4)]
            s_r = [base[i] + PAIR_W * math.tanh(pair_net[i] / 2.0)
                   for i in range(4)]
            p_c, p_r = softmax(s_c), softmax(s_r)
            w_r = math.exp(g["r"]) / (math.exp(g["c"]) + math.exp(g["r"]))
            mix = [p_c[i] ** (1 - w_r) * p_r[i] ** w_r for i in range(4)]
            mix_s = sum(mix)
            mix = [m / mix_s for m in mix]
            jev_pred = slate[max(range(4), key=lambda i: mix[i])]
            # --- Recent-4: scorer re-prompted with last 4 typed feedbacks ---
            if recent:
                hist = "\n".join(recent)
                prompt4 = (f"{ctx}\nPrevious corrections from this user:\n{hist}\n"
                           f"Which action is best? Reply with only one "
                           f"letter.\n{slate_txt}")
                try:
                    lp4 = scorer(prompt4)
                    r4 = [lp4[letters[i]] for i in range(4)]
                    recent_pred = slate[max(range(4), key=lambda i: r4[i])]
                except Exception as e:
                    print(f"RECENT4 ERROR user {u} event {t}: {e}", flush=True)
                    recent_pred = frozen_pred
            else:
                recent_pred = frozen_pred
            # --- commit, then reveal (predict-then-update) ---
            depth = n_corr
            rows.append({"user": u, "event": t, "depth": depth,
                         "slate": [IDENTITIES[s] for s in slate],
                         "target": IDENTITIES[target],
                         "frozen": IDENTITIES[frozen_pred],
                         "recent4": IDENTITIES[recent_pred],
                         "jevdlc": IDENTITIES[jev_pred],
                         "frozen_ok": frozen_pred == target,
                         "recent4_ok": recent_pred == target,
                         "jevdlc_ok": jev_pred == target})
            # feedback: accept if committed == target, else revoke
            if jev_pred == target:
                # accept: pair with highest-probability safe competitor
                comp = max((i for i in range(4) if slate[i] != jev_pred),
                           key=lambda i: mix[i])
                pref, rej = target, slate[comp]
            else:
                pref, rej = target, jev_pred
            # channel reliability update (zero-one loss, Eq. 10-11)
            for ch, sc in (("c", s_c), ("r", s_r)):
                arg = slate[max(range(4), key=lambda i: sc[i])]
                if arg != target:
                    g[ch] -= ETA
            w = {}
            tot = math.exp(g["c"]) + math.exp(g["r"])
            for ch in ("c", "r"):
                w[ch] = 0.9 * (math.exp(g[ch]) / tot) + SIGMA / 2
            g = {"c": math.log(w["c"]), "r": math.log(w["r"])}
            # state update: counts + directed pair (bounded: ints only)
            counts[pref] += 1
            pairs[(pref, rej)] = pairs.get((pref, rej), 0) + 1
            n_corr += 1
            recent.append(f"User chose '{IDENTITIES[pref]}' instead of "
                          f"'{IDENTITIES[rej]}'.")
            recent = recent[-4:]
            if (len(rows) % 32) == 0:
                print(f"progress: {len(rows)} events scored", flush=True)
    with open(f"{OUT}/jevdlc_events.csv", "w") as f:
        f.write("user,event,depth,target,frozen,recent4,jevdlc,"
                "frozen_ok,recent4_ok,jevdlc_ok\n")
        for r in rows:
            f.write(f"{r['user']},{r['event']},{r['depth']},"
                    f"\"{r['target']}\",\"{r['frozen']}\","
                    f"\"{r['recent4']}\",\"{r['jevdlc']}\","
                    f"{int(r['frozen_ok'])},{int(r['recent4_ok'])},"
                    f"{int(r['jevdlc_ok'])}\n")

    def user_macro(method, subset=None):
        per_user = []
        for u in range(N_USERS):
            rs = [r for r in rows if r["user"] == u and
                  (subset is None or subset(r))]
            if rs:
                per_user.append(sum(r[method + "_ok"] for r in rs) / len(rs))
        return (sum(per_user) / len(per_user) if per_user else 0.0,
                per_user)

    summary = {"n_events": len(rows), "n_users": N_USERS,
               "rankings": [[IDENTITIES[i] for i in p] for p in rankings]}
    for name, subset in (("overall", None),
                         ("depth>=1", lambda r: r["depth"] >= 1),
                         ("depth>=2", lambda r: r["depth"] >= 2),
                         ("depth>=4", lambda r: r["depth"] >= 4),
                         ("depth>=8", lambda r: r["depth"] >= 8)):
        summary[name] = {}
        for m in ("frozen", "recent4", "jevdlc"):
            acc, per = user_macro(m, subset)
            summary[name][m] = {"acc": round(acc, 4), "per_user": [round(x, 3) for x in per],
                                "n": len([r for r in rows if subset is None or subset(r)])}
    overall = summary["overall"]
    margin_recent = overall["jevdlc"]["acc"] - overall["recent4"]["acc"]
    margin_d2 = summary["depth>=2"]["jevdlc"]["acc"] - summary["depth>=2"]["recent4"]["acc"]
    if (overall["jevdlc"]["acc"] > overall["recent4"]["acc"] and margin_d2 > 0
            and overall["jevdlc"]["acc"] > overall["frozen"]["acc"]):
        verdict = "SUPPORTED"
    elif overall["jevdlc"]["acc"] < overall["frozen"]["acc"]:
        verdict = "NEGATIVE"
    else:
        verdict = "INCONCLUSIVE"
    summary["verdict"] = verdict
    summary["margin_vs_recent4_overall"] = round(margin_recent, 4)
    summary["margin_vs_recent4_depth>=2"] = round(margin_d2, 4)
    with open(f"{OUT}/jevdlc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
