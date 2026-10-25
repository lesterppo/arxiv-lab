"""Decision-level Jev-DLC (arXiv:2610.31521).

Jev-DLC turns typed corrections into compact causal state above a frozen
scorer: a conservative channel (base scores + a Dirichlet-smoothed count
prior over the semantic identity of the preferred action, paper Eq. 5-6)
and a responsive channel (base scores + a bounded directed-pair score
for the (preferred, rejected) contrast, the decision-level analogue of
the paper's capped PA-I correction vector, Eq. 7-9). The two channel
distributions are combined by geometric mixing with reliability weights
updated by zero-one loss after each revealed feedback (Eq. 10-11).

Paper constants: alpha=0.5, count weight 2.0, eta=0.25, sigma=0.1,
gamma=C=1. The paper's scorer is JEV-SFT (first-token option logprobs
with a learned 32-dim hidden-state projection); this module takes base
scores as input, so any frozen scorer can be used -- on the live Bolt
test the base scores were Ollama first-token logprobs for the candidate
letters. No raw interaction history is stored: state is per-identity
counts, directed pair counts, and two reliability weights. No numpy,
stdlib only, no network at import time.
"""
import math

ALPHA = 0.5
COUNT_W = 2.0
PAIR_W = 2.0
ETA = 0.25
SIGMA = 0.1


def _softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


class JevDLCState:
    """Per-user typed-feedback state above a frozen scorer."""

    def __init__(self, alpha=ALPHA, count_w=COUNT_W, pair_w=PAIR_W,
                 eta=ETA, sigma=SIGMA):
        self.alpha = alpha
        self.count_w = count_w
        self.pair_w = pair_w
        self.eta = eta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.counts = {}
        self.pairs = {}
        self.n_corr = 0
        self.g = {"c": 0.0, "r": 0.0}

    def predict(self, base, slate):
        """Return (predicted slate index, info). base: frozen scorer scores
        aligned with slate (any scale); slate: hashable action identities."""
        k = len(slate)
        q = [self.count_w * math.log((self.counts.get(s, 0) + self.alpha) /
                                     (self.n_corr + self.alpha * k))
             for s in slate]
        pair_net = []
        for s in slate:
            net = 0
            for o in slate:
                net += self.pairs.get((s, o), 0)
                net -= self.pairs.get((o, s), 0)
            pair_net.append(net)
        s_c = [base[i] + q[i] for i in range(k)]
        s_r = [base[i] + self.pair_w * math.tanh(pair_net[i] / 2.0)
               for i in range(k)]
        p_c, p_r = _softmax(s_c), _softmax(s_r)
        w_r = math.exp(self.g["r"]) / (math.exp(self.g["c"]) + math.exp(self.g["r"]))
        mix = [p_c[i] ** (1 - w_r) * p_r[i] ** w_r for i in range(k)]
        tot = sum(mix)
        mix = [m / tot for m in mix]
        pred = max(range(k), key=lambda i: mix[i])
        return pred, {"p_c": p_c, "p_r": p_r, "mix": mix, "rho": w_r,
                      "s_c": s_c, "s_r": s_r}

    def update(self, pred_idx, target, slate, info):
        """Apply revealed feedback. pred_idx: committed slate index;
        target: preferred action identity. On accept (pred == target) the
        paper pairs the confirmed action with its highest-probability safe
        competitor; on revoke the rejected action is the prediction."""
        mix = info["mix"]
        if slate[pred_idx] == target:
            comp = max((i for i in range(len(slate))
                        if slate[i] != target), key=lambda i: mix[i])
            pref, rej = target, slate[comp]
        else:
            pref, rej = target, slate[pred_idx]
        for ch, sc in (("c", info["s_c"]), ("r", info["s_r"])):
            arg = slate[max(range(len(slate)), key=lambda i: sc[i])]
            if arg != target:
                self.g[ch] -= self.eta
        tot = math.exp(self.g["c"]) + math.exp(self.g["r"])
        w = {}
        for ch in ("c", "r"):
            w[ch] = (1 - self.sigma) * (math.exp(self.g[ch]) / tot) \
                + self.sigma / 2
        self.g = {"c": math.log(w["c"]), "r": math.log(w["r"])}
        self.counts[pref] = self.counts.get(pref, 0) + 1
        self.pairs[(pref, rej)] = self.pairs.get((pref, rej), 0) + 1
        self.n_corr += 1
        return pref, rej

    def state_size(self):
        """Rough served-state size in entries (counts + pairs + 2 weights).
        The paper's served state is ~324 bytes on TCPA-Repeat; this
        decision-level state is the same order: a handful of small ints."""
        return len(self.counts) + len(self.pairs) + 2
