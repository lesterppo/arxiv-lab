"""Diagnostic: inspect candidate pools — traces, scores, and the
mean_logp vs nll_of_text discrepancy. Safe to import (no training)."""
import sys
sys.path.insert(0, "/content")

import colab_sft_sampling as S
from data_gen import gen_problems, expert_trace, is_correct_trace

model, tok = S.load_base()
train_probs = gen_problems(40, 0)

for pi in [0, 1, 2]:
    p = train_probs[pi]
    expert = expert_trace(p)
    lp_e = S.mean_logp(model, tok, p["question"], expert)
    nj_e = S.nll_of_text(model, tok, p["question"] + " " + expert)
    print("=" * 70, flush=True)
    print(f"Q: {p['question']}  ans={p['answer']}", flush=True)
    print(f"EXPERT: mean_logp={lp_e:.3f} nll_joint={nj_e:.3f} "
          f"ntok={len(tok(expert).input_ids)}", flush=True)
    print(f"  text: {expert[:160]!r}", flush=True)
    samps = S.generate(model, tok, p["question"], S.GEN_TOKENS,
                       do_sample=True, temperature=1.0, num_return=6)
    for si, s in enumerate(samps):
        lp = S.mean_logp(model, tok, p["question"], s)
        nj = S.nll_of_text(model, tok, p["question"] + " " + s)
        print(f"  s{si}: mean_logp={lp:.3f} nll_joint={nj:.3f} "
              f"correct={is_correct_trace(s, p['answer'])} "
              f"ntok={len(tok(s).input_ids)} text={s[:130]!r}", flush=True)
print("DIAG_DONE", flush=True)
