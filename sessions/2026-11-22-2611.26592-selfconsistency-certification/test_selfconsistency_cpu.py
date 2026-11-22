"""CPU test for self-consistency certification (arXiv:2611.26592).

Live experiments (all run here, no model calls; deterministic seeds):
  E0  Unit checks: normal_ppf/cdf sanity, rho/tau-sigma round trip,
      pi_positive, betting e-process crosses on all-+1 and never on
      all--1 increments.
  E1  rho=0 control (theta=-0.15): the single-unit rule is a valid
      population test when rollouts are independent -- crossing must
      stay at/below nominal eps = 0.05.
  E2  AIME regime (theta=-0.15, rho=0.48, population null TRUE):
      the single-unit rule crosses far above eps (paper: 24% of
      prompts at eps=0.05), and the rate grows with the rollout
      budget toward pi = P(mu > 0) -- claim (ii).
  E3  Realized-null split (same regime, N=256): conditioned on
      mu <= 0 the single-unit rule stays valid (<= eps), conditioned
      on mu > 0 it crosses with high probability -- claim (i).
  E4  ESS saturation (claim (iii)): variance of the within-unit mean
      of latent increments follows rho + (1-rho)/n and saturates at
      rho; ESS(256) ~ 1/rho ~ 2.1, i.e. 256 rollouts carry the
      information of ~2 independent draws.
  E5  Prompt-level rule restores validity (claim (iv)): one rollout
      per prompt, theta=-0.15, rho=0.48 -- crossing stays <= eps;
      and under a pure rollout budget (B=256, theta=+0.4) the
      prompt-level allocation has strictly higher power than
      spending all 256 rollouts on one prompt.
  E6  GSM8K regime (theta=+0.96, rho=0.435, population null FALSE):
      the gap is dormant -- the single-unit rule crosses with high
      probability, as it should when the model is right on average.
  E7  Determinism: E2 repeated gives identical numbers.

Usage: python3 test_selfconsistency_cpu.py  (writes results.json)
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.agent.selfconsistency import (  # noqa: E402
    SplitMix64,
    betting_eprocess,
    crossing_step,
    ess_within_unit,
    icc_anova,
    normal_cdf,
    normal_ppf,
    pi_positive,
    rho_from_tau_sigma,
    run_prompt_level_experiment,
    run_single_unit_experiment,
    simulate_unit,
    tau_sigma_from_rho,
    theta_hat_from_accuracy,
    within_unit_mean_var,
)

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 0.05


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def main():
    results = {}

    # ---- E0: unit checks ----
    check("E0 ppf(0.5)=0", abs(normal_ppf(0.5)) < 1e-12)
    check("E0 ppf(0.975)~1.96", abs(normal_ppf(0.975) - 1.959964) < 1e-5,
          f"got {normal_ppf(0.975):.6f}")
    check("E0 cdf(0)=0.5", abs(normal_cdf(0.0) - 0.5) < 1e-12)
    tau, sigma = tau_sigma_from_rho(0.48)
    check("E0 rho round trip",
          abs(rho_from_tau_sigma(tau, sigma) - 0.48) < 1e-12)
    check("E0 pi(theta=-0.15, tau=0.693)~0.414",
          abs(pi_positive(-0.15, tau) - 0.414) < 0.005,
          f"got {pi_positive(-0.15, tau):.3f}")
    path = betting_eprocess([1.0] * 20)
    check("E0 e-process crosses on all +1",
          crossing_step([1.0] * 20, eps=EPS) is not None)
    check("E0 e-process never crosses on all -1",
          crossing_step([-1.0] * 200, eps=EPS) is None)
    check("E0 e-process wealth starts at 1", path[0] == 1.0)
    check("E0 theta_hat(0.5)=0",
          abs(theta_hat_from_accuracy(0.5)) < 1e-9)
    check("E0 icc on perfectly split groups ~1",
          abs(icc_anova([[1, 1, 1, 1], [0, 0, 0, 0],
                          [1, 1, 1, 1], [0, 0, 0, 0]]) - 1.0) < 1e-9)
    check("E0 icc on identical groups ~0",
          abs(icc_anova([[1, 0, 1, 0], [1, 0, 1, 0],
                          [0, 1, 0, 1], [0, 1, 0, 1]])) < 1e-9)
    results["E0_units"] = "all passed"

    # ---- E1: rho=0 control ----
    e1 = run_single_unit_experiment(seed=7, theta=-0.15, rho=0.0,
                                    n_rollouts=64, n_units=2000, eps=EPS)
    print(f"E1 rho=0 control: crossing={e1['crossing_rate']:.4f}")
    check("E1 rho=0 crossing <= eps", e1["crossing_rate"] < 0.02,
          f"rate={e1['crossing_rate']:.4f}")
    results["E1_rho0_control"] = e1

    # ---- E2: AIME regime, population null true ----
    e2_64 = run_single_unit_experiment(seed=11, theta=-0.15, rho=0.48,
                                       n_rollouts=64, n_units=2000, eps=EPS)
    e2_256 = run_single_unit_experiment(seed=11, theta=-0.15, rho=0.48,
                                        n_rollouts=256, n_units=2000, eps=EPS)
    print(f"E2 AIME regime: N=64 crossing={e2_64['crossing_rate']:.3f} "
          f"(pi={e2_64['pi']:.3f}); N=256 crossing="
          f"{e2_256['crossing_rate']:.3f}")
    check("E2 N=64 crossing far above eps (paper: 0.24)",
          e2_64["crossing_rate"] > 0.10,
          f"rate={e2_64['crossing_rate']:.3f}")
    check("E2 crossing grows with rollout budget",
          e2_256["crossing_rate"] > e2_64["crossing_rate"],
          f"64:{e2_64['crossing_rate']:.3f} 256:{e2_256['crossing_rate']:.3f}")
    check("E2 N=256 crossing approaches pi from below",
          0.25 < e2_256["crossing_rate"] < e2_256["pi"] + 0.05,
          f"rate={e2_256['crossing_rate']:.3f} pi={e2_256['pi']:.3f}")
    results["E2_aime_regime_N64"] = e2_64
    results["E2_aime_regime_N256"] = e2_256

    # ---- E3: realized-null split ----
    print(f"E3 realized-null split (N=256): mu<=0 rate="
          f"{e2_256['crossing_rate_realized_null']:.4f} "
          f"(n={e2_256['n_realized_null']}); mu>0 rate="
          f"{e2_256['crossing_rate_realized_alt']:.3f} "
          f"(n={e2_256['n_realized_alt']})")
    check("E3 valid for the realized null (mu<=0)",
          e2_256["crossing_rate_realized_null"] < 0.08,
          f"rate={e2_256['crossing_rate_realized_null']:.4f}")
    check("E3 high power when realized null is false (mu>0)",
          e2_256["crossing_rate_realized_alt"] > 0.50,
          f"rate={e2_256['crossing_rate_realized_alt']:.3f}")
    results["E3_realized_null_split"] = {
        "rate_mu_le_0": e2_256["crossing_rate_realized_null"],
        "n_mu_le_0": e2_256["n_realized_null"],
        "rate_mu_gt_0": e2_256["crossing_rate_realized_alt"],
        "n_mu_gt_0": e2_256["n_realized_alt"],
    }

    # ---- E4: ESS saturation (latent continuous increments) ----
    rho = 0.48
    tau, sigma = tau_sigma_from_rho(rho)
    rng = SplitMix64(23)
    m_units, n_max = 4000, 256
    prefixes = [1, 4, 16, 64, 256]
    sums = {n: [] for n in prefixes}
    for _ in range(m_units):
        _, incs = simulate_unit(rng, 0.0, tau, sigma, n_max, binary=False)
        run = 0.0
        idx = 0
        for n in prefixes:
            while idx < n:
                run += incs[idx]
                idx += 1
            sums[n].append(run / n)
    ess_rows = []
    for n in prefixes:
        vals = sums[n]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        pred = within_unit_mean_var(n, rho)
        ess = ess_within_unit(n, rho)
        ess_rows.append({"n": n, "empirical_var": round(var, 4),
                         "predicted_var": round(pred, 4),
                         "ess": round(ess, 3)})
        print(f"E4 n={n:3d} emp_var={var:.4f} pred_var={pred:.4f} "
              f"ESS={ess:.2f}")
    check("E4 var(n=256) saturates near rho, not near 0",
          abs(ess_rows[-1]["empirical_var"] - rho) < 0.05,
          f"emp={ess_rows[-1]['empirical_var']:.4f} rho={rho}")
    check("E4 var matches rho+(1-rho)/n at every prefix",
          all(abs(r["empirical_var"] - r["predicted_var"]) < 0.03
              for r in ess_rows))
    check("E4 ESS(256) ~ 1/rho ~ 2.08",
          1.5 < ess_within_unit(256, rho) < 3.0,
          f"ESS={ess_within_unit(256, rho):.2f}")
    results["E4_ess_saturation"] = ess_rows

    # ---- E5: prompt-level restores validity; optimal allocation ----
    e5_valid = run_prompt_level_experiment(seed=31, theta=-0.15, rho=0.48,
                                          n_prompts=256, n_reps=2000,
                                          eps=EPS)
    print(f"E5 prompt-level validity: crossing="
          f"{e5_valid['crossing_rate']:.4f}")
    check("E5 prompt-level crossing <= eps",
          e5_valid["crossing_rate"] < 0.08,
          f"rate={e5_valid['crossing_rate']:.4f}")
    e5_pow_prompt = run_prompt_level_experiment(
        seed=37, theta=0.4, rho=0.48, n_prompts=256, n_reps=1000, eps=EPS)
    e5_pow_single = run_single_unit_experiment(
        seed=37, theta=0.4, rho=0.48, n_rollouts=256, n_units=1000, eps=EPS)
    print(f"E5 power (theta=+0.4, B=256): prompt-level="
          f"{e5_pow_prompt['crossing_rate']:.3f} single-unit="
          f"{e5_pow_single['crossing_rate']:.3f}")
    check("E5 prompt-level power exceeds single-unit by >0.15",
          e5_pow_prompt["crossing_rate"]
          > e5_pow_single["crossing_rate"] + 0.15,
          f"prompt={e5_pow_prompt['crossing_rate']:.3f} "
          f"single={e5_pow_single['crossing_rate']:.3f}")
    check("E5 prompt-level power high when null false",
          e5_pow_prompt["crossing_rate"] > 0.85,
          f"rate={e5_pow_prompt['crossing_rate']:.3f}")
    results["E5_prompt_level_validity"] = e5_valid
    results["E5_power_prompt_level"] = e5_pow_prompt
    results["E5_power_single_unit"] = e5_pow_single

    # ---- E6: GSM8K regime, gap dormant ----
    e6 = run_single_unit_experiment(seed=41, theta=0.96, rho=0.435,
                                    n_rollouts=64, n_units=2000, eps=EPS)
    print(f"E6 GSM8K regime: crossing={e6['crossing_rate']:.3f} "
          f"(pi={e6['pi']:.3f})")
    check("E6 GSM8K regime single-unit crosses with high probability",
          e6["crossing_rate"] > 0.80,
          f"rate={e6['crossing_rate']:.3f}")
    results["E6_gsm8k_regime"] = e6

    # ---- E7: determinism ----
    e2_repeat = run_single_unit_experiment(seed=11, theta=-0.15, rho=0.48,
                                           n_rollouts=64, n_units=2000,
                                           eps=EPS)
    check("E7 deterministic repeat identical",
          e2_repeat == e2_64)
    results["E7_determinism"] = "identical"

    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("ALL CHECKS PASSED — results written to results.json")


if __name__ == "__main__":
    main()
