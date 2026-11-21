#!/usr/bin/env python3
"""Run the LoopRoute Python-vs-Rust parity sweep and report mismatches.

Parity is defined at the task-spec level (the Python env generator
uses CPython's MT19937, which the Rust side reproduces, but the sweep
does not depend on that: both sides build the env from the same spec
text, so any generator difference would show up as a spec mismatch
instead of silently cancelling out).

Three case kinds:
- gen:  a seeded task printed in the spec format (byte-for-byte
         stdout compare — the generators must be identical because
         PyRandom reproduces CPython's random.Random)
- ep:   one episode run from a spec file (JSON compare: success,
         steps, assists, llm_calls, and the full trajectory of
         [state, action, source, observation])

Specs cover locked/unlocked tasks, seeds 5000..5039, and recipe sizes
n = 1, 2, 4, 5 (n = 3 is the default). Modes: fast-only, routed K=2,
routed K=3.

Exits non-zero on any mismatch. Builds the Rust binary first if it is
missing (CARGO_TARGET_DIR keeps target/ out of the session dir).
"""
import json
import os
import subprocess
import sys
import tempfile

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DIR = os.path.dirname(RUST_DIR)
sys.path.insert(0, os.path.join(SESSION_DIR, "..", "..", "src"))

from arxiv_lab.agent.looproute import (  # noqa: E402
    FastPolicy, KitchenWorld, LoopRouter, oracle_assistant, run_episode,
)

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", "/tmp/looproute-rs-target")
RS_BIN = os.environ.get(
    "LOOPROUTE_RS_BIN", os.path.join(TARGET_DIR, "release", "looproute"))


def ensure_binary():
    if os.path.exists(RS_BIN):
        return
    print("building Rust binary ...")
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = TARGET_DIR
    subprocess.run(["cargo", "build", "--release", "--offline"],
                   cwd=RUST_DIR, env=env, check=True)


def spec_text(env):
    """Canonical spec text for a Python env (must match Rust spec_of_env)."""
    lines = ["# looproute-spec-v1",
             f"locked={1 if env.locked else 0}",
             "recipe " + " ".join(env.recipe)]
    for i, ing in enumerate(env.recipe):
        if env.locked and i == len(env.recipe) - 1:
            lines.append(f"chest {ing}")
        else:
            lines.append(f"place {ing} {env.placement[ing][1]}")
    lines.append(f"key {env.key_room}")
    return "\n".join(lines) + "\n"


def env_from_spec(text, seed):
    """Build a Python env from spec text (mirrors Rust from_spec).

    The seed is carried through because Python's plan_to_goal rebuilds
    its search probe as KitchenWorld(env.seed, ...); the spec was
    extracted from that seeded env, so the probe regenerates the same
    task. (Rust's plan_to_goal copies the task fields instead, which
    is equivalent for a seeded task.)
    """
    recipe, placement, chest, key_room, locked = [], {}, None, None, None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "locked=1":
            locked = True
        elif parts[0] == "locked=0":
            locked = False
        elif parts[0] == "recipe":
            recipe = parts[1:]
        elif parts[0] == "place":
            placement[parts[1]] = parts[2]
        elif parts[0] == "chest":
            chest = parts[1]
        elif parts[0] == "key":
            key_room = parts[1]
        else:
            raise ValueError(f"unknown spec line: {line}")
    env = KitchenWorld.__new__(KitchenWorld)
    env.seed = seed
    env.recipe = tuple(recipe)
    env.placement = {}
    for ing in recipe:
        if ing == chest:
            env.placement[ing] = ("chest", "kitchen")
        else:
            env.placement[ing] = (ing, placement[ing])
    env.key_room = key_room
    env.locked = locked
    env.reset()
    return env


def state_str(state):
    room, inv, coll, chest, key, taken = state
    return (f"{room}|inv={','.join(sorted(inv))}|col={','.join(sorted(coll))}"
            f"|chest={int(chest)}|key={int(key)}|taken={','.join(sorted(taken))}")


def episode_dict(res):
    return {
        "success": res["success"],
        "steps": res["steps"],
        "assists": res["assists"],
        "llm_calls": res["llm_calls"],
        "trajectory": [[state_str(s), a, src, obs]
                       for (s, a, src, obs) in res["trajectory"]],
    }


def run_rust(args):
    out = subprocess.run([RS_BIN] + args, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"rust {' '.join(args)} failed: {out.stderr}")
    return out.stdout


def main():
    ensure_binary()
    policy = FastPolicy()
    gen_cases = gen_mismatch = 0
    ep_cases = ep_mismatch = 0
    mismatches = []

    with tempfile.TemporaryDirectory() as tmp:
        # --- gen-spec parity: seeds 5000..5039, locked + unlocked ---
        specs = []
        for seed in range(5000, 5040):
            for locked in (True, False):
                env = KitchenWorld(seed, 3, locked)
                want = spec_text(env)
                got = run_rust(["gen-spec", str(seed), "1" if locked else "0"])
                gen_cases += 1
                if got != want:
                    gen_mismatch += 1
                    mismatches.append(f"gen seed={seed} locked={locked}")
                specs.append((seed, 3, locked, want))
        # --- n-variant specs (no gen-spec check: gen-spec is n=3 only) ---
        for seed in range(6000, 6010):
            for n in (1, 2, 4, 5):
                for locked in (True, False):
                    env = KitchenWorld(seed, n, locked)
                    specs.append((seed, n, locked, spec_text(env)))

        # --- episode parity: every spec x {fast, routed K=2, routed K=3} ---
        for (seed, n, locked, text) in specs:
            path = os.path.join(tmp, f"spec-{seed}-{n}-{int(locked)}.txt")
            with open(path, "w") as fh:
                fh.write(text)
            py_env = env_from_spec(text, seed)
            for mode, threshold in (("fast", 2), ("routed", 2), ("routed", 3)):
                if mode == "fast":
                    py = run_episode(py_env, policy, router=None)
                    rs = json.loads(run_rust(
                        ["run-spec", path, "--mode", "fast"]))
                else:
                    router = LoopRouter(threshold=threshold)
                    py = run_episode(py_env, policy, router=router,
                                     assistant=oracle_assistant)
                    rs = json.loads(run_rust(
                        ["run-spec", path, "--threshold", str(threshold)]))
                ep_cases += 1
                if episode_dict(py) != rs:
                    ep_mismatch += 1
                    pyd = episode_dict(py)
                    first_diff = next(
                        (i for i, (a, b) in enumerate(zip(
                            pyd["trajectory"], rs["trajectory"]))
                         if a != b),
                        min(len(pyd["trajectory"]), len(rs["trajectory"])))
                    mismatches.append(
                        f"ep seed={seed} n={n} locked={locked} mode={mode} "
                        f"K={threshold}: py(success={pyd['success']},"
                        f"steps={pyd['steps']},assists={pyd['assists']},"
                        f"calls={pyd['llm_calls']}) rs(success={rs['success']},"
                        f"steps={rs['steps']},assists={rs['assists']},"
                        f"calls={rs['llm_calls']}) first-diff-step={first_diff} "
                        f"py={pyd['trajectory'][first_diff] if first_diff < len(pyd['trajectory']) else None} "
                        f"rs={rs['trajectory'][first_diff] if first_diff < len(rs['trajectory']) else None}")
                # Rebuild the Python env: run_episode resets it, but
                # rebuild anyway so each mode starts from a fresh env.
                py_env = env_from_spec(text, seed)

    print(f"gen-spec parity: {gen_cases - gen_mismatch}/{gen_cases} match")
    print(f"episode parity:  {ep_cases - ep_mismatch}/{ep_cases} match")
    print(f"total cases: {gen_cases + ep_cases}, "
          f"mismatches: {gen_mismatch + ep_mismatch}")
    if mismatches:
        for m in mismatches[:10]:
            print("MISMATCH:", m)
        sys.exit(1)
    print("PARITY SWEEP: 0 mismatches")


if __name__ == "__main__":
    main()
