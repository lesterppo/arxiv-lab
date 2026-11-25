#!/usr/bin/env python3
"""CPU test for the InstructReplay Colab driver (arXiv:2611.32918).

Extracts the driver's PURE HELPERS block (stdlib only) and checks:
  E1  fact generation is contamination-controlled (unique invented names,
      random dose/frequency pairings, well-formed codes);
  E2  document rendering + dose matching (exact and first-dose) behave;
  E3  GSM8K last-integer scoring (gold '#### N' format, commas, negatives);
  E4  replay mix is exactly 1:1 facts:replay, shuffled, no duplicates;
  E5  replay vs held-out length instruction sets are disjoint (no leakage
      of the length-eval prompts into replay training data).
Deterministic; no torch, no network.
"""
import os, re, sys, random

DRIVER = os.path.join(os.path.dirname(__file__),
                      "colab_instructreplay_driver.py")

def load_pure_helpers():
    src = open(DRIVER).read()
    start = src.index("# ---------------- PURE HELPERS (stdlib) ----------------")
    end = src.index("# ---------------- END PURE HELPERS ----------------")
    block = src[start:end]
    ns = {}
    exec(compile(block, "pure_helpers", "exec"), ns)
    return ns

def main():
    H = load_pure_helpers()
    build_fact = H["build_fact"]; render_document = H["render_document"]
    render_text = H["render_text"]; dose_matches = H["dose_matches"]
    first_dose_matches = H["first_dose_matches"]
    parse_last_int = H["parse_last_int"]; gsm8k_correct = H["gsm8k_correct"]
    build_replay_mix = H["build_replay_mix"]
    REPLAY_GENERAL = H["REPLAY_GENERAL"]; LENGTH_GENERAL = H["LENGTH_GENERAL"]

    # E1: facts are unique, well-formed, randomly paired
    rng = random.Random(20261125)
    used = set()
    facts = [build_fact(rng, used) for _ in range(200)]
    names = [f["name"] for f in facts]
    assert len(set(names)) == 200, "fact names must be unique"
    assert all(f["dose"] and f["freq"] and f["code"].startswith("FD-")
               for f in facts)
    assert len(set((f["name"], f["value"]) for f in facts)) == 200
    doses = set(f["dose"] for f in facts)
    assert len(doses) > 5, "doses should vary across facts"
    print("E1 facts: 200 unique names, %d distinct doses -- PASS" % len(doses))

    # E2: document rendering + dose matching
    f0 = facts[0]
    segs = render_document(f0)
    assert [c for c, _ in segs] == ["prose", "payload", "citation", "prose"]
    text = render_text(segs)
    assert f0["value"] in text and f0["code"] in text
    assert dose_matches("The recommended dose is %s." % f0["value"], f0["value"])
    assert not dose_matches("The recommended dose is 999 mg once daily.",
                            f0["value"])
    assert first_dose_matches("Dose: %s. Also 999 mg." % f0["value"], f0["value"])
    assert not first_dose_matches("Dose: 999 mg once daily. Also %s."
                                  % f0["value"], f0["value"])
    print("E2 render + dose match -- PASS")

    # E3: GSM8K scoring
    assert parse_last_int("The answer is 72.") == 72
    assert parse_last_int("... #### 1,234") == 1234
    assert parse_last_int("no numbers here") is None
    assert parse_last_int("x = -5, then 10") == 10
    assert gsm8k_correct("So the answer is 72.", "work...#### 72")
    assert gsm8k_correct("72 apples, total 72", "#### 72")
    assert not gsm8k_correct("The answer is 71.", "#### 72")
    assert not gsm8k_correct("I don't know.", "#### 72")
    print("E3 GSM8K last-int scoring -- PASS")

    # E4: replay mix is 1:1, shuffled, duplicate-free
    fact_texts = ["fact-%d" % i for i in range(80)]
    replay_texts = ["replay-%d" % i for i in range(80)]
    mixed = build_replay_mix(fact_texts, replay_texts, random.Random(7))
    assert len(mixed) == 160
    assert sorted(mixed) == sorted(fact_texts + replay_texts)
    assert mixed != fact_texts + replay_texts, "mix must be shuffled"
    assert mixed.count("fact-0") == 1 and mixed.count("replay-0") == 1
    print("E4 replay mix 1:1 shuffled duplicate-free -- PASS")

    # E5: replay vs held-out length sets are disjoint
    assert len(REPLAY_GENERAL) == 20 and len(LENGTH_GENERAL) == 20
    assert not set(REPLAY_GENERAL) & set(LENGTH_GENERAL), \
        "length-eval prompts must not appear in replay training data"
    assert len(set(REPLAY_GENERAL)) == 20 and len(set(LENGTH_GENERAL)) == 20
    print("E5 replay/length instruction sets disjoint (20/20 each) -- PASS")

    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
