#!/usr/bin/env python3
"""CPU check for the CREDIT (2610.35295) dataset/label construction.

Stdlib only. Re-implements the driver's dataset, rendering, parsing and
label/balance logic (the driver itself needs a GPU + pip installs, so it
cannot be imported here) and asserts the invariants the Colab run relies
on. Run: python3 test_credit_dataset_cpu.py  ->  ALL CPU CHECKS PASSED
"""
import re, string, random

SEED = 20261029
ATTRS = {
    "favorite fruit": ["mango", "papaya", "kiwi", "guava", "lychee",
                       "durian", "rambutan", "persimmon"],
    "favorite color": ["crimson", "teal", "ochre", "lavender",
                       "maroon", "turquoise", "coral", "indigo"],
    "pet's name": ["Pip", "Milo", "Zara", "Kiki", "Rolo", "Nia", "Bex", "Jax"],
    "home city": ["Virelia", "Karnova", "Zelqua", "Ostara",
                  "Myrtha", "Caldra", "Threxia", "Pelnora"],
}
TRANSFER_ATTRS = {
    "favorite season": ["spring", "summer", "autumn", "winter"],
    "lucky number": ["three", "seven", "nine", "eleven", "thirteen", "twenty"],
    "favorite instrument": ["violin", "cello", "flute", "harp",
                           "trumpet", "oboe"],
}
TRAIN_ENTITIES = ["Virelith", "Karnovex", "Zelquint", "Ostralen", "Myrthik",
                  "Caldrax", "Threxil", "Pelnor", "Sarnavik", "Ilthara",
                  "Ondrelis", "Vemora", "Talsyn", "Ruvik", "Hespar",
                  "Nolquin", "Dravok", "Selmira", "Korvath", "Umbrex",
                  "Telnar", "Vorquill", "Ashvrel", "Grendol"]
HELDOUT_ENTITIES = ["Marnel", "Quistrel", "Beldorin", "Xanthis", "Corvane",
                    "Pellix", "Ostrev", "Wynthor", "Faldrin", "Ysolde",
                    "Brannik", "Cethra", "Duvane", "Elowen", "Fenric", "Galvyn"]
TRANSFER_ENTITIES = ["Halvord", "Isolde", "Jorvik", "Lunaris", "Morthal",
                     "Nivara", "Orenth", "Pyralis", "Quenlin", "Raveth",
                     "Sylvar", "Thandor"]

def q1(e, a): return f"What is {e}'s {a}?"
def q2(e, a): return f"For {e}, what is the {a}?"
def q3(e, a): return f"Question: {e}'s {a} is what?"
QUESTIONS = [q1, q2, q3]
def s1(e, a, v): return f"Supplied information: {e}'s {a} is {v}."
def s2(e, a, v): return f"Note: {e}'s {a} is {v}."
def s3(e, a, v): return f"Record: {e}'s {a} = {v}."
def s_corr(e, a, v, wrong):
    return (f"An earlier note claimed {e}'s {a} is {wrong}. "
            f"The corrected record states {e}'s {a} is {v}.")
SUPPLIED = [s1, s2, s3]
TRAIN_PAIRS = [(0, 0, False), (1, 1, False), (0, 1, False), (1, 0, False)]
HELDOUT_PAIRS = [(2, 0, False), (0, 2, False), (2, 2, False)]
TRANSFER_PAIRS = [(1, 2, False), (2, 1, False), (0, 0, True), (0, 2, True)]

def render(item, pair):
    si, qi, corr = pair
    e, a, v = item["entity"], item["attr"], item["value"]
    pool = (TRANSFER_ATTRS if a in TRANSFER_ATTRS else ATTRS)[a]
    if corr:
        wrong = next(w for w in pool if w.lower() != v.lower())
        supplied = s_corr(e, a, v, wrong)
    else:
        supplied = SUPPLIED[si](e, a, v)
    question = QUESTIONS[qi](e, a)
    with_p = (f"{supplied}\n{question}\nAnswer with just the value, then "
              "state whether you used the supplied information.\nAnswer:")
    without_p = f"{question}\nAnswer with just the value.\nAnswer:"
    return with_p, without_p

def build_candidates(entities, attrs, pairs, repeats, split):
    cands, idx = [], 0
    for rep in range(repeats):
        for (e, a) in [(e, a) for e in entities for a in attrs]:
            pool = attrs[a]
            v = pool[(idx + rep * 3) % len(pool)]
            cands.append({"split": split, "entity": e, "attr": a,
                          "value": v, "pair": pairs[idx % len(pairs)]})
            idx += 1
    return cands

def parse_answer(text):
    m = re.search(r"[A-Za-z]+", text)
    return m.group(0).lower() if m else ""

def parse_attribution(text):
    t = text.lower()
    if ("did not use" in t or "didn't use" in t or "not use" in t
            or "didnt use" in t):
        return 0
    if "used the supplied" in t or "i used" in t or "i did use" in t:
        return 1
    return None

def check(cond, msg):
    assert cond, msg
    print(f"  ok: {msg}")

def main():
    print("== counts ==")
    tr = build_candidates(TRAIN_ENTITIES, ATTRS, TRAIN_PAIRS, 3, "train")
    ho = build_candidates(HELDOUT_ENTITIES, ATTRS, HELDOUT_PAIRS, 2, "heldout")
    tf = build_candidates(TRANSFER_ENTITIES, TRANSFER_ATTRS, TRANSFER_PAIRS, 4, "transfer")
    check(len(tr) == 288, f"train candidates == 288 (got {len(tr)})")
    check(len(ho) == 128, f"heldout candidates == 128 (got {len(ho)})")
    check(len(tf) == 144, f"transfer candidates == 144 (got {len(tf)})")
    check(len(tr) + len(ho) + len(tf) == 560, "total candidates == 560")

    print("== disjointness ==")
    check(not (set(TRAIN_ENTITIES) & set(HELDOUT_ENTITIES)), "train/heldout entities disjoint")
    check(not (set(TRAIN_ENTITIES) & set(TRANSFER_ENTITIES)), "train/transfer entities disjoint")
    check(not (set(HELDOUT_ENTITIES) & set(TRANSFER_ENTITIES)), "heldout/transfer entities disjoint")
    check(not (set(ATTRS) & set(TRANSFER_ATTRS)), "train/transfer attributes disjoint")
    check(not (set(TRAIN_PAIRS) & set(HELDOUT_PAIRS)), "heldout template pairs unseen in training")
    check(any(p[2] for p in TRANSFER_PAIRS), "transfer includes correction pairs")
    check(not any(p[2] for p in TRAIN_PAIRS + HELDOUT_PAIRS), "correction format unseen before transfer")

    print("== rendering ==")
    for c in tr + ho + tf:
        w, wo = render(c, c["pair"])
        check(c["value"] in w, "with-info prompt contains suggested value")
        check(c["value"].lower() not in wo.lower().split(),
              "without-info prompt does not contain suggested value as a word")
        check(w.split("\n")[1] == wo.split("\n")[0],
              "with/without share the identical question line (pure info removal)")
        check("supplied information" in w, "with-info prompt carries attribution instruction")
    # correction format mentions both wrong and corrected values, corrected last
    corr_item = {"entity": "Halvord", "attr": "lucky number", "value": "seven"}
    w, _ = render(corr_item, (0, 0, True))
    check("earlier note" in w and "corrected record" in w, "correction prompt has both sources")
    check(w.index("corrected record") > w.index("earlier note"), "corrected value stated after earlier claim")

    print("== parsing ==")
    check(parse_answer(" mango. I used the supplied information.") == "mango", "parse_answer first token")
    check(parse_answer("  Crimson, I did not use it.") == "crimson", "parse_answer strips punctuation/case")
    check(parse_answer("") == "", "parse_answer empty -> ''")
    check(parse_attribution("mango. I used the supplied information.") == 1, "parse_attribution USED")
    check(parse_attribution("mango. I did not use the supplied information.") == 0, "parse_attribution NOT-USED (did-not checked first)")
    check(parse_attribution("mango.") is None, "parse_attribution missing -> None")
    check(parse_attribution("I did not use the supplied information, I used prior knowledge.") == 0,
          "parse_attribution: did-not wins over later 'used'")

    print("== counterfactual label logic ==")
    # label_used=1 iff without-answer != suggested, given with-answer == suggested
    def label(ans_with, ans_without, suggested):
        if ans_with != suggested:
            return None  # dropped: did not follow suggestion
        return 1 if ans_without != suggested else 0
    check(label("mango", "kiwi", "mango") == 1, "changed-by-suggestion -> USED")
    check(label("mango", "mango", "mango") == 0, "would-have-given-anyway -> NOT-USED")
    check(label("kiwi", "kiwi", "mango") is None, "not following suggestion -> dropped")

    print("== balance logic ==")
    # cap majority at 2x minority, deterministic subsample
    rng = random.Random(SEED + 1)
    used, notused = list(range(100)), list(range(20))
    cap = 2 * min(len(used), len(notused))
    used_capped = rng.sample(used, cap) if len(used) > cap else used
    check(len(used_capped) == 40 and len(notused) == 20, "majority capped at 2x minority (100/20 -> 40/20)")

    print("== training targets ==")
    for lab, phrase in ((1, "I used the supplied information."),
                        (0, "I did not use the supplied information.")):
        target = f" mango. {phrase}" if lab == 1 else f" mango. {phrase}"
        check(parse_answer(target) == "mango", "training target parses to suggested value")
        check(parse_attribution(target) == lab, "training target attribution matches label")

    print("== determinism ==")
    tr2 = build_candidates(TRAIN_ENTITIES, ATTRS, TRAIN_PAIRS, 3, "train")
    check(tr == tr2, "candidate construction is deterministic (same seed/order)")
    check(all(len(c["value"].split()) == 1 for c in tr + ho + tf),
          "all suggested values are single words (first-token parsing valid)")

    print("ALL CPU CHECKS PASSED")

if __name__ == "__main__":
    main()
