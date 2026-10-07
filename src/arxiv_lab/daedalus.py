#!/usr/bin/env python3
"""DAEDALUS: bootstrapping agent memory from self-generated tasks.

Reproduces the *protocol* of arXiv:2610.08048 (Antoine Edy et al.):
an explorer agent generates challenging-yet-solvable tasks and adapts their
difficulty from outcomes; a solver attempts them; an extractor derives a
heuristic from each solver failure; the heuristic is accepted only after the
solver reaches Ns consecutive successes *with it in context* having failed
at least once before those successes (too-easy / too-hard tasks contribute
nothing); accepted heuristics are consolidated into a frozen memory bank
that is injected at test time.

Simulation note (read before citing numbers): the explorer, solver,
extractor and judge here are *scripted behavior models* standing in for LLM
agents -- this machine has no local LLM. The solver models an LLM that
(a) sometimes skips information-gathering, (b) sometimes misreads the
environment's hidden conventions, and (c) follows written advice when it is
in context. The extractor models an LLM that usually diagnoses the failure
type correctly but sometimes writes a wrong lesson. What this tests is the
memory-generation *protocol* -- the acceptance rule, difficulty
calibration, consolidation, and transfer of the bank to held-out tasks --
not whether real LLMs emit or consume such heuristics. The judge is
deterministic (checks final environment state), which is strictly stronger
than the paper's LLM judge.

Environment ("VaultWorld"): the agent must open a vault by entering a
4-digit code on a keypad. The code is written on a note hidden in one of R
rooms; other rooms hold decoy notes with wrong codes. A hidden convention
maps each task to the correct room:

  - "color":    the task instruction names a color; the correct room's name
               contains that color (e.g. "red_storage").
  - "initial":  the instruction names a keyword; the correct room starts
               with the same letter as the keyword.
  - "position": the instruction names an ordinal ("third"); the correct
               room is the n-th in scan order.

Tools: scan() lists rooms (shuffled); inspect() reveals a riddle describing
the convention; read_note(room) returns that room's code; enter_code(code)
tries the keypad. The base solver does not know the conventions, so it must
either get lucky or discover them -- the operational knowledge the memory
bank is meant to capture.
"""

import random

# ---------------------------------------------------------------- constants

CONVENTIONS = ("color", "initial", "position")

COLORS = ["red", "blue", "green", "amber", "violet", "grey"]
ROOM_WORDS = ["storage", "lab", "office", "vault", "archive", "deck",
              "garden", "harbor", "kitchen", "library", "mint", "nexus"]
KEYWORDS = ["ledger", "compass", "lantern", "atlas", "beacon", "cipher"]
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth"}

ACCEPTED = "accepted"
TOO_EASY = "too_easy"
TOO_HARD = "too_hard"


def _code(rng):
    return "%04d" % rng.randrange(10000)


def task_scan_order(task):
    """Canonical per-task scan order. Deterministic in the task id, so the
    'position' convention ('the k-th room as scanned') is well-defined and
    stable across attempts -- every scan() of the same task lists rooms in
    this order."""
    order = task.rooms[:]
    random.Random("daedalus-scan:" + task.id).shuffle(order)
    return order


# ------------------------------------------------------------------ tasks

class Task:
    """A generated vault task: instruction + hidden convention + room layout."""

    def __init__(self, task_id, instruction, convention, rooms, code_room,
                 codes, difficulty):
        self.id = task_id
        self.instruction = instruction
        self.convention = convention      # hidden from the solver
        self.rooms = rooms                # list of room names
        self.code_room = code_room        # room holding the true code
        self.codes = codes                # room -> 4-digit code
        self.difficulty = difficulty      # == number of rooms

    def feature(self):
        """Observable cue the solver may use to pick a relevant heuristic."""
        return self.convention  # instruction visibly names color/keyword/ordinal


def _room_names(rng, n):
    names = set()
    while len(names) < n:
        names.add(rng.choice(COLORS) + "_" + rng.choice(ROOM_WORDS))
    return sorted(names)


def make_task(task_id, convention, n_rooms, rng):
    """Build a task whose hidden convention is `convention`.

    The cue is enforced unique: exactly one room matches the convention, so
    the hidden rule deterministically identifies the code room.
    """
    rooms = _room_names(rng, n_rooms)
    if convention == "color":
        color = rng.choice([c for c in COLORS if any(c in r for r in rooms)])
        code_room = rng.choice([r for r in rooms if color in r])
        # scrub the cue from every other room (recolor, keep word)
        others = [c for c in COLORS if c != color]
        rooms = [r if r == code_room
                 else rng.choice(others) + "_" + r.split("_", 1)[1]
                 for r in rooms]
        rooms = list(dict.fromkeys(rooms))
        while len(rooms) < n_rooms:
            cand = rng.choice(others) + "_" + rng.choice(ROOM_WORDS)
            if cand not in rooms:
                rooms.append(cand)
        instruction = ("Open the vault. The %s seal marks the room whose "
                       "note holds the keypad code." % color)
    elif convention == "initial":
        keyword = rng.choice(KEYWORDS)
        cands = [r for r in rooms if r[0] == keyword[0]]
        if not cands:
            rooms[0] = keyword[0] + rooms[0][1:]
        code_room = rng.choice([r for r in rooms if r[0] == keyword[0]])
        # scrub the cue from every other room (new initial, keep uniqueness)
        fixed, kept = [], False
        for r in rooms:
            if r[0] == keyword[0] and r != code_room:
                alt = rng.choice([w for w in ROOM_WORDS if w[0] != keyword[0]])
                r = rng.choice(COLORS) + "_" + alt
            fixed.append(r)
        rooms = list(dict.fromkeys(fixed))
        while len(rooms) < n_rooms:
            alt = rng.choice([w for w in ROOM_WORDS if w[0] != keyword[0]])
            cand = rng.choice(COLORS) + "_" + alt
            if cand not in rooms:
                rooms.append(cand)
        instruction = ("Open the vault. The '%s' keyword shares its first "
                       "letter with the room whose note holds the keypad "
                       "code." % keyword)
    else:  # position
        k = rng.randrange(1, min(8, n_rooms) + 1)
        tmp = Task(task_id, "", "position", rooms, None, {}, n_rooms)
        code_room = task_scan_order(tmp)[k - 1]
        instruction = ("Open the vault. Count the rooms as they are scanned; "
                       "the %s room's note holds the keypad code."
                       % ORDINALS[k])
    codes = {r: _code(rng) for r in rooms}
    # true code distinct from decoys
    while len(set(codes.values())) < len(rooms):
        codes = {r: _code(rng) for r in rooms}
    return Task(task_id, instruction, convention, rooms, code_room, codes,
                n_rooms)


# --------------------------------------------------------------- environment

class VaultWorld:
    """Deterministic tool environment. Fresh instance per attempt."""

    def __init__(self, task, rng, max_steps=12):
        self.task = task
        self.rng = rng
        self.max_steps = max_steps
        self.steps = 0
        self.opened = False
        self._scan_order = None

    def _bump(self):
        self.steps += 1

    # tools -------------------------------------------------------
    def scan(self):
        self._bump()
        self._scan_order = task_scan_order(self.task)
        return "Rooms: " + ", ".join(self._scan_order)

    def inspect(self):
        self._bump()
        c = self.task.convention
        if c == "color":
            return ("Riddle: a note hides the code. The room you want wears "
                    "the color named in your orders.")
        if c == "initial":
            return ("Riddle: a note hides the code. The room you want begins "
                    "with the same letter as the keyword in your orders.")
        return ("Riddle: a note hides the code. Count the rooms as scanned; "
                "your orders name which count is yours.")

    def read_note(self, room):
        self._bump()
        if room not in self.task.codes:
            return "No such room."
        return "Note in %s: the keypad code is %s." % (room,
                                                       self.task.codes[room])

    def enter_code(self, code):
        self._bump()
        if code == self.task.codes[self.task.code_room]:
            self.opened = True
            return "The vault clicks open."
        return "Wrong code. The keypad buzzes."

    # judge -------------------------------------------------------
    def succeeded(self):
        """Deterministic judge: vault open within the step budget."""
        return self.opened and self.steps <= self.max_steps


# ------------------------------------------------------------- heuristics

class Heuristic:
    """A failure-derived lesson. `convention` is the lesson's subject --
    the scripted stand-in for the text's meaning."""

    def __init__(self, text, convention, source_task_id):
        self.text = text
        self.convention = convention
        self.source_task_id = source_task_id

    def __repr__(self):
        return "Heuristic(%s <- %s)" % (self.convention, self.source_task_id)


_HEURISTIC_TEXT = {
    "color": ("When the orders name a color: inspect first, then read the "
              "note ONLY from the room whose name contains that color, and "
              "enter that code. Never enter a code from an unmatched room."),
    "initial": ("When the orders name a keyword: inspect first, then read "
                "the note ONLY from the room starting with the keyword's "
                "first letter, and enter that code. Double-check the first "
                "letter before reading."),
    "position": ("When the orders name an ordinal: scan, count the rooms in "
                 "the scanned order, read the note ONLY from the room at that "
                 "count, and enter that code. Re-scan rather than guessing."),
}


def _resolve_room(task, convention, scan_order, instruction):
    """Follow a convention to pick a room (models an LLM applying advice)."""
    if convention == "color":
        for color in COLORS:
            if color in instruction:
                cands = [r for r in scan_order if color in r]
                if cands:
                    return cands[0]
    elif convention == "initial":
        for kw in KEYWORDS:
            if kw in instruction:
                cands = [r for r in scan_order if r[0] == kw[0]]
                if cands:
                    return cands[0]
    else:
        for k, word in ORDINALS.items():
            if word in instruction and k <= len(scan_order):
                return scan_order[k - 1]
    return None

# ------------------------------------------------------------------ solver

class Solver:
    """Scripted stand-in for the LLM solver agent.

    Behavior model (documented, not fitted):
      * without heuristics: inspects with prob `p_inspect`; if it inspects,
        parses the convention correctly with prob `p_parse`, else picks a
        random room; reads that room's note and enters the code, retrying up
        to `max_tries` fresh rooms on failure.
      * with heuristics: picks the heuristic whose stated trigger matches
        the task's observable feature (models the LLM reading the bank and
        applying the relevant advice); with prob `p_misapply` picks a wrong
        one. Then inspects, resolves the room via the heuristic's
        convention, reads its note, enters the code; a `p_slip` chance of a
        mechanical slip (typo) remains.
    """

    def __init__(self, seed=0, p_inspect=0.35, p_parse=0.5, max_tries=2,
                 p_slip=0.05, p_misapply=0.05):
        self.rng = random.Random(seed)
        self.p_inspect = p_inspect
        self.p_parse = p_parse
        self.max_tries = max_tries
        self.p_slip = p_slip
        self.p_misapply = p_misapply

    def _pick_heuristic(self, task, heuristics):
        rel = [h for h in heuristics if h.convention == task.feature()]
        if not rel:
            return None
        if self.rng.random() < self.p_misapply:
            others = [h for h in heuristics if h not in rel]
            return self.rng.choice(others) if others else rel[0]
        return rel[0]

    def attempt(self, task, heuristics=()):
        """One attempt from a fresh environment. Returns (success, trace).

        trace is a list of (event, detail) used by the extractor and tests.
        """
        rng = self.rng
        world = VaultWorld(task, random.Random(rng.randrange(1 << 30)))
        trace = [("instruction", task.instruction)]
        scan_out = world.scan()
        rooms = world._scan_order[:]
        trace.append(("scan", "%d rooms" % len(rooms)))

        h = self._pick_heuristic(task, heuristics) if heuristics else None
        if h is not None:
            trace.append(("heuristic", h.convention))
            world.inspect()
            room = _resolve_room(task, h.convention, rooms, task.instruction)
            if room is None:
                trace.append(("resolve_failed", h.convention))
                return False, trace
            note = world.read_note(room)
            trace.append(("read_note", room))
            code = task.codes[room]
            if rng.random() < self.p_slip:
                code = "0000"  # mechanical slip
                trace.append(("slip", room))
            world.enter_code(code)
            trace.append(("enter_code", room))
            return world.succeeded(), trace

        # ---- no-memory base policy ----
        tried = set()
        if rng.random() < self.p_inspect:
            riddle = world.inspect()
            trace.append(("inspect", riddle[:40]))
            if rng.random() < self.p_parse:
                room = _resolve_room(task, task.convention, rooms,
                                     task.instruction)
                trace.append(("parsed", str(room)))
            else:
                room = rng.choice(rooms)
                trace.append(("misread", room))
        else:
            room = rng.choice(rooms)
            trace.append(("no_inspect", room))
        for _ in range(self.max_tries):
            if room in tried:
                rest = [r for r in rooms if r not in tried]
                if not rest:
                    break
                room = rng.choice(rest)
            tried.add(room)
            world.read_note(room)
            trace.append(("read_note", room))
            world.enter_code(task.codes[room])
            trace.append(("enter_code", room))
            if world.succeeded():
                return True, trace
        return False, trace


# ---------------------------------------------------------------- extractor

class Extractor:
    """Scripted stand-in for the paper's LLM extractor agent.

    Derives a heuristic from the instruction + failed trajectory (it never
    sees which success conditions were missed -- here, it never sees the
    hidden convention directly, only the trace). With prob `p_error` it
    misdiagnoses and writes a lesson for the wrong convention, modelling
    extractor imperfection; the acceptance rule is what filters these out.
    """

    def __init__(self, seed=0, p_error=0.25):
        self.rng = random.Random(seed)
        self.p_error = p_error

    def derive(self, task, trace, previous=None):
        rng = self.rng
        events = [e for e, _ in trace]
        if "misread" in events or "resolve_failed" in events:
            truth = task.convention  # misread -> lesson is "check carefully"
        elif "no_inspect" in events:
            truth = task.convention  # skipped recon -> lesson is "inspect first"
        else:
            truth = task.convention
        if rng.random() < self.p_error:
            wrong = [c for c in CONVENTIONS if c != truth]
            convention = rng.choice(wrong)
        else:
            convention = truth
        if previous is not None and previous.convention == convention:
            return previous  # revision converges; keeps the (possibly wrong) lesson
        return Heuristic(_HEURISTIC_TEXT[convention], convention,
                         source_task_id=task.id)


# ------------------------------------------------------------ solver loop

def solver_loop(task, solver, extractor, Ns=3, Nf=6):
    """Paper's inner loop. Returns (outcome, heuristic-or-None).

    ACCEPTED: solver failed at least once, then reached Ns consecutive
        successes with the heuristic in context.
    TOO_EASY: solver never failed (Ns straight successes, no heuristic).
    TOO_HARD: Nf total failures.
    """
    heuristic = None
    consec, fails = 0, 0
    failed_once = False
    while True:
        ok, trace = solver.attempt(task, [heuristic] if heuristic else [])
        if ok:
            consec += 1
            if consec >= Ns:
                if failed_once:
                    return ACCEPTED, heuristic
                return TOO_EASY, None
        else:
            failed_once = True
            consec, fails = 0, fails + 1
            heuristic = extractor.derive(task, trace, heuristic)
            if fails >= Nf:
                return TOO_HARD, None


# --------------------------------------------------------------- explorer

class Explorer:
    """Scripted stand-in for the paper's explorer agent.

    Proposes tasks at a difficulty (number of rooms), verifies feasibility
    itself (it knows the hidden convention, so it can always solve -- the
    scripted analogue of "completes it itself to establish feasibility"),
    and refines difficulty from the solver-loop outcome, up to Nr
    refinements per session.
    """

    def __init__(self, seed=0, min_rooms=4, max_rooms=10, step=2):
        self.rng = random.Random(seed)
        self.min_rooms = min_rooms
        self.max_rooms = max_rooms
        self.step = step
        self._n = 0

    def propose(self, difficulty):
        convention = self.rng.choice(CONVENTIONS)
        self._n += 1
        return make_task("gen-%d" % self._n, convention, difficulty, self.rng)

    def feasible(self, task):
        # Explorer knows the convention -> can always solve within budget.
        return True

    def refine(self, task, outcome):
        d = task.difficulty
        if outcome == TOO_EASY:
            d = min(self.max_rooms, d + self.step)
        elif outcome == TOO_HARD:
            d = max(self.min_rooms, d - self.step)
        self._n += 1
        return make_task("gen-%d" % self._n,
                         self.rng.choice(CONVENTIONS), d, self.rng)


# ------------------------------------------------------------ memory bank

class MemoryBank:
    """Accepted heuristics + consolidation (paper's consolidator)."""

    def __init__(self):
        self.accepted = []       # list[Heuristic]
        self.consolidated = None

    def accept(self, heuristic):
        self.accepted.append(heuristic)

    def consolidate(self):
        """Merge accepted heuristics, dropping redundant ones.

        Scripted analogue of the paper's single-LLM-call consolidator:
        heuristics teaching the same convention are redundant -- keep the
        earliest, preserving one actionable lesson per convention.
        Returns the frozen bank (list[Heuristic]).
        """
        seen, bank = set(), []
        for h in self.accepted:
            if h.convention not in seen:
                seen.add(h.convention)
                bank.append(h)
        self.consolidated = bank
        return bank

    def acceptance_rate(self, n_sessions):
        return len(self.accepted) / max(1, n_sessions)


# ------------------------------------------------------------ full loop

def run_daedalus(n_sessions=12, seed=0, Ns=3, Nf=6, Nr=3, start_difficulty=6,
                 solver_kw=None, extractor_kw=None):
    """Run the paper's outer loop for n_sessions. Returns (bank, log)."""
    rng = random.Random(seed)
    explorer = Explorer(seed=rng.randrange(1 << 30))
    bank = MemoryBank()
    log = []
    difficulty = start_difficulty
    for s in range(n_sessions):
        task = explorer.propose(difficulty)
        assert explorer.feasible(task)
        outcome, heuristic = None, None
        for _ in range(Nr):
            solver = Solver(seed=rng.randrange(1 << 30),
                            **(solver_kw or {}))
            extractor = Extractor(seed=rng.randrange(1 << 30),
                                  **(extractor_kw or {}))
            outcome, heuristic = solver_loop(task, solver, extractor,
                                             Ns=Ns, Nf=Nf)
            if outcome == ACCEPTED:
                bank.accept(heuristic)
                break
            task = explorer.refine(task, outcome)
        difficulty = task.difficulty
        log.append({"session": s, "outcome": outcome,
                    "heuristic": (heuristic.convention if heuristic else None),
                    "difficulty": difficulty})
    bank.consolidate()
    return bank, log


def evaluate(tasks, solver_seed=0, heuristics=(), n_rollouts=5,
             solver_kw=None):
    """Test-time evaluation. Returns dict with success_rate and pass^k."""
    succ, passk = 0, 0
    total = 0
    for i, task in enumerate(tasks):
        solver = Solver(seed=solver_seed * 1000 + i, **(solver_kw or {}))
        wins = 0
        for _ in range(n_rollouts):
            ok, _ = solver.attempt(task, heuristics)
            wins += ok and 1
            succ += ok and 1
            total += 1
        if wins == n_rollouts:
            passk += 1
    n = len(tasks)
    return {"n_tasks": n, "n_rollouts": n_rollouts,
            "success_rate": succ / total,
            "pass_%d" % n_rollouts: passk / n}
