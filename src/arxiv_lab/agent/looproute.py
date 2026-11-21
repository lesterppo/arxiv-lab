"""LoopRoute: loop-triggered routing for System-1 decision agents.

Reproduces the core mechanism of arXiv:2611.25363 ("Stop Going in Circles:
Loop-Triggered Routing for System 1 Decision Agents", Yu et al., 2026):

* A fast decision policy acts by default.
* Recurring (state, action) pairs signal stalled execution.
* When a loop is detected, a slow assistant (here: a BFS oracle planner, or
  a real LLM via NVIDIA NIM in the model-backed leg) provides brief,
  feedback-driven assistance, then control returns to the fast policy.
* Later loops can request further assistance. No training, no confidence
  calibration.

This module is stdlib-only and deterministic. The environment is
KitchenWorld, a small TextWorld-cooking-like task family: collect the recipe
ingredients (one inside a locked chest that needs a key found elsewhere),
then cook in the kitchen. The fast policy is a greedy heuristic with one
realistic System-1 flaw: when the next needed ingredient sits in a locked
container and it does not hold the key, it keeps trying to open the
container instead of going to fetch the key -- a genuine state-action loop.
"""

from __future__ import annotations

import random
from collections import deque

# ---------------------------------------------------------------------------
# Environment: KitchenWorld
# ---------------------------------------------------------------------------

ROOMS = ("kitchen", "pantry", "garden", "cellar", "bedroom", "hallway")

# Undirected room graph.
ADJACENCY = {
    "kitchen": ("pantry", "hallway", "garden"),
    "pantry": ("kitchen", "cellar"),
    "garden": ("kitchen", "hallway"),
    "cellar": ("pantry", "bedroom"),
    "bedroom": ("cellar", "hallway"),
    "hallway": ("kitchen", "garden", "bedroom"),
}

INGREDIENTS = (
    "tomato", "onion", "garlic", "pepper", "carrot",
    "potato", "basil", "lemon", "mushroom", "chili",
)

CONTAINER = "chest"
KEY = "brass key"


def bfs_path(start, goal, adjacency=ADJACENCY):
    """Shortest room path (list of rooms) from start to goal, inclusive."""
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adjacency[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                if nxt == goal:
                    path = [goal]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                q.append(nxt)
    return None


class KitchenWorld:
    """Deterministic cooking task.

    State (hashable, used as the router's loop key):
        (room, inventory frozenset, collected frozenset,
         chest_open bool, key_taken bool, taken frozenset)

    Actions are strings: "go <room>", "take <item>", "open chest", "cook".
    ``step`` returns (observation, reward, done, info); invalid actions
    leave the state unchanged and report ok=False.
    """

    def __init__(self, seed, n_ingredients=3, locked=True):
        rng = random.Random(seed)
        self.seed = seed
        self.recipe = tuple(rng.sample(INGREDIENTS, n_ingredients))
        # Place ingredients in rooms; one (the last) goes in the locked chest.
        rooms = [r for r in ROOMS if r != "kitchen"]
        rng.shuffle(rooms)
        self.placement = {}
        for i, ing in enumerate(self.recipe):
            if locked and i == len(self.recipe) - 1:
                self.placement[ing] = ("chest", "kitchen")
            else:
                self.placement[ing] = (ing, rooms[i % len(rooms)])
        # Key lives in a non-kitchen room that holds no recipe ingredient.
        key_room = rng.choice([r for r in rooms if r not in
                               {p[1] for p in self.placement.values()}]
                              or ["bedroom"])
        self.key_room = key_room
        self.locked = locked
        self.reset()

    def reset(self):
        self.room = "kitchen"
        self.inventory = set()
        self.collected = set()
        self.chest_open = False
        self.key_taken = False
        self.taken = set()
        self.steps = 0
        return self.state()

    def state(self):
        return (self.room, frozenset(self.inventory), frozenset(self.collected),
                self.chest_open, self.key_taken, frozenset(self.taken))

    def describe(self):
        """Human-readable state for the LLM assistant prompt."""
        here = [ing for ing, (_, room) in self.placement.items()
                if room == self.room and ing not in self.collected]
        chest_here = self.room == "kitchen" and self.locked
        lines = [f"You are in the {self.room}.",
                 f"Recipe (in order): {', '.join(self.recipe)}.",
                 f"Collected: {', '.join(sorted(self.collected)) or 'none'}.",
                 f"Carrying: {', '.join(sorted(self.inventory)) or 'nothing'}."]
        if here:
            lines.append(f"Visible here: {', '.join(here)}.")
        if chest_here:
            lines.append(f"A locked chest is here. "
                         f"{'It is open.' if self.chest_open else 'It is locked; it needs the brass key.'}")
        if self.room == self.key_room and not self.key_taken:
            lines.append("A brass key is visible here.")
        return " ".join(lines)

    def _item_room(self, item):
        return self.placement[item][1]

    def legal_actions(self):
        acts = []
        for nxt in ADJACENCY[self.room]:
            acts.append(f"go {nxt}")
        for ing, (_, room) in self.placement.items():
            if room == self.room and ing not in self.collected:
                if ing == self.recipe[-1] and self.locked and not self.chest_open:
                    continue  # inside the locked chest
                acts.append(f"take {ing}")
        if self.room == "kitchen" and self.locked and not self.chest_open:
            acts.append("open chest")
        if self.room == self.key_room and not self.key_taken:
            acts.append(f"take {KEY}")
        if self.room == "kitchen" and set(self.recipe) <= self.collected:
            acts.append("cook")
        return acts

    def step(self, action):
        self.steps += 1
        parts = action.split(" ", 1)
        verb = parts[0]
        arg = parts[1] if len(parts) > 1 else ""
        if verb == "go" and arg in ADJACENCY[self.room]:
            self.room = arg
            return f"You go to the {arg}.", 0, False, {"ok": True}
        if verb == "take":
            if arg == KEY and self.room == self.key_room and not self.key_taken:
                self.key_taken = True
                self.inventory.add(KEY)
                return "You take the brass key.", 0, False, {"ok": True}
            if (arg in self.placement and self._item_room(arg) == self.room
                    and arg not in self.collected):
                if arg == self.recipe[-1] and self.locked and not self.chest_open:
                    return "The chest is locked.", 0, False, {"ok": False}
                self.collected.add(arg)
                self.taken.add(arg)
                self.inventory.add(arg)
                return f"You take the {arg}.", 0, False, {"ok": True}
            return "You cannot take that here.", 0, False, {"ok": False}
        if verb == "open" and arg == CONTAINER:
            if self.room != "kitchen":
                return "There is no chest here.", 0, False, {"ok": False}
            if self.chest_open:
                return "The chest is already open.", 0, False, {"ok": False}
            if KEY not in self.inventory:
                return "The chest is locked. You need the brass key.", 0, False, {"ok": False}
            self.chest_open = True
            return "You unlock and open the chest.", 0, False, {"ok": True}
        if verb == "cook":
            if self.room == "kitchen" and set(self.recipe) <= self.collected:
                return "You cook the meal. Task complete!", 1, True, {"ok": True}
            return "You cannot cook yet.", 0, False, {"ok": False}
        return "Unknown action.", 0, False, {"ok": False}


def plan_to_goal(env):
    """BFS over the full state space from env.state() to a cooked meal.

    Returns a list of actions, or [] if unreachable. Used as the
    deterministic stand-in for the paper's LLM assistant.
    """
    start = env.state()
    # Rebuild a fresh env with the same task to search from `start`.
    probe = KitchenWorld(env.seed, len(env.recipe), env.locked)
    probe.room, inv, coll, chest, key, taken = start
    probe.inventory = set(inv)
    probe.collected = set(coll)
    probe.chest_open = chest
    probe.key_taken = key
    probe.taken = set(taken)
    prev = {start: (None, None)}
    q = deque([start])
    goal_state = None
    goal_action = None
    while q:
        cur = q.popleft()
        probe.room, inv, coll, chest, key, taken = cur
        probe.inventory, probe.collected = set(inv), set(coll)
        probe.chest_open, probe.key_taken, probe.taken = chest, key, set(taken)
        for act in probe.legal_actions():
            probe.room, inv, coll, chest, key, taken = cur
            probe.inventory, probe.collected = set(inv), set(coll)
            probe.chest_open, probe.key_taken, probe.taken = chest, key, set(taken)
            probe.steps = 0
            _, _, done, info = probe.step(act)
            ns = probe.state()
            if done:
                # NB: cooking does not change the state, so ns == cur and
                # is already visited -- the done check must come first,
                # and the goal edge (cur, act) must be recorded explicitly
                # or the reconstructed plan would omit "cook" itself.
                goal_state, goal_action = cur, act
                q = None
                break
            if ns in prev:
                continue
            prev[ns] = (cur, act)
            q.append(ns)
        if goal_state is not None:
            break
    if goal_state is None:
        return []
    plan, cur = [goal_action], goal_state
    while prev[cur][0] is not None:
        plan.append(prev[cur][1])
        cur = prev[cur][0]
    return list(reversed(plan))


# ---------------------------------------------------------------------------
# Fast policy (System-1 stand-in) and the LoopRoute router
# ---------------------------------------------------------------------------

class FastPolicy:
    """Greedy heuristic with a realistic System-1 flaw.

    Navigates to the next needed ingredient and takes it; but when that
    ingredient is inside the locked chest and the key is not held, it keeps
    trying ``open chest`` instead of detouring to fetch the key. The
    (state, "open chest") pair therefore recurs -- the loop LoopRoute
    is designed to catch.
    """

    def act(self, env):
        if set(env.recipe) <= env.collected:
            if env.room == "kitchen":
                return "cook"
            path = bfs_path(env.room, "kitchen")
            return f"go {path[1]}"
        # Next needed ingredient in recipe order.
        nxt = next(i for i in env.recipe if i not in env.collected)
        if nxt == env.recipe[-1] and env.locked and not env.chest_open:
            if env.room == "kitchen":
                return "open chest"  # the flaw: no key check
            path = bfs_path(env.room, "kitchen")
            return f"go {path[1]}"
        room = env._item_room(nxt)
        if env.room == room:
            return f"take {nxt}"
        path = bfs_path(env.room, room)
        return f"go {path[1]}"


class LoopRouter:
    """LoopRoute router (paper §3, as specified in the abstract).

    Counts (state, action) pairs within an episode. When the fast policy
    proposes a pair for the ``threshold``-th time, control passes to the
    assistant, which returns a short plan; the plan is executed verbatim,
    then control returns to the fast policy. After an assist, that pair's
    count resets, so a later recurrence can trigger assistance again.
    """

    def __init__(self, threshold=2):
        self.threshold = threshold
        self.counts = {}
        self.assists = 0

    def reset(self):
        self.counts = {}
        self.assists = 0

    def should_assist(self, state, action):
        """True when this (state, action) pair has recurred to threshold."""
        return self.counts.get((state, action), 0) >= self.threshold - 1

    def note(self, state, action):
        self.counts[(state, action)] = self.counts.get((state, action), 0) + 1

    def assist(self, state, action, plan):
        """Record an assist for (state, action); reset its count."""
        self.assists += 1
        self.counts.pop((state, action), None)
        return plan


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, policy, router=None, assistant=None, max_steps=60):
    """Run one task.

    assistant: callable(env) -> list[str] of plan actions (or None).
    Returns dict with success, steps, assists, llm_calls, trajectory.
    """
    env.reset()
    if router is not None:
        router.reset()
    traj = []
    plan = []
    llm_calls = 0
    while env.steps < max_steps:
        state = env.state()
        if plan:
            action = plan.pop(0)
            source = "plan"
        else:
            action = policy.act(env)
            source = "fast"
            if router is not None and router.should_assist(state, action):
                if assistant is not None:
                    llm_calls += 1
                    plan = router.assist(state, action, assistant(env))
                    if plan:
                        action = plan.pop(0)
                        source = "assist"
                    else:
                        # Assistant produced nothing: count the attempt,
                        # execute the fast action once and keep counting.
                        router.note(state, action)
                        obs, _, done, _ = env.step(action)
                        traj.append((state, action, source, obs))
                        if done:
                            return _result(env, router, llm_calls, traj, True)
                        continue
        router.note(state, action) if router is not None else None
        obs, _, done, _ = env.step(action)
        traj.append((state, action, source, obs))
        if done:
            return _result(env, router, llm_calls, traj, True)
    return _result(env, router, llm_calls, traj, False)


def _result(env, router, llm_calls, traj, success):
    return {
        "success": success,
        "steps": env.steps,
        "assists": router.assists if router is not None else 0,
        "llm_calls": llm_calls,
        "trajectory": traj,
    }


def oracle_assistant(env):
    """Deterministic BFS plan from the current state to the goal."""
    return plan_to_goal(env)


def run_experiment(seeds, threshold=2, use_router=True):
    """Fast-only vs LoopRoute-oracle over seeded tasks.

    Task mix: even seeds have the locked chest (the fast policy loops on
    them); odd seeds are unlocked (the fast policy solves them unaided).
    A non-degenerate baseline: fast-only should win the unlocked half and
    lose the locked half; LoopRoute should rescue the locked half.
    """
    policy = FastPolicy()
    fast, routed = [], []
    for s in seeds:
        locked = (s % 2 == 0)
        env = KitchenWorld(s, locked=locked)
        fast.append(run_episode(env, policy, router=None))
        router = LoopRouter(threshold=threshold)
        routed.append(run_episode(KitchenWorld(s, locked=locked), policy,
                                  router=router, assistant=oracle_assistant))
    def agg(rs):
        n = len(rs)
        return {
            "n": n,
            "success_rate": sum(r["success"] for r in rs) / n,
            "mean_steps": sum(r["steps"] for r in rs) / n,
            "mean_assists": sum(r["assists"] for r in rs) / n,
            "mean_llm_calls": sum(r["llm_calls"] for r in rs) / n,
        }
    return {"fast_only": agg(fast), "looproute_oracle": agg(routed)}
