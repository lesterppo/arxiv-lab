"""arXiv 2610.10590 — Agent-Controlled Forgetting for Tool-Using Agents:
Reversible Context Curation in Practice.

Core claim implemented here: an agent harness can let the *acting model*
select previously observed tool results, replace each with a short note at
its original position, and retain the exact original in a recoverable
archive. Forgetting is eviction from the active request — not deletion from
storage, not unlearning, not a privacy mechanism. A recoverable archive plus
an explicit recovery call makes the context curation *reversible*.

Mechanism (paper Sec. 3.1-3.2):
- State: ordered message sequence H (user/system/assistant/tool-result),
  plus an external archive A mapping stable archive refs -> originals.
- context_apply(batch of (result_id, note)): batch-validated (all-or-nothing);
  originals staged in A first, then stubs (note + archive ref) committed at
  the selected result locations. User/system/assistant messages are not
  selectable. Result IDs are stable, never reused within a session.
- context_recover(archive_ref): appends the stored original as a NEW tool
  result with a fresh result ID at the current end of H. Does not rerun the
  tool; the original stub remains addressable.

The experiment in tests/test_context_forget_cpu.py replays the paper's
experimental design in simulation: noisy tool-use trajectories (large
payloads, small useful content) versus dense ones, comparing retained
history against reversible forgetting on cumulative prompt tokens, recovery
rate, task-oracle success, and workload dependence.
"""

TOKENS_PER_CHAR = 1 / 4.0  # deterministic token estimate for accounting


def estimate_tokens(text):
    """Deterministic token estimate for context accounting."""
    return int(len(text) * TOKENS_PER_CHAR)


class ForgettingError(Exception):
    """Batch-validation or protocol error (no partial mutation committed)."""


class ToolResult(object):
    __slots__ = ("result_id", "call", "body", "archived", "archive_ref",
                 "provenance")

    def __init__(self, result_id, call, body, provenance=None):
        self.result_id = result_id
        self.call = call
        self.body = body
        self.archived = False
        self.archive_ref = None
        self.provenance = provenance or {}


class Message(object):
    __slots__ = ("kind", "text", "tool_result")

    def __init__(self, kind, text="", tool_result=None):
        assert kind in ("user", "system", "assistant", "tool_result")
        self.kind = kind
        self.text = text
        self.tool_result = tool_result


def stub_text(note, archive_ref):
    return "[archived %s] %s" % (archive_ref, note)


class ForgettingHarness(object):
    """Ordered conversation H + external archive A (paper Sec. 3.1)."""

    def __init__(self):
        self.messages = []
        self.archive = {}
        self._next_result = 1
        self._next_archive = 1
        self._by_result = {}

    # -- conversation construction -------------------------------------
    def add(self, kind, text=""):
        msg = Message(kind, text)
        self.messages.append(msg)
        return msg

    def add_tool_result(self, call, body, provenance=None):
        rid = "r%d" % self._next_result
        self._next_result += 1
        tr = ToolResult(rid, call, body, provenance)
        self.messages.append(Message("tool_result", tool_result=tr))
        self._by_result[rid] = tr
        return rid

    # -- paper Sec. 3.1: context_apply / context_recover ----------------
    def context_apply(self, batch):
        """Apply a batch of (result_id, note) archival operations.

        All-or-nothing: the whole batch is validated first; any unknown,
        duplicate, already-archived, or non-tool identifier raises
        ForgettingError and commits nothing.
        """
        pairs = list(batch)
        seen = set()
        for rid, note in pairs:
            if rid in seen:
                raise ForgettingError("duplicate result id in batch: %s" % rid)
            seen.add(rid)
            tr = self._by_result.get(rid)
            if tr is None:
                raise ForgettingError("unknown result id: %s" % rid)
            if tr.archived:
                raise ForgettingError("already archived: %s" % rid)
        # Stage originals in the archive BEFORE committing stubs.
        staged = []
        for rid, note in pairs:
            tr = self._by_result[rid]
            ref = "a%d" % self._next_archive
            self._next_archive += 1
            staged.append((ref, tr, note))
        mapping = {}
        for ref, tr, note in staged:
            self.archive[ref] = {
                "result_id": tr.result_id,
                "call": tr.call,
                "text": tr.body,
                "provenance": dict(tr.provenance),
            }
            tr.body = stub_text(note, ref)
            tr.archived = True
            tr.archive_ref = ref
            mapping[tr.result_id] = ref
        return mapping

    def context_recover(self, archive_ref):
        """Append the archived original as a new tool result (fresh id).

        Recovery does not rerun the original tool: the payload is evidence
        of the earlier observation, not a fresh measurement.
        """
        entry = self.archive.get(archive_ref)
        if entry is None:
            raise ForgettingError("unknown archive ref: %s" % archive_ref)
        rid = self.add_tool_result(
            entry["call"], entry["text"], dict(entry["provenance"]))
        return rid

    # -- accounting ------------------------------------------------------
    def active_body(self, msg):
        if msg.kind == "tool_result":
            tr = msg.tool_result
            return "[%s %s] %s" % (tr.call, tr.result_id, tr.body)
        return msg.text

    def prompt_tokens(self):
        return sum(estimate_tokens(self.active_body(m))
                   for m in self.messages)

    def archived_refs(self):
        return sorted(self.archive.keys())


# ----------------------------------------------------------------------
# Simulation layer: the paper's experimental design, replayed in silico.
# ----------------------------------------------------------------------

def make_noisy_observation(rng, idx, payload_lines=120, signal_lines=3):
    """Tool observation: big noisy payload, tiny useful content.

    Each line is either signal (a KEY: value fact) or filler noise.
    Returns (full_text, signal_facts) where signal_facts is the list of
    exact facts the note can preserve.
    """
    facts = []
    lines = []
    sig_idx = set(rng.sample(range(payload_lines), signal_lines))
    for i in range(payload_lines):
        if i in sig_idx:
            fact = "FACT-%d-%d = value_%04d" % (idx, i, rng.randrange(10000))
            facts.append(fact)
            lines.append(">> %s <<" % fact)
        else:
            lines.append("noise log line %d: 0x%08x status ok retry %d" %
                         (i, rng.randrange(1 << 32), rng.randrange(5)))
    return "\n".join(lines), facts


def make_dense_observation(rng, idx, payload_lines=120):
    """Dense payload: nearly every line is signal (no cheap compression)."""
    lines = []
    for i in range(payload_lines):
        lines.append("STEP %d.%d: action=deploy target=node-%d param=%d" %
                     (idx, i, rng.randrange(64), rng.randrange(10000)))
    return "\n".join(lines), list(lines)


def extract_note(full_text, signal_facts, keep=None):
    """Simulated note policy: keep the signal facts as the note.

    `keep` models lossy extraction (paper Sec. 3.1: "The active
    representation is lossy because a note need not preserve every
    detail"). With keep < len(facts), later recall of a dropped fact
    requires recovery — the reversibility test.
    """
    kept = signal_facts if keep is None else signal_facts[:keep]
    return "signal: " + " | ".join(kept)


class TrajectoryReplay(object):
    """Replays a fixed tool-use trajectory under a context policy.

    Records per-step prompt tokens, recovery events, and answers to
    recall probes. Deterministic given the seed.
    """

    def __init__(self, seed, n_obs=24, policy=None, workload="noisy",
                 harness_cls=None):
        self.rng_seed = seed
        self.n_obs = n_obs
        self.policy = policy          # None -> retained history
        self.workload = workload
        self.harness_cls = harness_cls or ForgettingHarness
        self.requests = 0             # provider requests made
        self.recoveries = 0
        self.prompt_token_trace = []

    def run(self):
        import random
        rng = random.Random(self.rng_seed)
        h = self.harness_cls()
        h.add("system", "You are a debugging agent. Extract facts; keep going.")
        h.add("user", "Debug the failing deployment across %d services." % self.n_obs)
        all_facts = {}        # result_id -> signal facts
        probe_answers = []    # (probe_text, expected_fact or None-if-lost)
        self.requests = 2

        for i in range(self.n_obs):
            if self.workload == "noisy":
                body, facts = make_noisy_observation(rng, i)
            else:
                body, facts = make_dense_observation(rng, i)
            rid = h.add_tool_result("kubectl_logs", body,
                                    {"service": i, "seed": self.rng_seed})
            all_facts[rid] = facts
            h.add("assistant", "Reviewed logs for service %d." % i)
            self.requests += 1

            # Policy: archive once the useful content is extracted into a
            # note. The note is lossy (keeps all-but-one fact), so later
            # recall of the dropped fact exercises recovery.
            if self.policy == "forget" and facts:
                ratio = len("\n".join(facts)) / max(1, len(body))
                if ratio < 0.15:      # noisy enough to be worth forgetting
                    note = extract_note(body, facts, keep=len(facts) - 1)
                    h.context_apply([(rid, note)])
                    self.requests += 1  # context-management call

            # Every 6th observation, probe recall of an earlier fact. The
            # probe asks for the fact the note dropped, so the forget arm
            # must recover; the irreversible arm cannot.
            if i > 0 and i % 6 == 0:
                probe_rid = "r%d" % (i - 5)
                if probe_rid in all_facts and all_facts[probe_rid]:
                    want = all_facts[probe_rid][-1]
                    got = self._recall(h, probe_rid, want)
                    probe_answers.append((probe_rid, want, got))
            self.prompt_token_trace.append(h.prompt_tokens())

        total = sum(self.prompt_token_trace)
        ok = all(want == got for _, want, got in probe_answers)
        return {
            "cumulative_prompt_tokens": total,
            "final_prompt_tokens": h.prompt_tokens(),
            "requests": self.requests,
            "recoveries": self.recoveries,
            "probes": len(probe_answers),
            "probes_ok": sum(1 for _, w, g in probe_answers if w == g),
            "all_probes_ok": ok,
            "archived": len(h.archived_refs()),
        }

    def _recall(self, h, rid, want):
        """Scripted recall: answer from the note; recover if missing.

        The irreversible ablation cannot recover: context_recover raises
        and the probe fails, isolating the value of reversibility.
        """
        tr = h._by_result[rid]
        if self.policy != "forget" or not tr.archived:
            return want if want in tr.body else None
        if want in tr.body:          # note preserved the fact
            return want
        # Note was lossy: reversible recovery restores the original.
        try:
            new_rid = h.context_recover(tr.archive_ref)
        except ForgettingError:
            return None
        self.recoveries += 1
        self.requests += 1
        return want if want in h._by_result[new_rid].body else None


class IrreversibleHarness(ForgettingHarness):
    """Same as ForgettingHarness but recovery is impossible (ablation)."""

    def context_recover(self, archive_ref):
        raise ForgettingError("archive unavailable: irreversible truncation")
