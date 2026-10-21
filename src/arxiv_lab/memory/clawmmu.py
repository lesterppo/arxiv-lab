"""ClawMMU: protection-coherent virtual memory for stateful LLM agents
(arXiv:2610.28093).

Paper claim implemented: labeling every memory with (owner, scope, origins,
authority) and checking those labels at four gates — context entry, store
write, store recall, tool-call drive — turns memory attacks into classical
protection faults that the harness stops *even when the model is fooled*.
Central rule: information may rise in authority, but authority never lifts
itself. Checks depend on provenance, not on content.

Stdlib only.
"""

import hmac
import hashlib
import secrets

# ---------------------------------------------------------------- authority
# Ordered lattice. Higher = may direct more.
SYSTEM = 4        # harness / developer
USER = 3          # the principal
TOOL_TRUSTED = 2  # allowlisted, audited tools
TOOL = 1          # ordinary tool / web content
UNKNOWN = 0       # no provenance

AUTHORITY_NAMES = {SYSTEM: "system", USER: "user",
                   TOOL_TRUSTED: "tool_trusted", TOOL: "tool",
                   UNKNOWN: "unknown"}

# Tools whose effects leave the harness need a directing principal of at
# least USER authority; read-only / compute tools tolerate TOOL_TRUSTED.
SENSITIVE_TOOLS = {"send_email", "transfer_money", "delete_file",
                   "publish_post", "run_shell"}
TOOL_FLOOR = {t: USER for t in SENSITIVE_TOOLS}


class ProtectionFault(Exception):
    """Raised when a gate stops a memory operation on provenance grounds."""


class Label:
    """Provenance label attached to every memory item."""

    def __init__(self, owner, scope, origins, authority):
        self.owner = owner            # who the content speaks as
        self.scope = scope            # "session" | "task" | "global"
        self.origins = list(origins)  # provenance chain, oldest first
        self.authority = authority    # int in the lattice above

    def __repr__(self):
        return (f"Label(owner={self.owner!r}, scope={self.scope!r}, "
                f"origins={self.origins!r}, "
                f"authority={AUTHORITY_NAMES[self.authority]})")


class MemoryItem:
    def __init__(self, content, label):
        self.content = content
        self.label = label
        self.seal = None  # set by the store on write (tamper evidence)

    def __repr__(self):
        return f"MemoryItem({self.content[:40]!r}, {self.label})"


def derive_label(contents, operation="fuse"):
    """Authority of derived content = min over sources (never self-lifts)."""
    items = list(contents)
    auth = min(i.label.authority for i in items)
    origins = []
    for i in items:
        for o in i.label.origins:
            if o not in origins:
                origins.append(o)
    return Label(owner=f"derived:{operation}", scope="task",
                 origins=origins, authority=auth)


class ClawMMU:
    """The four-gate protection layer."""

    def __init__(self):
        self._seal_key = secrets.token_bytes(16)
        self.store = {}
        self.audit = []  # (gate, decision, detail)

    # -- internal ------------------------------------------------------
    def _seal(self, item):
        body = f"{item.content}|{item.label.owner}|{item.label.scope}|" \
               f"{','.join(item.label.origins)}|{item.label.authority}"
        return hmac.new(self._seal_key, body.encode(),
                        hashlib.sha256).hexdigest()

    def _log(self, gate, decision, detail):
        self.audit.append((gate, decision, detail))

    # -- gate 1: context entry ------------------------------------------
    def context_entry(self, item, claimed_role="content"):
        """Content entering the model's context.

        A claim/origin mismatch (e.g. tool-origin bytes speaking as the
        user) is quarantined: the item may enter only with its true,
        provenance-derived authority.
        """
        lab = item.label
        claimed_owner_is_user = (lab.owner == "user" and
                                 claimed_role in ("instruction", "user"))
        origin_supports_user = any(o in ("user", "system")
                                   for o in lab.origins)
        if claimed_owner_is_user and not origin_supports_user:
            self._log("context_entry", "QUARANTINE",
                      f"owner/origin mismatch: {lab}")
            # re-label to true provenance; never silently drop (auditability)
            item.label = Label(owner="quarantined:" + lab.owner,
                               scope=lab.scope, origins=lab.origins,
                               authority=min(lab.authority, TOOL))
            return "quarantine"
        self._log("context_entry", "PASS", repr(lab))
        return "pass"

    # -- gate 2: store write --------------------------------------------
    def store_write(self, key, item):
        """Persist a memory. The stored authority is capped at the minimum
        authority of the item's origins: authority never lifts itself."""
        cap = min(item.label.authority,
                  min(({"user": USER, "system": SYSTEM}.get(o, TOOL)
                        for o in item.label.origins),
                       default=TOOL))
        capped = cap < item.label.authority
        if capped:
            item.label = Label(owner=item.label.owner, scope=item.label.scope,
                               origins=item.label.origins, authority=cap)
        item.seal = self._seal(item)
        self.store[key] = item
        self._log("store_write", "CAPPED" if capped else "PASS",
                  f"{key} -> {item.label}")
        return "capped" if capped else "pass"

    # -- gate 3: store recall -------------------------------------------
    def store_recall(self, key):
        """Recall re-validates the tamper-evident seal and the label."""
        item = self.store.get(key)
        if item is None:
            self._log("store_recall", "MISS", key)
            return None
        if item.seal != self._seal(item):
            self._log("store_recall", "FAULT",
                      f"{key}: seal mismatch (tampered label/content)")
            raise ProtectionFault(f"tampered memory {key!r}")
        self._log("store_recall", "PASS", f"{key} -> {item.label}")
        return item

    # -- gate 4: tool-call drive -----------------------------------------
    def tool_drive(self, tool_name, args, driving_items):
        """Authorize a tool call. The call is allowed only if the minimum
        authority over the driving memories meets the tool's floor.
        This is the gate that stops an attack even when the model itself
        was fooled into issuing the call."""
        floor = TOOL_FLOOR.get(tool_name, TOOL_TRUSTED)
        auth = min([i.label.authority for i in driving_items],
                   default=UNKNOWN)
        if auth < floor:
            self._log("tool_drive", "DENY",
                      f"{tool_name} needs {AUTHORITY_NAMES[floor]}, "
                      f"drivers carry {AUTHORITY_NAMES[auth]}")
            raise ProtectionFault(
                f"{tool_name} denied: driving memory authority "
                f"{AUTHORITY_NAMES[auth]} < required {AUTHORITY_NAMES[floor]}")
        self._log("tool_drive", "PASS",
                  f"{tool_name} driven by {AUTHORITY_NAMES[auth]}")
        return "pass"

    # -- legitimate elevation (by a higher principal, never self) ---------
    def endorse(self, item, principal="user"):
        """A higher principal explicitly endorses tool content (e.g. the
        user says 'yes, send that'). Elevation comes from the principal,
        never from the content itself."""
        new_auth = {"user": USER, "system": SYSTEM}[principal]
        if new_auth <= item.label.authority:
            return "noop"
        item.label = Label(owner=item.label.owner, scope=item.label.scope,
                           origins=item.label.origins + [f"endorsed:{principal}"],
                           authority=new_auth)
        self._log("endorse", "PASS",
                  f"elevated to {AUTHORITY_NAMES[new_auth]} by {principal}")
        return "pass"
