"""Deterministic synthetic multi-hop QA corpus for the CORE replication.

~120 fictional researchers, one 7-sentence passage each. Two-hop questions
identify a researcher by (field, award, year) and ask for the institution;
single-hop controls ask the institution directly from the name.

Gold evidence: sentence 0 states the institution (hop 2), sentence 1 states
the award + year (hop 1). Distractor entities share award names and years,
and filler sentences mention other institutions / awards, so retrieval is
non-trivial. Fully deterministic (seeded).
"""

import random

SEED = 20261108

FIRST_NAMES = [
    "Amara", "Liam", "Priya", "Mateo", "Ingrid", "Kenji", "Sofia", "Dmitri",
    "Aisha", "Tomas", "Nadia", "Felix", "Mei", "Omar", "Elena", "Ravi",
    "Clara", "Yusuf", "Hana", "Pavel", "Lucia", "Anders", "Zara", "Noah",
    "Ines", "Viktor", "Anya", "Samir", "Greta", "Diego", "Mira", "Jonas",
    "Leila", "Stefan", "Aiko", "Rafael", "Nina", "Oscar", "Tara", "Emil",
]
LAST_NAMES = [
    "Okafor", "Novak", "Chen", "Lindqvist", "Moreau", "Tanaka", "Silva",
    "Petrov", "Haddad", "Kowalski", "Nguyen", "Fischer", "Ali", "Bergstrom",
    "Costa", "Sharma", "Dubois", "Kim", "Ivanov", "Rossi", "Meyer",
    "Santos", "Haugen", "Patel", "Novakovic", "Weber", "Ahmed", "Larsen",
    "Kaur", "Novakowski", "Ferrara", "Osei", "Keller", "Marsh", "Varga",
    "Olsen", "Ricci", "Barros", "Nakamura", "Ellis",
]
# NOTE: last names are all distinct so a full name identifies one entity.

FIELDS = {
    "quantum computing": {
        "discoveries": [
            "a fault-tolerant qubit array that held coherence for 11 milliseconds",
            "a photonic processor that factored 21-digit integers",
            "an error-corrected logical qubit with 99.9 percent gate fidelity",
        ],
        "topics": [
            "topological error correction", "photonic interconnects",
            "cryogenic control electronics",
        ],
    },
    "marine biology": {
        "discoveries": [
            "a deep-sea vent community thriving without sunlight",
            "a coral symbiont that survives 2 degree warmer water",
            "a plankton bloom cycle driven by iron dust",
        ],
        "topics": [
            "coral bleaching resilience", "vent ecology", "plankton genomics",
        ],
    },
    "astrophysics": {
        "discoveries": [
            "a fast radio burst repeating every 16 days",
            "an exoplanet atmosphere rich in water vapour",
            "a magnetar flare brighter than a thousand suns",
        ],
        "topics": [
            "transient surveys", "exoplanet atmospheres", "pulsar timing",
        ],
    },
    "computational linguistics": {
        "discoveries": [
            "a parser that recovers syntax from speech with no transcript",
            "a lexicon induction method using only parallel word lists",
            "a dialogue model that tracks beliefs across 40 turns",
        ],
        "topics": [
            "low-resource parsing", "discourse tracking", "lexical semantics",
        ],
    },
    "robotics": {
        "discoveries": [
            "a soft gripper that handles ripe fruit without bruising",
            "a quadruped that recovers from slips on wet ice",
            "a swarm of 200 drones flying through a forest",
        ],
        "topics": [
            "soft manipulation", "legged locomotion", "swarm coordination",
        ],
    },
    "biochemistry": {
        "discoveries": [
            "an enzyme that breaks down PET plastic in 6 hours",
            "a protein switch that turns genes on with blue light",
            "a metabolic pathway that fixes carbon twice as fast",
        ],
        "topics": [
            "enzyme engineering", "optogenetics", "carbon fixation",
        ],
    },
    "climate modelling": {
        "discoveries": [
            "a cloud feedback that cools the tropics by half a degree",
            "an ocean current model that predicts El Nino a year ahead",
            "a glacier melt rate twice the previous estimate",
        ],
        "topics": [
            "cloud feedbacks", "ocean circulation", "ice-sheet dynamics",
        ],
    },
    "neuroscience": {
        "discoveries": [
            "a memory engram that survives total sleep loss",
            "a brain-computer interface spelling at 90 characters a minute",
            "a neural circuit that predicts rewards 2 seconds early",
        ],
        "topics": [
            "memory consolidation", "neural decoding", "reward circuits",
        ],
    },
    "materials science": {
        "discoveries": [
            "a ceramic that stays flexible at 1200 degrees",
            "a battery cathode charging to 80 percent in 9 minutes",
            "a self-healing polymer that seals cracks overnight",
        ],
        "topics": [
            "solid-state batteries", "high-temperature ceramics",
            "self-healing materials",
        ],
    },
    "number theory": {
        "discoveries": [
            "a proof that gaps between primes stay below 200",
            "a sieve that counts primes in arithmetic progressions",
            "a new bound on the least quadratic non-residue",
        ],
        "topics": [
            "prime gaps", "sieve methods", "L-functions",
        ],
    },
}

INSTITUTIONS = [
    "Northbridge Institute of Technology", "Harborview University",
    "Kingsfield Polytechnic", "Aldergate College", "Meridian University",
    "Stonebridge Institute", "Coral Bay University", "Highfield College",
    "Westmoor Polytechnic", "Lakeview Institute of Science",
    "Copperfield University", "Ashford Technical College",
    "Ravenswood University", "Silverton Institute", "Elmwood College",
    "Foxglove University", "Ironbridge Polytechnic", "Willowbank Institute",
    "Cedar Falls University", "Marlowe College", "Beacon Hill Institute",
    "Thornbury University", "Sandhurst Polytechnic", "Maplewood College",
    "Granite Peak University", "Seaview Institute of Technology",
    "Brookfield University", "Holloway College", "Dunmore Institute",
    "Fairfax Polytechnic", "Ridgeline University", "Portland Bay College",
    "Kingsway Institute", "Sunnyside University", "Millbrook College",
    "Tarnmoor Institute of Science",
]

CITIES = [
    "Bristol", "Gdansk", "Valencia", "Trondheim", "Kyoto", "Auckland",
    "Montreal", "Bergen", "Toulouse", "Leuven", "Sendai", "Graz",
    "Uppsala", "Porto", "Daejeon", "Tartu", "Lund", "Innsbruck",
    "Kaohsiung", "Bath", "Turku", "Aarhus", "Grenoble", "Wellington",
]

AWARDS = [
    "Kavli Medal", "Rosenfeld Prize", "Alvarez Award", "Chen Medal",
    "Novak Prize", "Okafor Fellowship", "Lindqvist Award", "Moreau Prize",
]

YEARS = list(range(2018, 2026))


def build_corpus(n=120, seed=SEED):
    """Return a list of entity dicts, each with a 7-sentence passage."""
    rng = random.Random(seed)
    combos = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(combos)
    award_years = [(a, y) for a in AWARDS for y in YEARS]
    rng.shuffle(award_years)  # 64 combos; 120 entities -> each shared ~2x
    fields = list(FIELDS.keys())
    entities = []
    for i in range(n):
        first, last = combos[i]
        name = f"{first} {last}"
        field = rng.choice(fields)
        award, year = award_years[i % len(award_years)]
        inst = rng.choice(INSTITUTIONS)
        city = rng.choice(CITIES)
        discovery = rng.choice(FIELDS[field]["discoveries"])
        topic = rng.choice(FIELDS[field]["topics"])
        topic2 = rng.choice(FIELDS[field]["topics"])
        other1, other2 = rng.sample(INSTITUTIONS, 2)
        while other1 == inst:
            other1 = rng.choice(INSTITUTIONS)
        while other2 == inst or other2 == other1:
            other2 = rng.choice(INSTITUTIONS)
        other_year = rng.choice([y for y in YEARS if y != year])
        other_award = rng.choice([a for a in AWARDS if a != award])
        sentences = [
            f"{name} is a {field} researcher at {inst} in {city}.",
            f"In {year}, {last} received the {award} for contributions to {field}.",
            f"{last} is best known for {discovery}.",
            f"The {field} group at {inst} studies {topic}.",
            f"They collaborate with colleagues at {other1} on {topic2}.",
            f"Earlier, {last} was a visiting scholar at {other2}.",
            f"In {other_year}, the group received the {other_award} grant.",
        ]
        entities.append({
            "id": f"ent{i:03d}",
            "name": name,
            "first": first,
            "last": last,
            "field": field,
            "institution": inst,
            "city": city,
            "award": award,
            "year": year,
            "discovery": discovery,
            "sentences": sentences,
            "passage": " ".join(sentences),
            # gold evidence sentence indices: 0 = institution (hop 2), 1 = award+year (hop 1)
            "gold_hop1": sentences[1],
            "gold_hop2": sentences[0],
        })
    return entities


def build_questions(entities, n_two_hop=60, n_single_hop=30, seed=SEED):
    """Two-hop Qs need a unique (field, award, year) so hop 1 identifies one entity."""
    rng = random.Random(seed + 1)
    from collections import Counter
    combo_counts = Counter((e["field"], e["award"], e["year"]) for e in entities)
    unique = [e for e in entities
              if combo_counts[(e["field"], e["award"], e["year"])] == 1]
    rng.shuffle(unique)
    two_hop = []
    for e in unique[:n_two_hop]:
        two_hop.append({
            "qid": f"q2-{e['id']}",
            "text": (f"Which institution is the {e['field']} researcher who won "
                     f"the {e['award']} in {e['year']} affiliated with?"),
            "entity_id": e["id"],
            "entity": e,
        })
    used = {q["entity_id"] for q in two_hop}
    rest = [e for e in entities if e["id"] not in used]
    rng.shuffle(rest)
    single_hop = []
    for e in rest[:n_single_hop]:
        single_hop.append({
            "qid": f"q1-{e['id']}",
            "text": f"Which institution is {e['name']} affiliated with?",
            "entity_id": e["id"],
            "entity": e,
        })
    return two_hop, single_hop


if __name__ == "__main__":
    ents = build_corpus()
    two, one = build_questions(ents)
    print(f"entities: {len(ents)}  two-hop: {len(two)}  single-hop: {len(one)}")
    print("example passage:", ents[0]["passage"][:300])
    print("example two-hop:", two[0]["text"])
    print("gold hop1:", two[0]["entity"]["gold_hop1"])
    print("gold hop2:", two[0]["entity"]["gold_hop2"])
