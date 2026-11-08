//! Deterministic synthetic multi-hop QA corpus. Faithful port of
//! `corpus.py` in the parent session directory: same templates, same
//! draw-sequence structure, same seeds (`SEED = 20261108`).
//!
//! Divergence: Python uses `random.Random` (Mersenne Twister); Rust uses
//! the xorshift64star RNG in `crate::rng`. The generated entities
//! therefore differ from Python's — different names/awards/institutions
//! per entity — but the corpus has the same shape (120 researchers,
//! 7-sentence passages, gold sentences at positions 0-1) and the
//! mechanism assertions are evaluated on the Rust corpus itself. See
//! `rust/README.md`.

use std::collections::HashMap;

use crate::rng::XorShift64;

pub const SEED: u64 = 20261108;

pub const FIRST_NAMES: &[&str] = &[
    "Amara", "Liam", "Priya", "Mateo", "Ingrid", "Kenji", "Sofia", "Dmitri", "Aisha", "Tomas",
    "Nadia", "Felix", "Mei", "Omar", "Elena", "Ravi", "Clara", "Yusuf", "Hana", "Pavel", "Lucia",
    "Anders", "Zara", "Noah", "Ines", "Viktor", "Anya", "Samir", "Greta", "Diego", "Mira", "Jonas",
    "Leila", "Stefan", "Aiko", "Rafael", "Nina", "Oscar", "Tara", "Emil",
];

pub const LAST_NAMES: &[&str] = &[
    "Okafor",
    "Novak",
    "Chen",
    "Lindqvist",
    "Moreau",
    "Tanaka",
    "Silva",
    "Petrov",
    "Haddad",
    "Kowalski",
    "Nguyen",
    "Fischer",
    "Ali",
    "Bergstrom",
    "Costa",
    "Sharma",
    "Dubois",
    "Kim",
    "Ivanov",
    "Rossi",
    "Meyer",
    "Santos",
    "Haugen",
    "Patel",
    "Novakovic",
    "Weber",
    "Ahmed",
    "Larsen",
    "Kaur",
    "Novakowski",
    "Ferrara",
    "Osei",
    "Keller",
    "Marsh",
    "Varga",
    "Olsen",
    "Ricci",
    "Barros",
    "Nakamura",
    "Ellis",
];

pub struct FieldInfo {
    pub name: &'static str,
    pub discoveries: [&'static str; 3],
    pub topics: [&'static str; 3],
}

/// Same order as the `FIELDS` dict in `corpus.py` (dict insertion order).
pub const FIELDS: &[FieldInfo] = &[
    FieldInfo {
        name: "quantum computing",
        discoveries: [
            "a fault-tolerant qubit array that held coherence for 11 milliseconds",
            "a photonic processor that factored 21-digit integers",
            "an error-corrected logical qubit with 99.9 percent gate fidelity",
        ],
        topics: [
            "topological error correction",
            "photonic interconnects",
            "cryogenic control electronics",
        ],
    },
    FieldInfo {
        name: "marine biology",
        discoveries: [
            "a deep-sea vent community thriving without sunlight",
            "a coral symbiont that survives 2 degree warmer water",
            "a plankton bloom cycle driven by iron dust",
        ],
        topics: [
            "coral bleaching resilience",
            "vent ecology",
            "plankton genomics",
        ],
    },
    FieldInfo {
        name: "astrophysics",
        discoveries: [
            "a fast radio burst repeating every 16 days",
            "an exoplanet atmosphere rich in water vapour",
            "a magnetar flare brighter than a thousand suns",
        ],
        topics: [
            "transient surveys",
            "exoplanet atmospheres",
            "pulsar timing",
        ],
    },
    FieldInfo {
        name: "computational linguistics",
        discoveries: [
            "a parser that recovers syntax from speech with no transcript",
            "a lexicon induction method using only parallel word lists",
            "a dialogue model that tracks beliefs across 40 turns",
        ],
        topics: [
            "low-resource parsing",
            "discourse tracking",
            "lexical semantics",
        ],
    },
    FieldInfo {
        name: "robotics",
        discoveries: [
            "a soft gripper that handles ripe fruit without bruising",
            "a quadruped that recovers from slips on wet ice",
            "a swarm of 200 drones flying through a forest",
        ],
        topics: [
            "soft manipulation",
            "legged locomotion",
            "swarm coordination",
        ],
    },
    FieldInfo {
        name: "biochemistry",
        discoveries: [
            "an enzyme that breaks down PET plastic in 6 hours",
            "a protein switch that turns genes on with blue light",
            "a metabolic pathway that fixes carbon twice as fast",
        ],
        topics: ["enzyme engineering", "optogenetics", "carbon fixation"],
    },
    FieldInfo {
        name: "climate modelling",
        discoveries: [
            "a cloud feedback that cools the tropics by half a degree",
            "an ocean current model that predicts El Nino a year ahead",
            "a glacier melt rate twice the previous estimate",
        ],
        topics: ["cloud feedbacks", "ocean circulation", "ice-sheet dynamics"],
    },
    FieldInfo {
        name: "neuroscience",
        discoveries: [
            "a memory engram that survives total sleep loss",
            "a brain-computer interface spelling at 90 characters a minute",
            "a neural circuit that predicts rewards 2 seconds early",
        ],
        topics: ["memory consolidation", "neural decoding", "reward circuits"],
    },
    FieldInfo {
        name: "materials science",
        discoveries: [
            "a ceramic that stays flexible at 1200 degrees",
            "a battery cathode charging to 80 percent in 9 minutes",
            "a self-healing polymer that seals cracks overnight",
        ],
        topics: [
            "solid-state batteries",
            "high-temperature ceramics",
            "self-healing materials",
        ],
    },
    FieldInfo {
        name: "number theory",
        discoveries: [
            "a proof that gaps between primes stay below 200",
            "a sieve that counts primes in arithmetic progressions",
            "a new bound on the least quadratic non-residue",
        ],
        topics: ["prime gaps", "sieve methods", "L-functions"],
    },
];

pub const INSTITUTIONS: &[&str] = &[
    "Northbridge Institute of Technology",
    "Harborview University",
    "Kingsfield Polytechnic",
    "Aldergate College",
    "Meridian University",
    "Stonebridge Institute",
    "Coral Bay University",
    "Highfield College",
    "Westmoor Polytechnic",
    "Lakeview Institute of Science",
    "Copperfield University",
    "Ashford Technical College",
    "Ravenswood University",
    "Silverton Institute",
    "Elmwood College",
    "Foxglove University",
    "Ironbridge Polytechnic",
    "Willowbank Institute",
    "Cedar Falls University",
    "Marlowe College",
    "Beacon Hill Institute",
    "Thornbury University",
    "Sandhurst Polytechnic",
    "Maplewood College",
    "Granite Peak University",
    "Seaview Institute of Technology",
    "Brookfield University",
    "Holloway College",
    "Dunmore Institute",
    "Fairfax Polytechnic",
    "Ridgeline University",
    "Portland Bay College",
    "Kingsway Institute",
    "Sunnyside University",
    "Millbrook College",
    "Tarnmoor Institute of Science",
];

pub const CITIES: &[&str] = &[
    "Bristol",
    "Gdansk",
    "Valencia",
    "Trondheim",
    "Kyoto",
    "Auckland",
    "Montreal",
    "Bergen",
    "Toulouse",
    "Leuven",
    "Sendai",
    "Graz",
    "Uppsala",
    "Porto",
    "Daejeon",
    "Tartu",
    "Lund",
    "Innsbruck",
    "Kaohsiung",
    "Bath",
    "Turku",
    "Aarhus",
    "Grenoble",
    "Wellington",
];

pub const AWARDS: &[&str] = &[
    "Kavli Medal",
    "Rosenfeld Prize",
    "Alvarez Award",
    "Chen Medal",
    "Novak Prize",
    "Okafor Fellowship",
    "Lindqvist Award",
    "Moreau Prize",
];

pub const YEARS: &[u32] = &[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub first: String,
    pub last: String,
    pub field: String,
    pub institution: String,
    pub city: String,
    pub award: String,
    pub year: u32,
    pub discovery: String,
    pub sentences: Vec<String>,
    pub passage: String,
    /// Gold evidence sentence indices: 0 = institution (hop 2),
    /// 1 = award + year (hop 1).
    pub gold_hop1: String,
    pub gold_hop2: String,
}

#[derive(Debug, Clone)]
pub struct Question {
    pub qid: String,
    pub text: String,
    pub entity_id: String,
    pub entity_idx: usize,
}

fn pick<'a>(rng: &mut XorShift64, xs: &'a [&'a str]) -> &'a str {
    xs[rng.next_below(xs.len())]
}

/// Two distinct items from `xs`, via a partial Fisher-Yates shuffle.
/// Divergence: Python's `random.sample` uses selection sampling; the
/// resulting pair distribution is the same (uniform over unordered
/// pairs) but the draw sequence differs — irrelevant here because the
/// RNG stream already differs.
fn sample_two<'a>(rng: &mut XorShift64, xs: &'a [&'a str]) -> (&'a str, &'a str) {
    let mut idxs: Vec<usize> = (0..xs.len()).collect();
    for i in 0..2 {
        let j = i + rng.next_below(xs.len() - i);
        idxs.swap(i, j);
    }
    (xs[idxs[0]], xs[idxs[1]])
}

/// Build `n` entities, each with a 7-sentence passage. Same templates and
/// draw order as `corpus.py::build_corpus`.
pub fn build_corpus(n: usize, seed: u64) -> Vec<Entity> {
    let mut rng = XorShift64::new(seed);
    let mut combos: Vec<(String, String)> = FIRST_NAMES
        .iter()
        .flat_map(|f| {
            LAST_NAMES
                .iter()
                .map(move |l| (f.to_string(), l.to_string()))
        })
        .collect();
    rng.shuffle(&mut combos);
    let mut award_years: Vec<(String, u32)> = AWARDS
        .iter()
        .flat_map(|a| YEARS.iter().map(move |y| (a.to_string(), *y)))
        .collect();
    rng.shuffle(&mut award_years);
    let mut entities = Vec::with_capacity(n);
    for i in 0..n {
        let (first, last) = combos[i].clone();
        let name = format!("{} {}", first, last);
        let field = &FIELDS[rng.next_below(FIELDS.len())];
        let (award, year) = award_years[i % award_years.len()].clone();
        let inst = pick(&mut rng, INSTITUTIONS).to_string();
        let city = pick(&mut rng, CITIES).to_string();
        let discovery = field.discoveries[rng.next_below(field.discoveries.len())].to_string();
        let topic = field.topics[rng.next_below(field.topics.len())].to_string();
        let topic2 = field.topics[rng.next_below(field.topics.len())].to_string();
        let (o1, o2) = sample_two(&mut rng, INSTITUTIONS);
        let mut other1 = o1.to_string();
        let mut other2 = o2.to_string();
        while other1 == inst {
            other1 = pick(&mut rng, INSTITUTIONS).to_string();
        }
        while other2 == inst || other2 == other1 {
            other2 = pick(&mut rng, INSTITUTIONS).to_string();
        }
        let other_years: Vec<u32> = YEARS.iter().copied().filter(|y| *y != year).collect();
        let other_year = other_years[rng.next_below(other_years.len())];
        let other_awards: Vec<&str> = AWARDS
            .iter()
            .copied()
            .filter(|a| *a != award.as_str())
            .collect();
        let other_award = other_awards[rng.next_below(other_awards.len())].to_string();
        let sentences = vec![
            format!(
                "{} is a {} researcher at {} in {}.",
                name, field.name, inst, city
            ),
            format!(
                "In {}, {} received the {} for contributions to {}.",
                year, last, award, field.name
            ),
            format!("{} is best known for {}.", last, discovery),
            format!("The {} group at {} studies {}.", field.name, inst, topic),
            format!(
                "They collaborate with colleagues at {} on {}.",
                other1, topic2
            ),
            format!("Earlier, {} was a visiting scholar at {}.", last, other2),
            format!(
                "In {}, the group received the {} grant.",
                other_year, other_award
            ),
        ];
        entities.push(Entity {
            id: format!("ent{:03}", i),
            name,
            first,
            last,
            field: field.name.to_string(),
            institution: inst,
            city,
            award,
            year,
            discovery,
            passage: sentences.join(" "),
            gold_hop1: sentences[1].clone(),
            gold_hop2: sentences[0].clone(),
            sentences,
        });
    }
    entities
}

/// Two-hop questions need a unique (field, award, year) so hop 1
/// identifies one entity. Same structure as
/// `corpus.py::build_questions` (RNG seeded `seed + 1`).
pub fn build_questions(
    entities: &[Entity],
    n_two_hop: usize,
    n_single_hop: usize,
    seed: u64,
) -> (Vec<Question>, Vec<Question>) {
    let mut rng = XorShift64::new(seed + 1);
    let mut combo_counts: HashMap<(String, String, u32), usize> = HashMap::new();
    for e in entities {
        *combo_counts
            .entry((e.field.clone(), e.award.clone(), e.year))
            .or_insert(0) += 1;
    }
    let mut unique: Vec<usize> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| combo_counts[&(e.field.clone(), e.award.clone(), e.year)] == 1)
        .map(|(i, _)| i)
        .collect();
    rng.shuffle(&mut unique);
    let mut two_hop = Vec::new();
    for &idx in unique.iter().take(n_two_hop) {
        let e = &entities[idx];
        two_hop.push(Question {
            qid: format!("q2-{}", e.id),
            text: format!(
                "Which institution is the {} researcher who won the {} in {} affiliated with?",
                e.field, e.award, e.year
            ),
            entity_id: e.id.clone(),
            entity_idx: idx,
        });
    }
    let used: std::collections::HashSet<&str> =
        two_hop.iter().map(|q| q.entity_id.as_str()).collect();
    let mut rest: Vec<usize> = entities
        .iter()
        .enumerate()
        .filter(|(_, e)| !used.contains(e.id.as_str()))
        .map(|(i, _)| i)
        .collect();
    rng.shuffle(&mut rest);
    let mut single_hop = Vec::new();
    for &idx in rest.iter().take(n_single_hop) {
        let e = &entities[idx];
        single_hop.push(Question {
            qid: format!("q1-{}", e.id),
            text: format!("Which institution is {} affiliated with?", e.name),
            entity_id: e.id.clone(),
            entity_idx: idx,
        });
    }
    (two_hop, single_hop)
}
