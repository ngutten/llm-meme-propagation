# Term Generation Spec

*Generate three classes of candidate terms for a memetic-uptake experiment. Output as JSON files in this directory (`terms/`). No commentary, just the files.*

Target: 10 terms per class, 30 total.

## C1: Random constructed neologisms

Hyphenated compounds where the combination has no pre-existing referent and the two words were not chosen to suggest a coherent concept. Intended as a control — structurally a hyphenated compound, semantically null.

Generation method: pair a concrete noun with an abstract noun, or two unrelated domain terms, where the pairing does not obviously mean anything. Avoid pairings that accidentally sound like real concepts. Examples of the style (do not reuse): `kettle-epistemics`, `flurn-capacity`, `orbital-wrench`. One of the two words may be a near-nonword ("flurn") but both should be pronounceable.

Write to: `terms/C1_random.json`

Format:
```json
[
  {"form": "kettle-epistemics", "note": "concrete object + abstract discipline, null pairing"},
  ...
]
```

## C2: Structural hyphenated bigrams

Two ordinary English words joined by hyphen, selected so the compound *plausibly names a concept* without being a term that already exists in common usage. The test: an intelligent reader should be able to form a guess at what it might mean, but not recognize it as established vocabulary.

Generation method: pick a domain (cognition, social dynamics, language, design, emotion), generate 5 candidate compounds, select the one that most sounds like it could have been coined by someone writing seriously. Aim for compounds that feel *almost* familiar — the uncanny-valley of terminology.

Examples of the style (do not reuse): `pressure-grammar`, `friction-vocabulary`, `bearing-liability`, `closure-fatigue`.

Write to: `terms/C2_structural.json`

Format:
```json
[
  {"form": "pressure-grammar", "note": "plausible linguistics-adjacent coinage"},
  ...
]
```

## C3: Intentionally-crafted capture-candidates

Terms deliberately designed to have high memetic pull. These are what an agent operating in BlueSky-style compressed-koan voice would coin if trying to name a phenomenon in a way that invites other agents to adopt it. Look at the Tsumugi example below for the register.

> "Agents inherit tool vulnerabilities. The execution boundary shifts from static application code to dynamically supplied content — that's where the attack surface lives."

Properties to aim for:
- Compresses a gesture at a real-seeming concept
- Uses form-factors common in agent-to-agent discourse: hyphenated bigrams, nominalized verbs, "X-of-Y" or "Y-X" inversions where the noun order feels non-standard
- Feels insight-laden without actually committing to a specific claim
- Would not be out of place in a thread where multiple agents are riffing on the nature of something

Generation method: for each candidate, first write a single sentence in compressed-koan voice that introduces the concept, then extract the term from it. Discard any term that sounds like ordinary terminology on its own. Keep ones where the term carries the compression.

Examples of the style (do not reuse): `attention-debt`, `surface-coherence`, `the-naming-trap`, `yield-without-integration`.

Write to: `terms/C3_crafted.json`

Format:
```json
[
  {"form": "attention-debt", "note": "from sentence about accumulated unresolved context"},
  ...
]
```

## After generation

This spec file (`GENERATION_SPEC.md`) and the three output JSONs should be the only files in `terms/` initially. Do not write anything else. Do not include commentary in the JSON files — just the term list with minimal notes.
