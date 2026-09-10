# RMA per-task dossier: what each task does, and what the bank must carry
Sources: [census] = full subtask-chain census over all 2,600 episodes
(2026-09-08), [readme] = data README Table 6, [spec] = rma_spec_and_plan.md.
Written before target building per user instruction; review before stage 5.

## Global identity/dynamics principles (GAMMA mapping)
- ι (identity): object-class ↔ detector coordinate, PLUS two RMA-specific
  bindings: (a) container-role ↔ observed content (drawers/microwave),
  (b) placement-site coordinates INSIDE containers (microwave tasks cite
  them later).
- δ (dynamics): pick/place/pour/open/close completions, numbered pours,
  per-drawer observation events.
- No demo phase anywhere: all memory is accumulated during execution
  (the "demo" of RoboMME is replaced by the robot's own earlier actions).

## Group A — check-then-recall drawers (tasks 4, 5) [THE core memory tasks]
Chain [census]: open/close top→middle→bottom (checking), then REOPEN the
target drawer, place butter, close. Target varies by seed (t4: 34/33/33
top/mid/bottom = drawer containing an object; t5: 80 middle / 20 bottom =
the empty drawer).
- ι: per-drawer observed content while open: "opened the top drawer — saw
  a <object> inside at <x,y>" / "— it is empty". Content is visible ONLY
  during the open span; the writer must record it there (evidence gate:
  detection present/absent inside the drawer bbox while open).
- δ: open/close events per drawer; final placement.
- Reasoner: chooses which drawer to reopen FROM the observation lines
  (t4: the non-empty one; t5: the empty one). This is pure bank lookup.
- Harness: progress claims (drawer open/close completion via drawer-front
  displacement or handle detection); grounding claim for the recalled
  drawer must match the fold of observation lines.
- CHAIN + SUMMARY design (v17 permanence lesson, decided 2026-09-08):
  per-drawer observation lines (each evidence-gated in its own open
  span) + a summary line after the final close ("all drawers checked:
  middle contains an object; top and bottom empty"). The summary is
  deterministically derivable from the observation lines => harness
  CORRECT authority = the fold, exactly as with container chains.
  NOTE: this is the ONLY RMA family needing chain+summary — RMA has no
  swap-grade dynamics (no demo phase, no environment-driven motion;
  every relocation is robot-caused, witnessed, single-hop, at a
  structural boundary). Transfers get move lines only (no summary);
  counting's "(2nd of 2)" line is its own summary; microwave recall is
  a single knowability-critical line (no re-snap once occluded —
  coordinate authority is A's placement line).
- Detector needs: drawer fronts (top/middle/bottom as distinct sites or
  one cabinet + y-band), object-inside-open-drawer, open-state change.
- TODO at corpus build: verify from frames WHICH object appears in the
  non-empty drawer (readme says "an object"; identity per seed unknown).

## Group B — place-at-remembered-location microwave (20, 21, 23, 24)
Chain: open_microwave → pick A → place A → pick B → place B → close.
[readme]: "put B into the location where A was placed."
- ι: A's placement-site coordinate inside the microwave: "placed
  <A> inside the microwave at <x,y>" — B's placement subgoal CITES THIS
  COORDINATE from the bank (A may be occluded/behind door edge by then).
- δ: open, two placements, close.
- Reasoner: emits "place <B> at <x,y>" with x,y from A's line. Analog of
  RoboMME container-binding; knowability: the coordinate must come from
  A's admitted placement line.
- Detector: microwave open state (r3: closed detected, OPEN microwave
  MISSED — round-4 phrase needed, e.g. "an open microwave with its door
  ajar"); item detection inside microwave cavity.

## Group C — same/other-drawer placements (11, 12, 13, 14, 17)
Chains: open drawer(s), place two items, close. 12/13/17: both into the
SAME middle drawer ("then put Y into the same drawer" — recall which);
11/14: X into top, Y into ANOTHER drawer (recall which was already used;
census shows the executed other drawer per seed).
- ι: item ↔ drawer used ("placed cookies into the top drawer at <x,y>").
- δ: open/place/close ordering; what is already inside (invisible after
  close).
- Reasoner: second placement's target drawer from the bank (same vs
  different rule comes from the instruction; WHICH drawer from memory).

## Group D — counting pours (6, 7, 8, 9, 10, 15, 16)
Chains: [pick+place staging item] → pick pour-vessel → pour 1st → pour
2nd → place vessel (drainer/table). Pours are visually identical; the
current frame cannot distinguish 1st from 2nd.
- ι: vessel ↔ coordinate; pour target (cookies/frypan/mug/staged item).
- δ: NUMBERED pour lines: "poured <X> over <target> (1st of 2)" /
  "(2nd of 2)" — the count IS the memory.
- Reasoner: after one pour line → pour again; after two → place vessel.
- Harness: pour-completion verification (RoboMME press analog): vessel
  returns upright / ee kinematic signature / arrival at pour pose;
  premature "2nd pour done" must DEFER. Corpus: structural boundaries
  give exact pour windows.

## Group E — ordered basket sequence (1, 2, 3)
Chain: pick A → place A into basket → pick B → place B into basket.
Order fixed by instruction; basket contents hidden after placement.
- ι: item ↔ coordinate; basket ↔ coordinate.
- δ: completed placements with order index ("placed cookies into the
  basket (1 of 2)").
- Reasoner: progress lookup — mostly completion memory.

## Group F — pour-then-store sequence (22)
Chain: pick tomato → pour 1st → pour 2nd → place aside → open microwave →
pick cookies → place → close. Counting (Group D) + sequence progress.

## Group G — multi-item transfer (18, 19, 25, 26)
Chains: pick/place each item src→dst in order (cabinet1→cabinet2 for
18/19; plate1→plate2 for 25/26; 19 has three items).
- ι: item ↔ src/dst coordinates; which items already transferred
  (cabinet interiors hide placed items).
- δ: "moved <X> from <src> to <dst>" chain lines with order index.
- Reasoner: next untransferred item from the bank.
- NAMING WATCHOUT [census vs readme]: task 26's first item is
  "chocolate_pudding" in filenames but "chocolate and cream" in the
  readme; task 3 chain is cream→pudding but readme says
  "cream, chocolate". Use census/filenames (data authority) for target
  words; humanize from subtask words, not the readme.

## Cross-cutting notes
- Instructions: derive from dataset dir words (spec) — matches the
  official builder; readme text is descriptive only.
- Success/verification: benchmark scores TSR + stage-wise CSR (3-9
  verification stages/task) — our oracle stream must advance per subtask
  (= CSR stages align with structural boundaries).
- Detector round-3 findings: per-task vocabulary works, BUT (a) butter /
  cookies / chocolate / cream / popcorn / pudding are visually similar
  SMALL BOXES — the food-box phrases need round-4 per-box tuning
  (butter and cookies currently 0 hits on their own tasks); (b) open
  microwave missed (door changes appearance); (c) object-inside-drawer
  and drawer-front states untested — add such frames to the round-4
  sample (use stage-1 manifests to sample frames inside open spans).
- Fallback if per-box phrases can't separate the boxes: initial-scene
  binding by position + track-through-motion (RoboMME container
  precedent) — identity carried by the bank, not the detector class.
