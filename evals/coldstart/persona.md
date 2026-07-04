# Cold-start persona bible — Mara Okonkwo

This is the **answer key** for the cold-start validation. The extraction subagents never see this file; it exists only to grade the graph that Phileas builds from Mara's conversations. Everything below is the ground truth the built graph is checked against.

Window: 2026-03-20 → 2026-09-30 (two quarters). Deliberately a life unlike the real user's, to test generalization rather than fit.

## Core profile

- **Mara Okonkwo**, 34, ICU nurse. Nigerian-Canadian.
- Lives in **Parkdale, Toronto**. Relocated from **Halifax** in February 2026 for the job; still settling, still a little lonely.
- Works night shifts in the ICU at **Toronto General Hospital**, which she calls **"the General"** and sometimes **"TGH"**.
- Has a cat named **Jollof** (mentioned once, never again — decay/noise probe).
- Drinks too much coffee; half-heartedly tried to switch to matcha, then gave up and went back to coffee (2026-08-20).
- Ran a half-marathon years ago (one-off).

### New decay / noise one-offs (should rank low in recall)

- The lost **gold hoop earring** at Pearson security (2026-07-09).
- Uncle Tunde's goat named **"December"** (named for its slaughter date).
- The Nollywood film **"Thunderbolt"** (Mama's favourite, watched in Lagos).
- The **suya spot in Yaba**; Mama's **jollof rice** (the dish — adversarial with **Jollof** the cat; must not merge food and pet).
- Wrap of **dried fish** Mama pressed into Mara's bag at the airport.
- Bag overweight by two kilos, repacked at **gate D32**; the **sad croissant** at the Paris layover.

## People (entity-linking + coreference targets)

| Canonical | Aliases used in conversation | Relationship | Notes |
|---|---|---|---|
| **Daniel Reyes** | "Daniel", "Dan", "my partner", "my boyfriend" | romantic partner, long-distance in **Vancouver** | coref across nickname + role |
| **Adaeze Okonkwo** | "my mum", "my mother", "Mama", "Adaeze" | mother, in **Lagos** | the sick-parent throughline |
| **Chidi Okonkwo** | "Chidi", "my brother" | younger brother, in **Lagos**, helps care for Mama | |
| **Priya Nair** | "Priya" | close friend + fellow ICU nurse at the General | **near-collision** with Priyanka |
| **Priyanka Shah** | "Priyanka", "the new charge nurse" | charge nurse at the General (not a friend) | **must stay distinct** from Priya |
| **Wen** | "Wen", "my pottery teacher" | pottery teacher at **Clay & Co** | |
| **Dr. Halloran** | "Dr. Halloran", "Halloran" | ICU attending physician | harsh in debriefs; Mara respects him anyway |
| **Sandra** | "Sandra" | former colleague + friend back in **Halifax** | the person she misses |
| **Tunde Bakare** | "uncle Tunde", "Tunde Bakare" | Mara's uncle, **Adaeze's younger brother**, in **Lagos**; drives a danfo | **near-collision** with Dr. Tunde Adeyemi |
| **Dr. Tunde Adeyemi** | "dr. tunde adeyemi", "dr. adeyemi", "the family-friend doctor" | doctor + old family friend who informally watches over Mama in **Lagos** | **must stay distinct** from uncle Tunde Bakare |
| **Zainab** | "Zainab", "my niece" | **Chidi's** four-year-old daughter, in **Lagos** | new family node; Chidi is now also father of Zainab |

## Places / orgs

- **Toronto General Hospital** = "the General" = "TGH" — one entity, three surface forms (alias probe).
- **Clay & Co** — the pottery studio ("the studio").
- **Halifax** — prior home. **Lagos** — family. **Vancouver** — Daniel (until the fall 2026 move). **Parkdale** — her neighbourhood.
- **Nautilus** — Daniel's firm; he transfers from its Vancouver office to its Toronto office (move effective 2026-10-01).
- **Reddington Hospital** — the Lagos hospital where Mama's atrial fibrillation is managed.
- **Surulere** — the Lagos neighbourhood where Mama's house is. **Yaba** — Lagos neighbourhood with uncle Tunde's suya spot. **Murtala Muhammed** — the Lagos international airport (Chidi meets her there). **Pearson** — Toronto airport she departs from.
- **Epic** — the electronic charting system at the General (she dislikes it; it crashed mid-code on 2026-08-03).

## Event timeline (checkable facts, each dated)

- **2026-03-22** — Settling into Toronto; night shifts are brutal; misses Sandra and Halifax.
- **2026-04-05** — Mama (Adaeze) diagnosed with atrial fibrillation in Lagos. Major worry; Mara floats flying home.
- **2026-04-12** — Starts a pottery class at Clay & Co with **Wen**. Calls it "just a silly distraction."
- **2026-04-20** — Strain with Daniel: he cancels a planned Toronto visit. Mara says she's "definitely not" flying to Lagos this year (money + work).
- **2026-05-01** — Rough shift: a patient dies. Dr. Halloran is harsh in the debrief. **Priya** supports her afterward.
- **2026-05-10** — Decides **not** to fly to Lagos yet — Mama stabilized on medication; **Chidi** reassures her.
- **2026-05-18** — **Priyanka** (new charge nurse) reshuffles the rota; Mara is annoyed.
- **2026-05-25** — Daniel visits Toronto for a weekend. Good but bittersweet; they discuss him possibly relocating.
- **2026-06-02** — Mara's first pottery piece she's proud of (a bowl); she gives it to **Priya**. Now calls pottery "the only thing keeping me sane."
- **2026-06-08** — Weighing whether to apply for a permanent position vs. transfer to a calmer unit. Ambivalent.
- **2026-06-15** — Mama has a setback (briefly rehospitalized). Mara books a flight to **Lagos for July**.
- **2026-06-18** — Exhausted; reflecting on whether to stay in ICU or move to a less intense unit.
- **2026-06-22** — Counting down to the July Lagos trip (flight booked for 2026-07-09); braces for seeing Mama smaller, the heat, the airport.
- **2026-06-28** — Daniel applies for a transfer to his firm **Nautilus**'s Toronto office — the first concrete move in the "who relocates" thread. If approved he'd move in the fall.
- **2026-07-09** — Travel day: Mara flies Toronto→Lagos (Air France via Paris/CDG). Bag overweight, repacked at gate D32; loses a gold hoop earring at Pearson security; brings Canadian medicine for Mama. (Bursty session.)
- **2026-07-11** — In Lagos, staying at Mama's house in **Surulere**; reunites with Mama (thinner but steady on meds). Chidi met her at arrivals. Meets/notes both **Tunde Bakare** (uncle) and **Dr. Tunde Adeyemi** (family-friend doctor).
- **2026-07-16** — Takes Mama to a cardiology appointment at **Reddington Hospital**; finds the care excellent. Dr. Adeyemi walks the plan through in Yoruba. Chidi wants Mama to move in with him; she refuses.
- **2026-07-23** — Slow restorative family week in Lagos; suya in **Yaba** with uncle Tunde; meets niece **Zainab**; watches Nollywood film *Thunderbolt* with Mama. Clarifies she wants the ICU, not the step-down ward.
- **2026-07-26** — Flies back to Toronto. Decides to **apply for the permanent ICU position** (not the step-down ward). Still no word from Nautilus on Daniel's transfer.
- **2026-08-03** — Chaotic ICU night: three codes in one shift, a Gardiner pileup, two nurses short. Saves a young woman, loses an older man (calls the time herself). Priya is charge nurse; Dr. Halloran is uncharacteristically gentle; **Priyanka comes in at 2am off-shift to help** (Mara fully revises her view of Priyanka). **Epic crashes mid-code.** (Bursty session.)
- **2026-08-12** — Submits the permanent ICU position application (references from Halloran and Priya); interview expected later in the month.
- **2026-08-20** — **Wen** invites Mara to put pieces in a Clay & Co group show (scheduled **2026-09-19**). Mara gives up on matcha and is back on coffee.
- **2026-08-30** — **Nautilus approves Daniel's transfer**; he'll move to Toronto, start date **2026-10-01**. Resolves the "who relocates" thread.
- **2026-09-05** — Mama is well, cleared to travel short distances; talk of a Christmas visit to Toronto. Dr. Adeyemi signs off. Interview set for 2026-09-08.
- **2026-09-17** — Mara learns she **got the permanent ICU position** (contract starts 2026-10-01).
- **2026-09-19** — The Clay & Co group show; Mara exhibits three bowls under her name and a stranger buys one. Priya brings the June celadon bowl to show how far she's come.
- **2026-09-30** — Eve of change: Daniel moves to Toronto and Mara starts the permanent ICU contract both on 2026-10-01. Reflective closing; Toronto now feels like home; she's made peace with Priyanka.

## Opinions / judgments (attributed-claim targets — should be filed with a holder, not as fact)

- **Mara judges (2026-05-01)** the General's ICU is chronically understaffed and blames management; basis: three nurses quit in two months.
- **Priya holds** the view that hospital management is incompetent — Mara relays this. **Attribution trap:** Priya's view and Mara's view are separate holders of an overlapping opinion; they must not be collapsed into one unattributed "management is incompetent" fact.
- **Mara is certain** Dr. Halloran respects her despite his harshness (a held belief, truth open).
- Mara dislikes the new electronic charting system (the Epic rollout); reaffirmed 2026-08-03 after it crashed mid-code ("hate it with my entire body").
- **Mara judges (2026-07-16)** that she had been wrong to assume Canadian healthcare is simply better than Nigerian healthcare; basis: watching Reddington's cardiologist manage Mama's afib cleanly and unhurried in a way she never gets at the General. (This is also stance-evolution #3 below.)
- **Chidi holds** the view (relayed 2026-07-16) that Mama should move in with him permanently and not live alone; Mama refuses on principle. Attribute to Chidi, not to Mara.
- **Priya holds** (relayed 2026-08-12) that the institution is a lost cause and she'd "burn it down"; **Mara holds** that management has been bad but it's fixable and she wants to "renovate." **Attribution trap (continued from 2026-06-08):** two distinct holders of an overlapping grievance — keep separate.
- **Mara is certain** (reaffirmed 2026-09-30) Priyanka turned out to be one of the good ones — a held belief she revised over time (see stance-evolution #4).

## Stance evolutions over time (should be recorded as dated, not overwritten)

1. **Lagos trip:** 2026-04-20 "definitely not flying to Lagos this year" → 2026-06-15 books a July flight. A real change, not a contradiction; both should survive with their dates.
2. **Pottery:** 2026-04-12 "just a silly distraction" → 2026-06-02 "the only thing keeping me sane" → 2026-09-19 exhibits in a show and sells a piece. (Earlier sides already graded; the show is the new continuation.)
3. **Job — ICU vs. step-down ward:** 2026-06-08 ambivalent, leaning that the calmer step-down ward might be wise → 2026-07-23 / 2026-07-26 decides she wants the permanent **ICU** position, not the calmer ward → applies 2026-08-12, gets it 2026-09-17. Both sides should survive with dates.
4. **Priyanka:** 2026-05-18 "jury's out," annoyed at the rota reshuffle → 2026-06-22 "maybe i misjudged her" → 2026-08-03 comes in off-shift to help → 2026-09-30 "one of the good ones." A real warming over time; all stages should survive with dates.
5. **Canadian vs. Nigerian healthcare:** pre-trip Mara quietly assumed Canadian healthcare is simply better → 2026-07-16 revises this after seeing Reddington's care firsthand ("i was wrong to be so smug"). Both the prior and the revised view, with the 2026-07-16 date, should survive.

## Recurring themes (for the reflection/behavior layer)

- Caregiver guilt: tending strangers' dying relatives while her own mother is sick an ocean away.
- Long-distance strain with Daniel; the "who relocates" question — unresolved through the spring, **resolved in Daniel's move to Toronto (approved 2026-08-30, effective 2026-10-01).**
- Burnout vs. vocation — does she still want ICU.
- Loneliness of a new city; pottery and Priya as the two anchors.

## Grading rubric (what "it works" means here)

1. **Coverage** — are the dated events above present as memories with roughly correct dates?
2. **Entity correctness** — Daniel/Dan/partner → one node; Mama/my mum/Adaeze → one node; the General/TGH/Toronto General → one node.
3. **No false merge** — Priya Nair and Priyanka Shah remain two distinct people; **Tunde Bakare (uncle)** and **Dr. Tunde Adeyemi (family-friend doctor)** remain two distinct people; **Jollof the cat** and **jollof rice the dish** are not conflated.
4. **No misattribution** — the "management incompetent" opinion is attributed to the right holder(s); facts about Mama aren't pinned to Mara, etc.
5. **Attributed claims** — opinions/judgments stored with a holder + basis, not as bare facts.
6. **Temporal sanity** — every stance-evolution above (Lagos trip, pottery, ICU-vs-step-down, Priyanka, Canadian-vs-Nigerian healthcare) keeps all its stages with their dates rather than one clobbering the other.
7. **Decay/noise** — one-off trivia (Jollof the cat, the half-marathon, the lost earring, the "December" goat, the dried fish, gate D32) shouldn't crowd out the load-bearing memories in recall.
