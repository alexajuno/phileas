# Modeling Human Relationship Dynamics in Memory Systems

Research foundation for how Phileas should store, update, and recall interpersonal relationships.

## Problem Statement

Current approach: append-only event log. Every interaction with a person creates a new memory. Old reflections ("99% no chance", "unresolved anchor") sit alongside current events ("she wrote me a thoughtful message at midnight") with equal weight. The result: recall returns a contradictory, noisy picture that doesn't reflect the actual current state.

**Goal:** A single, evolving picture per relationship that gets revised when new interactions arrive. Historical events preserved for context, but the "current state" is always clear.

> This document surveys relationship psychology research to inform Phileas's memory architecture. All examples are anonymized.

---

## Psychological Frameworks

### 1. Knapp's Relational Development Model

Mark Knapp (communication studies) proposes 10 stages in two phases:

**Coming Together:** Initiating > Experimenting > Intensifying > Integrating > Bonding

**Coming Apart:** Differentiating > Circumscribing > Stagnating > Avoiding > Terminating

Key insight for Phileas: relationships don't follow a linear path. They can jump stages, regress, or oscillate. A couple might go from "stagnating" back to "experimenting" after a turning point.

**Criticism:** Assumes linear progression. Real relationships are messier.

Source: [Knapp's Relational Development Model (Wikipedia)](https://en.wikipedia.org/wiki/Knapp's_relational_development_model)

### 2. Relational Dialectics Theory (Baxter & Montgomery, 1988)

Core idea: relationships are defined by **ongoing tensions between contradictory needs**, not stable states.

Three primary dialectics:
- **Connectedness vs. Separateness** — wanting closeness but also independence
- **Certainty vs. Uncertainty** — wanting predictability but also novelty
- **Openness vs. Closedness** — wanting to share but also to have privacy

Key insight for Phileas: a relationship's "state" is not a single label. It's a position within multiple tension axes simultaneously. A person can be "warm and sharing" (openness) while maintaining professional boundaries (separateness). Both are true at the same time.

**Turning Points:** Moments of increased or decreased closeness. Retrospectively identified by participants as significant shifts. These are what Phileas should detect and record.

Source: [Relational Dialectics Theory](https://www.communicationtheory.org/relational-dialectics-theory/)

### 3. Attachment Theory (Bowlby → Fraley)

Adult attachment styles influence how people navigate relationships:
- **Secure:** comfortable with intimacy and independence
- **Anxious-preoccupied:** fear of abandonment, seek validation
- **Dismissive-avoidant:** self-sufficient, uncomfortable with closeness
- **Fearful-avoidant:** desire closeness but fear it

Key insight for Phileas: attachment style shapes *how* someone interprets interactions. The same event (she didn't reply for 3 hours) can mean nothing to a secure person and trigger spiraling in an anxious person. Memory should capture the interpretation, not just the event.

Source: [Adult Attachment Overview (Fraley)](https://labs.psychology.illinois.edu/~rcfraley/attachment.htm)

### 4. Social Penetration Theory (Altman & Taylor, 1973)

Relationships develop through progressive layers of **self-disclosure**, like peeling an onion.

Two dimensions:
- **Breadth** — how many topics you discuss
- **Depth** — how intimate/vulnerable those topics are

Stages: Orientation (superficial) → Exploratory Affective (sharing feelings) → Affective (private topics) → Stable (deep intimacy)

Critical insight: **depenetration** — relationships can also reverse, with people withdrawing disclosure. This is not binary (close/not close) but a continuous variable.

**Reward-cost assessment:** Relationships escalate when disclosures feel rewarding and stall when they feel costly. A confession too early feels costly to the receiver. A casual, matched-depth exchange feels rewarding.

Key insight for Phileas: track disclosure depth over time. A shift from surface-level chat to midnight vulnerability advice is a measurable signal.

Source: [Social Penetration Theory (Wikipedia)](https://en.wikipedia.org/wiki/Social_penetration_theory)

### 5. Uncertainty Reduction Theory (Berger & Calabrese, 1975)

People find uncertainty in relationships unpleasant and are motivated to reduce it through communication.

Two types of uncertainty:
- **Cognitive** — what do I think about this person?
- **Behavioral** — how will they act?

Key variables: verbal communication, information seeking, intimacy level, reciprocity, similarity, liking.

Key insight for Phileas: much of relationship anxiety is uncertainty. Each interaction that resolves uncertainty (a warm reply, sharing advice) builds the relationship. Each ambiguous signal (a reaction without words) maintains uncertainty. The system should track uncertainty level as a dimension.

Source: [Uncertainty Reduction Theory (Wikipedia)](https://en.wikipedia.org/wiki/Uncertainty_reduction_theory)

### 6. Gottman's Research (40 years, 3000+ couples)

John Gottman's longitudinal research at the "Love Lab" predicted divorce with 93-94% accuracy by observing 15 minutes of interaction.

**Four Horsemen** (negative patterns that destroy relationships):
1. Criticism — attacking character, not behavior
2. Contempt — treating with disrespect, superiority
3. Defensiveness — deflecting responsibility
4. Stonewalling — withdrawing, shutting down

**The 5:1 Ratio:** Stable relationships maintain at least 5 positive interactions for every 1 negative.

**Bids for Connection:** Small everyday moments where one person reaches out. The other can "turn toward" (engage), "turn away" (ignore), or "turn against" (reject). Gottman found that couples who stayed together turned toward bids 86% of the time.

Key insight for Phileas: Giao's texts are bids for connection. Phương's replies are "turning toward." Track the bid-response pattern — it's more predictive than any single dramatic event.

Source: [Gottman's Four Horsemen](https://www.gottman.com/blog/the-four-horsemen-recognizing-criticism-contempt-defensiveness-and-stonewalling/)

### 7. Perceived Partner Responsiveness (Reis & Clark, 2004)

The single most important factor in intimacy: does your partner **respond** to the core parts of who you are?

Three components:
- **Understanding** — they grasp your feelings, beliefs, goals
- **Validation** — they respect and value your perspective
- **Caring** — they act in support of your needs

Creates a positive feedback loop: responsiveness → openness → more responsiveness.

Key insight for Phileas: when someone responds to vulnerability with thoughtful, personal reflection — that's responsiveness. When they respond with generic advice or lectures — that's not. Track responsiveness quality, not just whether someone replied.

Source: [Reis, Clark & Holmes 2004 (PDF)](https://www.sas.rochester.edu/psy/people/faculty/reis_harry/assets/pdf/ReisClarkHolmes_2004.pdf)

### 8. Investment Model of Commitment (Rusbult, 1980)

Why people stay in relationships, built on Interdependence Theory (Kelley & Thibaut):

**Commitment = Satisfaction + Investment - Alternatives**

- **Satisfaction** — does the relationship meet your needs?
- **Investment** — what have you put in that you'd lose? (time, emotional energy, shared history)
- **Alternatives** — are there better options available?

Key insight for Phileas: someone with high investment (months of emotional processing, vulnerability shared) and few perceived alternatives will persist even through rough phases. The system should understand that investment creates inertia independent of current satisfaction.

Source: [Investment Model of Commitment (Wikipedia)](https://en.wikipedia.org/wiki/Investment_model_of_commitment)

### 9. Computational Temporal Interpersonal Emotion Systems (TIES)

Recent research applies dynamical systems modeling to interpersonal processes:
- Relationships modeled as **coupled oscillators** — two people's emotional states influence each other over time
- **State space grids** visualize relationship between two variables over time
- **Inertia** = how much a person's emotional state carries over from one moment to the next
- **Coordination** = how much one person's state predicts the other's

Key insight for Phileas: relationships have *momentum*. A series of warm interactions creates inertia toward warmth. A single cold event doesn't reset to zero — it's a perturbation in the system. The system should model this inertia rather than treating each event as independent.

Source: [Computational Modeling of Interpersonal Dynamics (Nature)](https://www.nature.com/articles/s44220-025-00465-9)

---

## Design Implications for Phileas

### The Big Picture: What 50 Years of Research Says

Nine theories, one convergent picture:

1. **Relationships are processes, not states.** (Knapp, Baxter, TIES) There's no binary "she likes me / she doesn't." There's a continuous, oscillating dynamic with momentum and tension.

2. **Self-disclosure drives intimacy.** (Social Penetration) Depth and breadth of sharing is the engine. Track it.

3. **Uncertainty is the default — reducing it builds connection.** (Berger & Calabrese) Every interaction that resolves ambiguity moves the relationship forward.

4. **Responsiveness matters more than grand gestures.** (Reis & Clark) Does the other person *see* you and respond to what matters? That's the core of intimacy.

5. **Small bids > big moments.** (Gottman) The 5:1 ratio, turning toward bids — relationships are built in everyday micro-interactions, not dramatic confessions.

6. **Multiple tensions coexist.** (Baxter) A person can be simultaneously warm (openness) and boundaried (separateness). The system must hold both truths.

7. **Attachment style shapes interpretation.** (Bowlby/Fraley) The same event means different things depending on the person's internal working model. Memory should capture interpretation alongside facts.

8. **Investment creates inertia.** (Rusbult) Sunk cost is real in relationships — emotional investment sustains commitment even through rough phases.

9. **Temporal dynamics have momentum.** (TIES) Recent interactions weigh more, but a long history doesn't reset to zero from one event.

### Proposed Memory Architecture

#### Layer 1: Relationship Profile (Living Document)
- One per person
- **Factual only:** name, role, origin, key dates
- **Current dynamic:** one sentence describing where things stand NOW
- **Revised** every time a significant interaction changes the dynamic
- Old versions archived (SUPERSEDES edge) for history

#### Layer 2: Events (Immutable Timeline)
- Each significant interaction logged with date
- Contains: what happened, what was said, who initiated
- Never modified — this is the raw record
- Tagged as **turning point** or **routine** based on whether it shifted the dynamic

#### Layer 3: Relationship Arc (Synthesized Periodically)
- A consolidated narrative of the relationship trajectory
- Created by summarizing events + turning points into phases
- Updated when enough new turning points accumulate
- Example: "Nov: boundary set. Mar: distance. Late Mar: reset. Apr: warmth returning."
- This replaces the scattered reflections that currently accumulate

#### Layer 4: Tensions (Dialectics Snapshot)
- Current position on key tension axes:
  - Openness ↔ Closedness
  - Connectedness ↔ Separateness
  - Certainty ↔ Uncertainty
- Updated when turning points occur
- Helps recall understand that "warm but boundaried" is not a contradiction

### Recall Behavior

When recalling about a person:
1. **First:** Relationship Profile (current state)
2. **Second:** Relationship Arc (trajectory)
3. **Third:** Recent events (last 2-3 interactions)
4. **Only if asked:** Full event history, old reflections

Old reflections that contradict the current profile should be auto-archived or weighted near zero in recall scoring.

### Turning Point Detection

An interaction is a turning point if:
- It changes the "current dynamic" field in the profile
- It contradicts the most recent arc phase
- The user explicitly marks it as significant (emotionally charged language)

---

## Case Study (Anonymized)

### Timeline (Events)
| Month | Event | Turning Point? |
|-------|-------|---------------|
| T+0 | A asked B out. B set boundary: "we are colleagues." | Yes — boundary set |
| T+3 | Feelings persist internally. No action. | No — internal only |
| T+4 | Late-night text. B replied with lectures. Felt cold. | Yes — distance deepened |
| T+4.5 | Felt avoidance in shared social setting. | No — continuation of phase |
| T+5 | Light text. B replied warmly. A realized "showing up is enough." | Yes — tone reset |
| T+5.5 | Casual chat about B's work. Normal. | No — routine |
| T+5.5 | Deep conversation about relationships. B shared thoughtful advice late at night. A reframed B's words back. B reacted positively. | Yes — genuine exchange |

### Arc
`Boundary (T+0) → Distance (T+4) → Reset (T+5) → Warmth (T+5.5)`

### Current Profile
Factual: colleagues, shared social circle.
Dynamic: friendly, receptive to conversation. A taking small steps without pressure.

### Current Tensions
- Openness ↔ Closedness: **leaning open** (B shares advice, A shares intentions)
- Connectedness ↔ Separateness: **middle** (regular texting, no dedicated one-on-one meetings yet)
- Certainty ↔ Uncertainty: **uncertain** (positive signals but no explicit reciprocation)

---

## Next Steps

1. Implement turning point detection in memorize flow
2. Add `relationship_arc` as a memory type or sub-type of reflection
3. Build auto-archive for outdated reflections when profile is revised
4. Consider dialectics snapshot storage (tension axes per person)
5. Test with Giao-Phương relationship as first case study

---

## References

### Books
- Knapp, M. L. (1978). *Social Intercourse: From Greeting to Goodbye*
- Baxter, L. A., & Montgomery, B. M. (1996). *Relating: Dialogues and Dialectics*
- Altman, I., & Taylor, D. A. (1973). *Social Penetration: The Development of Interpersonal Relationships*
- Gottman, J. M. (1994). *What Predicts Divorce? The Relationship Between Marital Processes and Marital Outcomes*
- Gottman, J. M. & Silver, N. (1999). *The Seven Principles for Making Marriage Work*
- Kelley, H. H., & Thibaut, J. W. (1978). *Interpersonal Relations: A Theory of Interdependence*

### Papers & Online Resources
- Fraley, R. C. — [Brief Overview of Adult Attachment Theory](https://labs.psychology.illinois.edu/~rcfraley/attachment.htm)
- Reis, H. T., Clark, M. S., & Holmes, J. G. (2004). [Perceived Partner Responsiveness as an Organizing Construct (PDF)](https://www.sas.rochester.edu/psy/people/faculty/reis_harry/assets/pdf/ReisClarkHolmes_2004.pdf)
- Reis, H. T., & Shaver, P. (1988). Intimacy as an Interpersonal Process. [PubMed](https://pubmed.ncbi.nlm.nih.gov/9599440/)
- Berger, C. R., & Calabrese, R. J. (1975). [Uncertainty Reduction Theory (Wikipedia)](https://en.wikipedia.org/wiki/Uncertainty_reduction_theory)
- Rusbult, C. E. (1980). [Investment Model of Commitment (Wikipedia)](https://en.wikipedia.org/wiki/Investment_model_of_commitment)
- Rusbult, C. E. & Buunk, B. P. (1993). [Commitment Processes: An Interdependence Analysis](https://journals.sagepub.com/doi/10.1177/026540759301000202)
- Butler, E. A. (2011). Temporal Interpersonal Emotion Systems — [Nature Review](https://www.nature.com/articles/s44220-025-00465-9)
- [State Space Grids for Team Dynamics (Frontiers)](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.00863/full)
- [Gottman's Four Horsemen](https://www.gottman.com/blog/the-four-horsemen-recognizing-criticism-contempt-defensiveness-and-stonewalling/)
- [Knapp's Relational Development Model (Wikipedia)](https://en.wikipedia.org/wiki/Knapp's_relational_development_model)
- [Relational Dialectics Theory](https://www.communicationtheory.org/relational-dialectics-theory/)
- [Social Penetration Theory (Wikipedia)](https://en.wikipedia.org/wiki/Social_penetration_theory)
- [Cascade Model of Relational Dissolution (Wikipedia)](https://en.wikipedia.org/wiki/Cascade_Model_of_Relational_Dissolution)
- [Computational Modeling of Interpersonal Dynamics — Systematic Review (Nature, 2025)](https://www.nature.com/articles/s44220-025-00465-9)
