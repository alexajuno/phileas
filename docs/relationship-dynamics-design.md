# Relationship Dynamics — Technical Design

Extends Phileas's memory architecture to model interpersonal relationships as evolving, multi-dimensional processes rather than static snapshots.

**Research foundation:** See [relationship-dynamics-research.md](./relationship-dynamics-research.md) for the 9 psychological frameworks informing this design.

---

## Design Principles

1. **One living picture per relationship.** Not an accumulation of frozen snapshots. When new data arrives, revise the picture.
2. **Don't predict.** Store what IS, not what will be. Relationships shift in ways no model can anticipate.
3. **Separate facts from interpretation.** Events are immutable. Interpretations (reflections) are revisable.
4. **Track dimensions, not labels.** A relationship is a position in multi-dimensional space, not a single label like "close" or "distant."

---

## Data Model

### Relationship Profile (extends existing `profile` memory type)

One per person. Contains only facts and current dynamic. Revised on turning points.

```
{
  "memory_type": "profile",
  "summary": "Alice — engineer at BigCo. Met at a meetup. Currently friendly, occasional texting.",
  "person_handle": "@alice",
  "dimensions": {
    "disclosure_depth": 0.6,       // 0-1, how deep conversations go (Social Penetration)
    "reciprocity": 0.3,            // -1 to 1, who initiates more (-1=them, 0=balanced, 1=me)
    "uncertainty": 0.7,            // 0-1, how ambiguous the relationship feels (URT)
    "responsiveness": 0.8,         // 0-1, quality of their responses (Reis & Clark)
    "bid_acceptance": 0.9,         // 0-1, how often they "turn toward" bids (Gottman)
    "investment": 0.5,             // 0-1, sunk emotional/time investment (Rusbult)
  },
  "tensions": {                    // Baxter's dialectics, each 0-1
    "openness_closedness": 0.6,    // 0=closed, 1=open
    "connection_autonomy": 0.4,    // 0=autonomous, 1=connected
    "novelty_predictability": 0.5  // 0=predictable, 1=novel
  }
}
```

**Revision rule:** When a turning point event is stored, the profile dimensions and tensions are updated in the same operation. Old version archived via SUPERSEDES edge.

### Relationship Events (extends existing `event` memory type)

Immutable. Each significant interaction logged with metadata.

```
{
  "memory_type": "event",
  "summary": "Late-night text about feeling lost. She replied with thoughtful personal advice.",
  "event_meta": {
    "initiated_by": "user",                  // "user" | "other" | "mutual"
    "disclosure_depth": "deep",              // "surface" | "mid" | "deep" | "core"
    "reciprocity_match": true,               // did both sides go to similar depth?
    "reward_signal": "positive",             // "positive" | "neutral" | "negative"
    "uncertainty_impact": "reduced",         // "reduced" | "maintained" | "increased"
    "responsiveness_quality": "high",        // "high" | "medium" | "low"
    "is_turning_point": true,                // does this shift the relationship dynamic?
    "bid_type": "emotional",                 // "emotional" | "informational" | "social" | null
    "bid_response": "turn_toward"            // "turn_toward" | "turn_away" | "turn_against" | null
  }
}
```

### Relationship Arc (new: sub-type of `reflection`)

Periodic synthesis of the relationship trajectory. Replaces scattered reflections.

```
{
  "memory_type": "reflection",
  "sub_type": "relationship_arc",
  "summary": "Boundary (Nov) → Distance (Mar) → Reset (late Mar) → Warmth (Apr). Pattern: when user slows down and matches pace, warmth returns.",
  "person_handle": "@alice",
  "phases": [
    {"label": "boundary", "period": "2025-11", "trigger": "user escalated too fast"},
    {"label": "distance", "period": "2026-03", "trigger": "lectures instead of responsiveness"},
    {"label": "reset", "period": "2026-03-30", "trigger": "user adopted lighter approach"},
    {"label": "warmth", "period": "2026-04", "trigger": "genuine reciprocal sharing"}
  ]
}
```

**Revision rule:** New arc supersedes old arc when a new turning point creates a new phase. Arc is the consolidated narrative — only one active arc per relationship at a time.

---

## Dimensions: What We Track and Why

### From Social Penetration Theory

**Disclosure Depth** (0-1): How deep do conversations go?

```
0.0-0.2  surface     "how's work?" "nice weather"
0.2-0.4  mid         "work is stressful" "I like this"
0.4-0.7  deep        "I feel lonely" "I don't know where I belong"
0.7-1.0  core        identity, fears, dreams, wounds
```

Tracked per event. Profile stores the rolling average or recent high-water mark.

### From Uncertainty Reduction Theory

**Uncertainty Level** (0-1): How ambiguous does this relationship feel?

Decreased by: warm replies, direct communication, mutual disclosure.
Maintained by: emoji reactions without words, short replies, ambiguous signals.
Increased by: silence, avoidance, contradictory signals.

Each event tagged with `uncertainty_impact`. Profile uncertainty = trend.

### From Gottman

**Bid Acceptance Rate** (0-1): When one person reaches out (a "bid"), does the other engage?

Bid types:
- **Emotional:** sharing feelings, seeking comfort
- **Informational:** asking about their day, their work
- **Social:** inviting to do something together

Responses:
- **Turn toward:** engages, responds with interest
- **Turn away:** ignores, doesn't reply, changes subject
- **Turn against:** rejects, criticizes the bid

Profile stores the running ratio. Gottman's research: stable relationships = 86% turn-toward rate.

### From Reis & Clark

**Responsiveness Quality** (0-1): When they respond, do they actually SEE you?

Three components scored per interaction:
- Understanding: do they grasp what you're saying?
- Validation: do they respect your perspective?
- Caring: do they respond to your actual needs?

Lectures and generic advice = low responsiveness (addresses the topic but not the person).
Thoughtful personal reflection that mirrors vulnerability = high responsiveness.

### From Rusbult (Investment Model)

**Investment Level** (0-1): How much has been put into this relationship?

Factors:
- Time spent (duration of relationship)
- Emotional investment (vulnerability shared, emotional energy spent)
- Shared experiences
- Social integration (mutual friends, shared spaces)

Investment creates inertia — explains why people persist through rough phases.

### From Baxter (Relational Dialectics)

**Tension Axes** (each 0-1):

```
Openness ←————→ Closedness
    How much are we sharing vs. protecting?

Connection ←————→ Autonomy
    How intertwined are our lives vs. independent?

Novelty ←————→ Predictability
    How much surprise vs. routine?
```

These are NOT on a good-bad spectrum. Both sides are healthy. The interesting signal is how they shift over time.

### From Attachment Theory

**Not stored as a dimension on the relationship.** Instead, the user's attachment patterns are stored as a `behavior` memory on their own profile:

```
{
  "memory_type": "behavior",
  "summary": "Attachment pattern trending from anxious-preoccupied toward secure. Evidence: handles delayed replies with less spiraling than before. Some anxiety still surfaces but is managed."
}
```

This contextualizes how the user interprets relationship events — it's about the user, not about the relationship.

---

## Recall Behavior Changes

### Current: flat recall

All memories about a person returned by relevance score, mixing old pain with current warmth.

### Proposed: layered recall

When recalling about a person:

```
Layer 1: Relationship Profile          ← always first (current state)
Layer 2: Relationship Arc              ← trajectory context
Layer 3: Recent events (last 30 days)  ← what's happening now
Layer 4: Older events                  ← only if explicitly requested
Layer 5: Archived reflections          ← only if explicitly requested
```

**Implementation:** The `recall` tool, when it detects a query is about a person (entity type = Person in KuzuDB), applies this layered ordering:

1. Check if query matches a Person entity in the graph
2. If yes, fetch their profile memory (boost score by +0.3)
3. Fetch their relationship arc (boost score by +0.2)
4. Apply recency boost for events within 30 days (+0.1)
5. Apply recency penalty for reflections older than 60 days (-0.2)

This ensures the current picture surfaces first without deleting historical data.

### Stale Reflection Handling

When a relationship profile is updated (turning point detected):

1. Find all `reflection` memories linked to that person
2. For each reflection, check if it contradicts the new profile
3. If contradicted: archive with reason "superseded by profile update on {date}"
4. The reflection is preserved in archive but no longer surfaces in recall

**This is a client-side operation** — Claude Code performs it when updating a profile, not the server.

---

## Turning Point Detection

An event is a turning point if any of the following:

1. **Dimension shift:** one or more dimensions change by > 0.2 compared to the profile's current value
2. **Phase contradiction:** the event's tone contradicts the current arc phase (e.g., warmth during a "distance" phase)
3. **User signal:** the user uses emotionally charged language about the interaction
4. **First occurrence:** first time the other person initiates, first in-person meeting, first conflict, etc.

When a turning point is detected:
1. Store the event with `is_turning_point: true`
2. Update the relationship profile dimensions
3. Append a new phase to the relationship arc
4. Archive contradicted reflections

---

## Implementation Plan

### Phase 1: Event Metadata (low effort, high value)

Add `event_meta` fields to the skill instructions. Claude Code starts tagging events with:
- `initiated_by`
- `disclosure_depth`
- `reward_signal`
- `is_turning_point`

No server changes needed — metadata goes into the summary or as entities.

### Phase 2: Relationship Dimensions on Profile

Extend profile memories with dimension fields. Start with the three most implementable:
- `disclosure_depth` (from Social Penetration)
- `reciprocity` (who initiates)
- `uncertainty` (from URT)

Server change: none (dimensions stored in summary text or as structured JSON in a new column).

### Phase 3: Layered Recall

Modify the `recall` scoring formula to apply person-aware boosts:
- Profile memories: +0.3
- Recent events (30d): +0.1
- Old reflections (60d+): -0.2

Server change: update `engine.py` recall pipeline.

### Phase 4: Relationship Arc

Add arc synthesis to the consolidation flow. When consolidating memories about a person:
- Instead of generic cluster summaries, produce a relationship arc
- One active arc per person, supersedes previous

Server change: extend `consolidate` tool with person-aware clustering.

### Phase 5: Stale Reflection Cleanup

Automate the "archive contradicted reflections" flow:
- On profile update, Claude Code queries for old reflections about the same person
- Archives those that contradict the new profile
- Links via SUPERSEDES edge

Client-side only — update skill instructions.

### Phase 6: Tension Axes & Full Dimensions

Add Baxter's dialectic tensions and remaining dimensions. This is the most speculative phase — may need iteration to find the right granularity.

---

## Open Questions

1. **Where to store dimensions?** Options:
   - In the summary text (human-readable, no schema change)
   - In a new JSON column on memory_item (structured, queryable)
   - As properties on the Person node in KuzuDB (graph-native)

2. **How to compute dimension values?** Options:
   - Claude Code estimates from conversation context (current approach for importance)
   - Derived from event metadata averages (more systematic but requires enough data)
   - Hybrid: Claude estimates, but metadata provides ground truth over time

3. **Arc granularity:** How many phases before an arc becomes unwieldy? Should old phases be pruned?

4. **Cross-relationship patterns:** Should the system detect that the user behaves similarly across multiple relationships? (e.g., always escalates too fast) This is a behavior-type memory about the user, not about any single relationship.

---

## Non-Goals

- **Predicting relationship outcomes.** The system describes, it doesn't forecast.
- **Scoring relationships.** No "relationship health score." Dimensions are descriptive, not evaluative.
- **Automated advice.** The system provides context; Claude Code provides counsel.
- **Modeling the other person's internal state.** We only have the user's perspective. Dimensions reflect perceived behavior, not ground truth.
