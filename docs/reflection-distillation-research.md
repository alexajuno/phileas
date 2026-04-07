# Reflection & Distillation in AI Agent Memory Systems — Research Survey

How do AI agents convert raw episodic memories into higher-level behavioral understanding? This survey examines the mechanisms by which agent systems go from "I observed X" to "I learned principle Y," covering foundational systems (2023), recent advances (2025-2026), and implications for Phileas.

---

## 1. Stanford Generative Agents — Reflection Mechanism (UIST 2023)

### Mechanism: Importance-Triggered Recursive Reflection

The reflection system in Generative Agents is the most detailed published implementation of episodic-to-abstract synthesis. It works as follows:

**Trigger:** Reflections are generated when the cumulative sum of importance scores for recent observations exceeds a threshold (150 in their implementation). Each observation gets an importance score from 1-10 via an LLM prompt ("On the scale of 1 to 10, where 1 is purely mundane and 10 is extremely poignant, rate the likely poignancy of the following piece of memory"). In practice, agents reflect roughly 2-3 times per simulated day.

**Step 1 — Question Generation:** The system feeds the 100 most recent memory stream records to the LLM and prompts: "Given only the information above, what are the 3 most salient high-level questions we can answer about the subjects in the statements?" Example output: "What topic is Klaus Mueller passionate about?" and "What is the relationship between Klaus Mueller and Maria Lopez?"

**Step 2 — Evidence Gathering:** Each generated question is used as a retrieval query against the full memory stream (including prior reflections). The retrieval function scores memories by a weighted combination of recency (exponential decay, factor 0.995), importance (LLM-rated 1-10), and relevance (embedding cosine similarity). All weights are set to 1.0.

**Step 3 — Insight Synthesis:** Retrieved memories are fed to the LLM with the prompt: "What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))." Each insight is stored as a new reflection in the memory stream with pointers back to the cited evidence.

**Recursive Abstraction:** Reflections are stored in the same memory stream as raw observations — they are simply tagged as [Reflection] rather than [Observation]. This means reflections participate in future retrieval and can serve as evidence for higher-level reflections. The result is a "reflection tree" where leaf nodes are raw observations and interior nodes are progressively more abstract reflections. For example: observations about Klaus reading papers, discussing research, and visiting the library consolidate into "Klaus Mueller is dedicated to research," which further abstracts to "Klaus Mueller is highly dedicated to research."

**Storage:** Reflections live alongside observations in the memory stream with identical structure (natural language description, creation timestamp, last access timestamp, importance score). They are retrieved via the same scoring function.

**Cross-domain Transfer:** Reflections are agent-scoped (about "me" or "others I know"), not task-scoped. They naturally generalize across contexts — a reflection about a relationship applies wherever that relationship is relevant.

**Limitations:**
- Reflections are static once generated — they are never revised or invalidated
- The importance threshold is hand-tuned and domain-dependent
- No principled forgetting — the memory stream only grows
- Reflection quality depends heavily on LLM capability
- Tested only in a small sandbox (25 agents, 2 simulated days)

**Link:** https://arxiv.org/abs/2304.03442

---

## 2. Reflexion — Verbal Self-Reflection (NeurIPS 2023)

### Mechanism: Failure-Driven Verbal Reinforcement

Reflexion converts environmental feedback (binary success/fail, scalar rewards, or free-form language) into natural language self-reflections that serve as "semantic gradient signals" for future attempts.

**Architecture:** Three models collaborate — an Actor (generates actions), an Evaluator (scores outcomes), and a Self-Reflection model (generates verbal feedback). The Actor also has access to a persistent memory buffer `mem`.

**The Reflexion Loop:**
1. Actor generates trajectory tau_0 by interacting with the environment
2. Evaluator produces a score r_0 (binary, scalar, or verbal)
3. Self-Reflection model analyzes {tau_0, r_0} to produce a verbal summary sr_0
4. sr_0 is appended to memory `mem`
5. On the next trial, the Actor conditions on both its current trajectory (short-term memory) and `mem` (long-term memory)
6. Loop repeats until the Evaluator accepts the trajectory or max trials reached

**What Reflections Look Like:** In the decision-making domain (AlfWorld), a reflection might be: "I tried to pick up the pan from stoveburner 1 but I had already put it in stoveburner 1. I should check whether I already have the item before trying to pick it up." In programming (HumanEval), reflections identify specific bugs and propose fixes.

**Memory Management:** The memory buffer is bounded — typically limited to the last 1-3 reflections (Omega = 1-3) to fit within context window limits. This is a sliding window, not a growing store. There is no deduplication, no merging, no hierarchical abstraction.

**Trigger:** Reflection is triggered by task failure. The system only reflects when the Evaluator rejects a trajectory. There is no time-based or count-based trigger.

**Transfer:** Reflections are strictly task-scoped. They accumulate only within repeated attempts at the same task. There is no mechanism for transferring lessons learned on task A to task B. The authors explicitly acknowledge this as a limitation and suggest future work using vector databases or SQL for cross-task memory.

**Limitations:**
- No cross-task transfer (the fundamental limitation)
- Sliding window loses older reflections
- Depends on LLM self-evaluation quality (no formal guarantee)
- Can converge to local optima
- Memory never consolidates — reflections are just appended text

**Link:** https://arxiv.org/abs/2303.11366

---

## 3. VOYAGER — Skill Library (2023)

### Mechanism: Code-as-Distilled-Knowledge

VOYAGER takes a fundamentally different approach: instead of distilling text reflections, it distills executable code programs as reusable skills.

**Skill Creation Pipeline:**
1. An automatic curriculum proposes a task based on the agent's current state, inventory, biome, and exploration progress (GPT-4 with temperature 0.1 for diversity)
2. GPT-4 generates executable JavaScript code to accomplish the task, prompted with: control primitive APIs, relevant existing skills from the library, environment feedback, execution errors, current state, and chain-of-thought reasoning
3. The code is executed in the Minecraft environment
4. An iterative refinement loop incorporates three types of feedback: (a) environment feedback ("I cannot make stick because I need 2 more planks"), (b) execution errors from the code interpreter, (c) self-verification by a separate GPT-4 instance that confirms task completion or suggests fixes
5. When self-verification passes, the program is committed to the skill library

**Skill Storage:** Each skill is stored as a key-value pair in a vector database. The key is the embedding of a natural language description of the program (generated by GPT-3.5). The value is the executable code itself. This dual representation enables semantic retrieval by description while preserving exact executable behavior.

**Skill Retrieval:** When a new task arrives, GPT-3.5 generates a general solution description, which is combined with environment context into a query. The top-5 most similar skills are retrieved from the library and provided as in-context examples for code generation.

**Compositionality:** Skills explicitly compose — complex skills call simpler ones. The prompt instructs GPT-4: "Your function will be reused for building more complex functions. Therefore, you should make it generic and reusable." For example, `craftStoneShovel()` calls `craftItem()` which calls `mineBlock()`.

**Transfer:** The skill library transfers across Minecraft worlds. When placed in a new world, VOYAGER can reuse previously learned skills to solve novel tasks from scratch, while baselines (ReAct, Reflexion, AutoGPT) fail completely. The library can even boost other methods — giving AutoGPT access to VOYAGER's skill library improves its performance.

**What Triggers Skill Creation vs. Reuse:** If retrieved skills are sufficient for the current task, they are reused. If the task requires new behavior, new code is generated (potentially composing existing skills). The automatic curriculum ensures tasks progressively increase in difficulty.

**Limitations:**
- Cost: GPT-4 API is 15x more expensive than GPT-3.5
- Hallucinations: the curriculum sometimes proposes unachievable tasks ("copper sword" does not exist in Minecraft)
- Self-verification can fail (not recognizing success)
- Skills are domain-specific (Minecraft JavaScript APIs)
- No mechanism for skill revision — once committed, skills are never updated

**Link:** https://arxiv.org/abs/2305.16291

---

## 4. ExpeL — Experiential Learning (NeurIPS 2023)

### Mechanism: Cross-Task Insight Extraction with Voting

ExpeL is the most directly relevant system for Phileas because it explicitly extracts cross-task principles from accumulated experience — not just within-task reflections.

**Three-Phase Pipeline:**

**Phase 1 — Experience Gathering:** The agent attempts training tasks using ReAct (reasoning + action). For each task, it gets up to Z retries. Successful trajectories and success/failure comparison pairs are stored in an experience pool (Faiss vectorstore with `all-mpnet-base-v2` embeddings). Failed attempts include Reflexion-style self-reflection before retrying.

**Phase 2 — Insight Extraction:** The LLM processes experiences in two ways:
- **From success/failure pairs** (same task): "Here are two previous task trials — one successful, one unsuccessful. You failed because [reasons]. Examine the trials and critique to avoid similar failures. Have an emphasis on critiquing to perform better Thought and Action."
- **From lists of successes** (different tasks): "Here are successful task trials. Examine them for common 'good practices' the agent can adopt."

The LLM operates on an existing list of insights using four operators: ADD (new insight), EDIT (improve existing), UPVOTE (agree — increment importance count), DOWNVOTE (disagree — decrement importance count). If an insight's count reaches zero, it is removed. This voting mechanism robustifies the process against suboptimal or misleading trajectories.

**Phase 3 — Task Inference:** For new tasks, the agent receives: (a) the full list of extracted insights concatenated, (b) top-k most similar successful trajectories retrieved from the experience pool via task similarity.

**Insight Examples:** On HotpotQA: "Consider the answer might be in the observations already made." On ALFWorld: "When searching for an item, consider its nature and its typical usage." These are genuinely cross-task principles, not task-specific patches.

**Transfer Learning:** ExpeL explicitly demonstrates cross-domain transfer. Insights extracted from HotpotQA (question answering) are "finetuned" via a transfer prompt and applied to FEVER (fact verification), improving performance from 63% (ReAct) to 70% (ExpeL Transfer). The transfer prompt: "You are a teacher agent that passes on experience to student agents. You came up with the following rules to help you achieve the task of [Source Task]. Now a student is trying to solve a similar [Target Task]."

**Emergent Behaviors:** The paper documents unexpected capabilities that emerged from accumulated insights:
- Hypothesis formulation: the agent learns to reassess its whole trajectory rather than giving up
- World model belief updates: the agent changes its priors about where objects are located based on extracted insights
- Self-correction: the agent develops the ability to identify and recover from missteps mid-task

**Limitations:**
- Text-only observations (no visual)
- Insights currently fit within context window — as they grow, retrieval would be needed
- Prompting techniques lack theoretical underpinnings
- Success/failure comparison requires paired data
- Insight quality depends heavily on LLM quality (GPT-4 >> GPT-3.5-turbo for insight extraction)

**Link:** https://arxiv.org/abs/2308.10144

---

## 5. A-MEM — Agentic Memory with Zettelkasten (NeurIPS 2025)

### Mechanism: Dynamic Note Evolution and Linking

A-MEM focuses on the organizational layer — how memories relate to each other and evolve — rather than explicit reflection or insight extraction.

**Note Construction:** Each memory is stored as a structured note with six components: content c_i (raw interaction), timestamp t_i, keywords K_i (LLM-generated), tags G_i (LLM-generated categories), contextual description X_i (LLM-generated semantic enrichment), embedding e_i (dense vector of concatenated textual components), and links L_i (connections to related notes).

**Link Generation:** When a new note m_n is created:
1. Compute cosine similarity between m_n's embedding and all existing notes
2. Retrieve top-k most similar notes
3. Prompt the LLM to analyze potential connections based on shared attributes, causal relationships, and conceptual connections (not just embedding similarity)
4. Establish links, grouping related notes into "boxes" (Zettelkasten concept)

**Memory Evolution:** When a new note arrives, related historical notes are updated. The LLM evaluates whether each linked note's context, keywords, and tags should change given the new information: m_j* <- LLM(m_n, neighbors, m_j, prompt). The evolved note replaces the original. This is the key differentiator from append-only systems — existing memories actively change based on new experience.

**Retrieval:** Query-based cosine similarity retrieval. When a memory is retrieved, linked memories in the same "box" are also surfaced, enabling multi-hop reasoning across related memories.

**Does It Create Abstractions?** Partially. The memory evolution process can produce higher-order patterns as the contextual descriptions of existing notes become enriched with patterns discovered across multiple experiences. However, A-MEM does not explicitly generate "insight" or "principle" nodes. The abstraction is implicit — distributed across evolved note descriptions — rather than stored as separate higher-level entities.

**Limitations:**
- Memory quality depends on LLM capability (different LLMs produce different organizations)
- Text-only (no multimodal)
- No explicit forgetting mechanism
- LLM calls for every memory operation add cost (though only ~1,200 tokens per operation vs. 16,900 for baselines)
- No explicit reflection or insight extraction step

**Link:** https://arxiv.org/abs/2502.12110

---

## 6. Recent Work (2025-2026): The Consolidation Frontier

### TiMem — Temporal-Hierarchical Memory Consolidation (January 2026)

The most complete episodic-to-semantic consolidation framework found. TiMem organizes memory into a five-level Temporal Memory Tree:

| Level | Scope | Content | Trigger |
|-------|-------|---------|---------|
| L1 | Segment | Fine-grained dialog details | After each dialog turn (online) |
| L2 | Session | Merged non-redundant event summaries | When session ends |
| L3 | Daily | Routine contexts and interests | When day ends |
| L4 | Weekly | Behavioral patterns and preferences | When week ends |
| L5 | Profile | Stable personality traits and values | Monthly |

Each level uses level-specific LLM prompts: lower levels perform factual summarization, middle levels extract behavioral patterns, and the profile level synthesizes long-term characteristics. Historical context windows of three prior memories per level maintain consolidation continuity.

**Complexity-Aware Retrieval:** A recall planner classifies queries as simple (factual + profile layers), hybrid (partial pattern layers), or complex (full hierarchy traversal). This prevents information overload while ensuring sufficient context.

**Link:** https://arxiv.org/abs/2601.02845

### MemRL — Self-Evolving Agents via Runtime RL on Episodic Memory (January 2026)

MemRL introduces reinforcement learning to memory retrieval itself. Rather than passively matching by semantic similarity, it learns Q-values for memory entries based on whether retrieving them actually helped task performance.

**Two-Phase Retrieval:** (1) Filter candidates by semantic relevance, (2) Select from candidates based on learned Q-values (utility scores from environmental feedback). This addresses the noise problem in naive similarity-based retrieval — a memory might be semantically similar but behaviorally irrelevant.

**Key insight for Phileas:** Memory utility is not the same as memory similarity. A system that tracks which retrieved memories actually led to good outcomes can preferentially surface useful ones.

**Link:** https://arxiv.org/abs/2601.03192

### MACLA — Hierarchical Procedural Memory (December 2025)

MACLA compresses 2,851 ALFWorld training trajectories into 187 reusable procedures (93% compression) through:
- Semantic abstraction and duplicate detection
- Bayesian posteriors tracking procedure reliability
- Contrastive refinement by comparing successes and failures
- Meta-procedural composition of complex behaviors from simpler ones

Achieves 78.1% average across benchmarks using only a 7B model — demonstrating that well-organized procedural memory can substitute for model scale.

**Link:** https://arxiv.org/abs/2512.18950

### Mem^p (Memp) — Agent Procedural Memory (August 2025)

Distills experiences into two levels of abstraction: fine-grained step-by-step instructions and higher-level script-like abstractions. The memory repository continuously updates, corrects, and deprecates its contents based on new experiences. Procedural memory transfers across models — weaker agents benefit from stronger agents' distilled memories.

**Link:** https://arxiv.org/abs/2508.06433

### SKILLRL — Recursive Skill-Augmented RL (2026)

Evolves VOYAGER's approach: collects trajectories, distills them into a hierarchical skill library, performs cold-start SFT (supervised fine-tuning) to enable skill utilization, then conducts RL training with dynamic skill evolution based on validation failures. Skills that fail validation are revised rather than simply accumulated.

**Link:** https://arxiv.org/abs/2602.08234

### ERL — Experience Reflection Learning (ICLR 2026 MemAgents Workshop)

Extracts heuristics from single-attempt trajectories (no need for repeated execution or curated training sets). Heuristics are retrieved by task similarity at test time and injected into context. Preserves granular trajectory details that cross-task aggregation methods (like ExpeL) tend to lose.

**Link:** https://arxiv.org/abs/2603.24639

---

## 7. Synthesis: Patterns Across Systems

### The Spectrum of Distillation

These systems form a clear spectrum from task-specific to cross-domain generalization:

| System | Raw Input | Distilled Output | Scope | Transfer? |
|--------|-----------|-----------------|-------|-----------|
| Reflexion | Failed trajectory + feedback | Verbal self-reflection text | Single task | No |
| VOYAGER | Successful action sequence | Executable code program | Domain (Minecraft) | Within domain |
| ExpeL | Success/failure trajectory pairs | Natural language principles | Cross-task | Yes (with finetuning prompt) |
| Generative Agents | Stream of observations | Higher-level reflections | Agent-wide | Naturally |
| A-MEM | Raw interactions | Evolved note descriptions | Agent-wide | Implicit |
| TiMem | Conversations over time | Hierarchical persona model | Agent-wide | Naturally |
| MACLA | Training trajectories | Bayesian-scored procedures | Cross-task | Yes |

### Three Synthesis Triggers

1. **Failure-driven** (Reflexion, ExpeL): Reflect only when something goes wrong. Efficient but misses positive patterns.
2. **Threshold-driven** (Generative Agents): Reflect when enough "important" events accumulate. Catches both positive and negative patterns but requires tuning.
3. **Time-driven** (TiMem): Consolidate on schedule (session end, day end, week end). Systematic but may over-process quiet periods.

### Two Storage Strategies

1. **Same-stream** (Generative Agents, Reflexion): Reflections live alongside raw memories, distinguished only by tag. Simple but mixes abstraction levels during retrieval.
2. **Separate-store** (ExpeL, VOYAGER, TiMem, MACLA): Insights/skills/procedures stored in dedicated structures with different retrieval logic. Cleaner but more complex architecture.

### The Voting/Validation Pattern

Multiple systems independently converge on importance scoring for distilled knowledge:
- ExpeL uses UPVOTE/DOWNVOTE/EDIT/ADD operators with count-based removal
- MACLA uses Bayesian posteriors tracking reliability
- MemRL uses Q-values from environmental feedback
- VOYAGER uses self-verification before committing skills

This suggests raw distillation is not enough — the system needs a mechanism to validate, reinforce, or deprecate insights over time.

---

## 8. Implications for Phileas

### The Gap

Phileas currently stores episodic memories (events, facts, preferences) and retrieves them by semantic similarity. It has no mechanism for:
1. Synthesizing patterns across multiple memories ("I've seen this user struggle with X three times")
2. Extracting behavioral principles ("For multi-session projects, setting up task tracking early prevents losing progress")
3. Validating or deprecating insights as new evidence arrives
4. Distinguishing between memories that are semantically similar and memories that are actually useful

### Concrete Patterns to Adopt

**Pattern 1: Scheduled Consolidation (from TiMem)**

The most natural fit for Phileas. At session boundaries (which Phileas already detects via `mark_session_done`), run a consolidation step:

1. Retrieve the session's memories
2. Retrieve related memories from previous sessions
3. Prompt the LLM: "Given these experiences across N sessions, what behavioral patterns, preferences, or principles can you identify?"
4. Store the result as a special memory type (e.g., `kind: insight` alongside existing `kind: episodic`)
5. Insights participate in future recall alongside episodic memories

This maps to TiMem's L2-L3 levels. No need for the full five-level hierarchy initially.

**Pattern 2: Success/Failure Comparison (from ExpeL)**

When the user explicitly marks outcomes ("that worked" / "that didn't work" / session completion signals), Phileas can compare:
- What was the approach in session A (success) vs. session B (failure)?
- What was different about the context, tools used, or sequence of actions?
- Extract a principle: "When doing X, approach Y works better than Z because..."

This requires tracking session outcomes, which `mark_session_done` partially provides but could be enriched.

**Pattern 3: Insight Voting (from ExpeL + MACLA)**

Give insights an importance score that changes over time:
- When a recalled insight leads to a conversation where the user confirms it helped: increment
- When new evidence contradicts an insight: decrement or edit
- When an insight's score drops to zero: deprecate it
- This prevents stale or wrong insights from persisting indefinitely

**Pattern 4: Reflection Trees (from Generative Agents)**

Allow insights to reference the memories they were derived from (evidence pointers). This enables:
- Traceability: "Why does Phileas think I prefer X?" -> "Because of memories M1, M5, M12"
- Recursive abstraction: insights about insights ("Overall, this user values pragmatic solutions over theoretical elegance")
- Debugging: if an insight seems wrong, inspect its evidence

### Minimum Viable Reflection (Solving "I finished CI but Phileas doesn't know")

The core problem: Phileas stores episodic memories about individual sessions but never synthesizes them into understanding about completed projects, learned skills, or behavioral patterns. The user finishes setting up CI across three sessions, but Phileas only knows about each session individually — it has no memory that says "The user now knows how to set up CI" or "The CI project is complete."

**MVP implementation:**

1. **Trigger:** After `mark_session_done` or after `consolidate` is called
2. **Gather:** Retrieve all memories from the completed session + related memories from the graph
3. **Synthesize:** Prompt the LLM with a reflection prompt:
   ```
   Given these memories from a session and related past context:
   [memories]

   Extract any of the following that apply:
   - Completed milestones or projects
   - Skills or knowledge the user demonstrated or acquired
   - Patterns in how the user works (preferences, habits, effective strategies)
   - Corrections to previous understanding

   Format each as a concise principle with evidence references.
   ```
4. **Store:** Save as `kind: insight` memories with links to source memories in the graph
5. **Retrieve:** Include insights in `recall` results when they match the query context

**What this solves:**
- "I finished CI" -> consolidation produces insight: "User completed CI setup for project X using GitHub Actions (sessions 12-14). Key decisions: chose matrix strategy for multi-version testing."
- "I used backlog and it helped" + "I didn't use backlog and lost track" -> consolidation produces insight: "For multi-session projects, setting up task tracking early prevents losing progress (evidence: sessions 8, 15)."
- Next time the user starts a new project, recalling "starting a project" surfaces these insights alongside raw memories.

**Cost:** One additional LLM call per session close. The consolidation prompt is small (session memories + related context). At Phileas's current usage volume, this adds negligible overhead.

### What to Defer

- **Full five-level TiMem hierarchy:** Overkill for a personal memory system that processes a few sessions per day. Start with session-level consolidation and add higher levels (weekly/monthly) only if needed.
- **Procedural memory / executable skills (VOYAGER, MACLA):** Relevant for coding agents but not for a memory companion. Phileas stores understanding, not procedures.
- **RL-based retrieval scoring (MemRL):** Powerful but requires explicit feedback signals that Phileas doesn't currently collect. Add later when there's a way to measure "did this recalled memory actually help?"
- **Bayesian reliability tracking (MACLA):** Interesting for insight validation but premature — simple importance counts (ExpeL-style) are sufficient initially.

---

## 9. Key References

| System | Venue | Year | Link |
|--------|-------|------|------|
| Generative Agents | UIST 2023 | 2023 | https://arxiv.org/abs/2304.03442 |
| Reflexion | NeurIPS 2023 | 2023 | https://arxiv.org/abs/2303.11366 |
| VOYAGER | arxiv | 2023 | https://arxiv.org/abs/2305.16291 |
| ExpeL | NeurIPS 2023 | 2023 | https://arxiv.org/abs/2308.10144 |
| A-MEM | NeurIPS 2025 | 2025 | https://arxiv.org/abs/2502.12110 |
| TiMem | arxiv | 2026 | https://arxiv.org/abs/2601.02845 |
| MemRL | arxiv | 2026 | https://arxiv.org/abs/2601.03192 |
| MACLA | arxiv | 2025 | https://arxiv.org/abs/2512.18950 |
| Mem^p | arxiv | 2025 | https://arxiv.org/abs/2508.06433 |
| SKILLRL | arxiv | 2026 | https://arxiv.org/abs/2602.08234 |
| ERL | ICLR MemAgents 2026 | 2026 | https://arxiv.org/abs/2603.24639 |
| "Memory in the Age of AI Agents" (survey) | arxiv | 2025 | https://arxiv.org/abs/2512.13564 |
| "Episodic Memory is the Missing Piece" (position) | arxiv | 2025 | https://arxiv.org/abs/2502.06975 |
