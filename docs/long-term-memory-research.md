# Long-Term Memory for AI Companions — Research Survey

## 1. Foundational & Landmark Systems

### Stanford Generative Agents (2023)
- **Core idea:** Agents maintain a *memory stream* — a timestamped log of all observations in natural language. Three operations: *observation* (recording events), *reflection* (periodically synthesizing higher-level abstractions), and *planning* (generating future actions based on retrieved memories).
- **Retrieval:** Scored by weighted combination of *recency* (exponential decay), *importance* (LLM-rated 1-10), and *relevance* (embedding cosine similarity to current context).
- **Limitations:** Expensive (every agent queries the LLM frequently); reflections are static once generated; no principled forgetting — the stream only grows; tested only in a small sandbox (25 agents).
- **Venue:** UIST 2023 (ACM)
- **Link:** https://arxiv.org/abs/2304.03442

### MemGPT / Letta (2023-2024)
- **Core idea:** Treats the LLM's context window like an OS treats RAM. Two-tier memory hierarchy: *main context* (in-context working memory) and *external context* (archival storage in a vector DB + recall storage of conversation logs). The LLM itself issues function calls to page data in and out.
- **Storage:** Main context holds a system prompt with a self-editable "core memory" block (user facts, persona). Archival storage uses vector-indexed passages. Recall storage keeps raw conversation history.
- **Retrieval:** The LLM decides when to search archival/recall storage via tool calls. Eviction from main context is managed by recursive summarization of older messages.
- **Limitations:** Relies heavily on the LLM's own judgment for memory management (which can be unreliable); recursive summarization loses detail over time; latency from multiple LLM calls per turn.
- **Venue:** arxiv Oct 2023; Letta is the open-source framework (2024+)
- **Link:** https://arxiv.org/abs/2310.08560
- **Docs:** https://docs.letta.com/concepts/memgpt/

### Reflexion (2023)
- **Core idea:** Agents learn from failures by generating *verbal self-reflections* stored in an episodic memory buffer. On subsequent attempts at a task, these reflections are retrieved and included in the prompt.
- **Memory:** A sliding window of natural-language reflection strings, not raw observations.
- **Limitations:** Task-scoped (reflections don't transfer across tasks); buffer grows without principled pruning; no long-term consolidation.
- **Venue:** NeurIPS 2023
- **Link:** https://arxiv.org/abs/2303.11366

---

## 2. Memory Banks & Forgetting Mechanisms

### MemoryBank / SiliconFriend (2023)
- **Core idea:** Three-component memory system — *storage* (daily conversation summaries, event summaries, user personality profiles), *retrieval* (vector-encoded for similarity search), and *memory intensity update* (Ebbinghaus forgetting curve).
- **Forgetting:** Each memory has an activation score that decays exponentially over time. Memories that are recalled get reinforced (strength increases); unused ones fade. Directly modeled on Ebbinghaus's exponential decay formula.
- **Application:** Built "SiliconFriend," a long-term AI companion fine-tuned on psychological dialogue data.
- **Limitations:** Forgetting curve parameters are hand-tuned; summarization may lose nuance; personality modeling is coarse.
- **Venue:** AAAI 2024
- **Link:** https://arxiv.org/abs/2305.10250

### ACT-R Inspired Memory Architecture (2024)
- **Core idea:** Applies the ACT-R cognitive architecture's *base-level activation* equation to LLM agent memory. Each memory chunk has an activation score combining *frequency* (how often accessed), *recency* (time decay), *contextual relevance* (embedding similarity), and *importance* (LLM-judged).
- **Forgetting:** Chunks below an activation threshold are suppressed or discarded, following ACT-R's power-law decay. Resembles human memory where unused information naturally becomes inaccessible.
- **Limitations:** Activation parameters need calibration; no mechanism for memory consolidation or abstraction.
- **Venue:** HAI 2024 (ACM Conference on Human-Agent Interaction)
- **Link:** https://dl.acm.org/doi/10.1145/3765766.3765803

---

## 3. RAG-Based & Knowledge Graph Approaches

### HippoRAG (2024)
- **Core idea:** Inspired by hippocampal indexing theory. Uses an LLM to extract knowledge graph triples from passages, stores them in a KG, then retrieves via *Personalized PageRank* over the graph. The LLM acts as the "neocortex" (pattern completion), the KG as the "hippocampal index."
- **Retrieval:** Given a query, the LLM identifies key entities, which seed a Personalized PageRank walk over the KG. Enables multi-hop reasoning across document boundaries — something standard RAG cannot do.
- **Limitations:** KG extraction quality depends on the LLM; graph can grow large; not designed for conversational/personal memory specifically.
- **Venue:** NeurIPS 2024
- **Link:** https://arxiv.org/abs/2405.14831

### MemoRAG (2024)
- **Core idea:** Augments RAG with a lightweight "global memory" model that reads the entire long context and forms a compressed representation. When a query arrives, this global memory generates "clue" queries that guide the retriever, rather than relying solely on the original query.
- **Retrieval:** Two-stage: the global memory model drafts retrieval clues, then a standard retriever fetches relevant chunks.
- **Limitations:** Requires training the global memory model; may not scale to very dynamic/personal memory scenarios.
- **Venue:** ACM Web Conference 2025
- **Link:** https://arxiv.org/abs/2409.05591

### Zep / Graphiti (2025)
- **Core idea:** A *temporal knowledge graph* engine for agent memory. Unlike static KGs, Graphiti maintains temporal edges — facts have valid-from/valid-to timestamps, enabling the system to track how knowledge evolves over time (e.g., "user moved from city A to city B in March 2024").
- **Storage:** Neo4j-based graph with entities and temporally-scoped relationships. Ingests both unstructured conversation data and structured business data.
- **Retrieval:** Graph traversal with temporal awareness; achieves 94.8% on Deep Memory Retrieval benchmark (vs. MemGPT's 93.4%) and up to 18.5% accuracy improvement on LongMemEval with 90% latency reduction.
- **Limitations:** Requires Neo4j infrastructure; temporal graph maintenance adds complexity; entity resolution across conversations is nontrivial.
- **Venue:** arxiv Jan 2025
- **Link:** https://arxiv.org/abs/2501.13956
- **Project:** https://www.getzep.com/

### ChatDB (2023)
- **Core idea:** Uses SQL databases as symbolic memory. The LLM generates SQL INSERT/SELECT/UPDATE/DELETE statements to manage structured information, enabling precise multi-hop reasoning over stored facts.
- **Limitations:** Requires schema design; SQL generation can be error-prone; less natural for fuzzy/personal memories.
- **Venue:** arxiv June 2023
- **Link:** https://arxiv.org/abs/2306.03901

---

## 4. Agentic & Self-Organizing Memory

### A-MEM (2025)
- **Core idea:** Zettelkasten-inspired agentic memory. Each memory is stored as a "note" with structured attributes (contextual description, keywords, tags). The agent autonomously decides how to *create*, *link*, *update*, and *evolve* notes. Links between notes are generated via embedding similarity + LLM reasoning.
- **Key innovation:** The agent can revise *existing* memories when new contradictory or complementary information arrives, not just append.
- **Limitations:** LLM calls for every memory operation add cost; linking quality depends on LLM judgment.
- **Venue:** NeurIPS 2025
- **Link:** https://arxiv.org/abs/2502.12110

### Mem0 (2025)
- **Core idea:** A production-oriented memory layer that extracts, consolidates, and retrieves salient facts from conversations. Offers both a flat fact-store variant and a *graph-based* variant that captures relational structure.
- **Architecture:** Memory extraction (LLM identifies key facts from conversation), deduplication/merging with existing memories, vector + graph storage, retrieval by semantic similarity.
- **Performance:** 26% improvement over OpenAI's memory in LLM-as-a-Judge evaluations; 91% lower p95 latency; 90%+ token cost savings.
- **Limitations:** Extraction quality depends on the base LLM; graph variant adds infrastructure complexity.
- **Venue:** arxiv April 2025
- **Link:** https://arxiv.org/abs/2504.19413
- **Project:** https://mem0.ai/

---

## 5. Memory Operating Systems

### MemoryOS (BAI-LAB, EMNLP 2025 Oral)
- **Core idea:** Three-tier storage hierarchy for personalized agents — *short-term memory* (current conversation buffer), *mid-term memory* (recent session summaries, FIFO eviction), *long-term personal memory* (consolidated user facts, segmented page organization).
- **Operations:** Dynamic migration between tiers; short-to-mid follows dialogue-chain FIFO; mid-to-long uses segmented page strategy for consolidation.
- **Performance:** 48.36% F1 improvement and 46.18% BLEU-1 improvement over baselines on LoCoMo benchmark with GPT-4o-mini.
- **Venue:** EMNLP 2025 (Oral)
- **Link:** https://aclanthology.org/2025.emnlp-main.1318/
- **Code:** https://github.com/BAI-LAB/MemoryOS

### MemOS (MemTensor, 2025)
- **Core idea:** Treats memory as a first-class system resource, unifying three memory types: *plaintext* (facts, summaries), *activation-based* (cached hidden states), and *parameter-level* (LoRA adapters encoding learned knowledge). Provides scheduling and lifecycle management across all three.
- **Limitations:** Complex infrastructure; parameter-level memory requires fine-tuning support.
- **Venue:** arxiv May 2025
- **Link:** https://arxiv.org/abs/2507.03724

---

## 6. Context Compression & Gist-Based Approaches

### ReadAgent (2024)
- **Core idea:** Human-inspired reading agent that segments long documents into episodes, compresses each into a *gist memory* (short summary preserving essential meaning), and looks up original passages on demand.
- **Effective context extension:** 3.5-20x the native context window.
- **Limitations:** Gist quality depends on the LLM; lossy compression; designed for document reading, not conversational memory.
- **Venue:** ICLR 2025
- **Link:** https://arxiv.org/abs/2402.09727

### SCM — Self-Controlled Memory (2023)
- **Core idea:** Plug-and-play framework with three components: LLM agent, *memory stream* (all past interactions), and *memory controller* (decides when/how to store, retrieve, and forget). The LLM itself controls memory operations.
- **Limitations:** Controller decisions are only as good as the LLM's judgment; no external structure to enforce consistency.
- **Venue:** arxiv April 2023
- **Link:** https://arxiv.org/abs/2304.13343

---

## 7. Parametric / Continual Learning Approaches

### Doc-to-LoRA / Text-to-LoRA (2024-2025)
- **Core idea:** Instead of storing knowledge externally and retrieving it, *bake* new knowledge directly into LoRA adapters. A document is converted into a LoRA adapter that the model loads, internalizing the content without full retraining.
- **Limitations:** Each document/knowledge update requires generating an adapter; adapter interference when composing multiple LoRAs; doesn't handle contradiction resolution well.
- **Venue:** Sakana AI, 2025
- **Link:** https://pub.sakana.ai/doc-to-lora/

### FOREVER: Forgetting Curve-Inspired Memory Replay (2025)
- **Core idea:** Applies Ebbinghaus-style spaced replay schedules to continual learning — replay training examples more frequently soon after learning, with increasing intervals over time, to prevent catastrophic forgetting.
- **Venue:** arxiv January 2025
- **Link:** https://arxiv.org/html/2601.03938v1

---

## 8. Benchmarks & Evaluation

| Benchmark | What it tests | Scale | Key finding |
|-----------|--------------|-------|-------------|
| **LoCoMo** (2024) | Very long-term conversational memory (up to 35 sessions) | Multi-session dialogues | RAG and LLMs struggle with long-range temporal/causal reasoning |
| **LongMemEval** (ICLR 2025) | 5 memory abilities: extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention | 500 questions, up to 1.5M tokens | Commercial systems (GPT-4o) achieve only 30-70% accuracy; 30-60% performance drop as history lengthens |
| **Deep Memory Retrieval (DMR)** | Fact retrieval from long conversation histories | Multi-session | Used to benchmark Zep (94.8%) vs MemGPT (93.4%) |

---

## 9. Key Surveys

- **"A Survey on the Memory Mechanism of Large Language Model-based Agents"** — ACM TOIS 2024. Systematic review of memory module design and evaluation.
  - https://arxiv.org/abs/2404.13501
- **"From Human Memory to AI Memory"** — arxiv 2025. Maps human memory concepts (episodic, semantic, procedural) to AI agent memory.
  - https://arxiv.org/html/2504.15965v2
- **"Memory in the Age of AI Agents"** — arxiv Dec 2025. Proposes taxonomy of factual/experiential/working memory; covers formation, evolution, and retrieval lifecycle.
  - https://arxiv.org/abs/2512.13564
  - Paper list: https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- **"Lifelong and Continual Learning Dialogue Systems"** — Springer book, 2024. Covers systems that learn new knowledge through conversation.
  - https://link.springer.com/book/10.1007/978-3-031-48189-5

---

## Summary of Major Architectural Patterns

| # | Pattern | Example | Core Idea | Trade-off |
|---|---------|---------|-----------|-----------|
| 1 | Memory Stream + Retrieval Scoring | Generative Agents | Store everything, retrieve by recency/relevance/importance | Simple but doesn't scale |
| 2 | Tiered Memory with Paging | MemGPT/Letta, MemoryOS | OS-inspired hierarchy, agent pages info in/out of context | Good for bounded context but lossy via summarization |
| 3 | Structured Knowledge Graphs | HippoRAG, Zep/Graphiti, ChatDB | Extract structured facts/relations into a graph or database | Enables multi-hop reasoning and temporal tracking but requires good extraction |
| 4 | Agentic Self-Organizing Memory | A-MEM, Mem0 | LLM decides how to organize, link, merge, update memories | Flexible but costly and dependent on LLM quality |
| 5 | Forgetting-Curve Models | MemoryBank, ACT-R-inspired | Explicitly model memory decay and reinforcement | More human-like but parameters need tuning |
| 6 | Compression / Gist Memory | ReadAgent, SCM | Compress old information into summaries to fit more history | Simple but irreversibly lossy |
| 7 | Parametric Memory | Doc-to-LoRA, FOREVER | Encode knowledge into model weights via adapters | Avoids retrieval latency but harder to update/inspect |

The field is converging toward **hybrid systems** that combine multiple patterns — tiered storage with graph-based relations, forgetting curves for lifecycle management, and agentic control for organization decisions.
