# Phileas — System Design

## The Landscape

The user already has two knowledge systems:

```
~/life                              ~/wiki
├── daily/        (raw events)      ├── *.md  (technical notes)
├── threads/      (active arcs)     └── CLAUDE.md (conventions)
├── themes/       (long reflections)
├── people/       (relationship profiles)
├── context/      (factual reference)
└── CLAUDE.md     (agreements)
```

These are **human-maintained, file-based** systems. They work well for what they do.
Phileas doesn't replace them — it sits alongside and adds **intelligent retrieval +
conversation memory** that these systems can't do on their own.

## What Phileas Adds

```
┌─────────────────────────────────────────────────────────┐
│                      Claude Code                         │
│                     (the brain)                          │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐  │
│  │  ~/life   │  │  ~/wiki  │  │   Phileas MCP Server  │  │
│  │  (files)  │  │  (files) │  │   (structured memory) │  │
│  └──────────┘  └──────────┘  └───────────────────────┘  │
│       │              │                    │               │
│       │   Claude reads/writes     Claude calls tools     │
│       │   files directly          memorize / recall      │
│       │                                   │               │
└───────┼───────────────────────────────────┼───────────────┘
        │                                   │
        ▼                                   ▼
   Human-readable                    Machine-optimized
   narrative memory                  structured memory
   (you read these)                  (Claude queries these)
```

## The Three Systems — What Goes Where

```
┌─────────────┬──────────────────────┬───────────────────────┐
│   ~/life    │      ~/wiki          │      Phileas          │
├─────────────┼──────────────────────┼───────────────────────┤
│ Life events │ Technical knowledge  │ Conversation memories │
│ Reflections │ Book summaries       │ Extracted facts       │
│ People      │ How-to guides        │ Pattern recognition   │
│ Threads     │ Concepts learned     │ Cross-session context │
│             │                      │                       │
│ Written by: │ Written by:          │ Written by:           │
│ You + Claude│ You + Claude         │ Claude (via MCP)      │
│             │                      │                       │
│ Format:     │ Format:              │ Format:               │
│ Markdown    │ Markdown + YAML      │ SQLite (structured)   │
│ Narrative   │ Zettelkasten         │ Typed memory items    │
└─────────────┴──────────────────────┴───────────────────────┘
```

## Data Flow

```
                    A conversation happens
                            │
                            ▼
                ┌───────────────────────┐
                │  Claude Code session  │
                │                       │
                │  "I got a new job at   │
                │   a robotics startup" │
                └───────────┬───────────┘
                            │
              Claude decides what to do:
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
     Update ~/life    Update ~/wiki    Call Phileas MCP
                                            │
     ┌────────────┐  ┌────────────┐   ┌─────┴──────────┐
     │ daily entry │  │ (nothing   │   │ memorize(      │
     │ people/     │  │  relevant) │   │   "New job at   │
     │ threads/    │  │            │   │    robotics     │
     │ career      │  │            │   │    startup",   │
     └────────────┘  └────────────┘   │   type="event", │
                                      │   cat="career") │
                                      └────────────────┘
```

## Retrieval Flow

```
              User asks something in a new session
                            │
                            ▼
                ┌───────────────────────┐
                │  Claude Code session  │
                │                       │
                │  "What's been going   │
                │   on with my career?" │
                └───────────┬───────────┘
                            │
              Claude gathers context:
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
      Read ~/life      Read ~/wiki    Call Phileas MCP
      threads/career   (if relevant)        │
      themes/career                   ┌─────┴──────────┐
      context/work/                   │ recall(         │
                                      │   "career",    │
                                      │   type="event")│
                                      └───────┬────────┘
                                              │
                                              ▼
                                      Returns memories
                                      from past sessions
                                      that ~/life may
                                      not have captured
```

## Phileas Internal Architecture

```
┌─────────────────────────────────────────┐
│           Phileas MCP Server            │
│                                         │
│  Tools (what Claude Code calls):        │
│  ┌────────────┐  ┌──────────────────┐   │
│  │ memorize   │  │ store_conversation│   │
│  │ recall     │  │ profile          │   │
│  │ categories │  │                  │   │
│  └─────┬──────┘  └────────┬─────────┘   │
│        │                  │             │
│        ▼                  ▼             │
│  ┌──────────────────────────────────┐   │
│  │         Memory Engine            │   │
│  │                                  │   │
│  │  store_memory()  recall()        │   │
│  │  store_resource() get_profile()  │   │
│  └──────────────┬───────────────────┘   │
│                 │                       │
│        ┌────────┴────────┐              │
│        ▼                 ▼              │
│  ┌──────────┐    ┌────────────┐         │
│  │ Keyword  │    │ Embedding  │         │
│  │ Search   │    │ Search     │         │
│  │ (default)│    │ (optional) │         │
│  └──────────┘    └────────────┘         │
│        │                 │              │
│        └────────┬────────┘              │
│                 ▼                       │
│  ┌──────────────────────────────────┐   │
│  │        SQLite Database           │   │
│  │                                  │   │
│  │  resources     (L1: immutable)   │   │
│  │  memory_items  (L2: editable)    │   │
│  │  categories    (organization)    │   │
│  │  category_items (links)          │   │
│  └──────────────────────────────────┘   │
│                                         │
│  Storage: ~/.phileas/memory.db          │
└─────────────────────────────────────────┘
```

## Memory Types

| Type | What it captures | Example |
|------|-----------------|---------|
| `profile` | Who the user is | "Software developer interested in AI" |
| `event` | Things that happened | "Got new job at robotics startup" |
| `knowledge` | Things user knows/cares about | "Studying memory architectures for AI" |
| `behavior` | Patterns and preferences | "Prefers local-first architecture" |
| `reflection` | Higher-level inferences | "Going through a career transition" |

## What's NOT in Phileas (stays in ~/life and ~/wiki)

- Daily journal entries → ~/life/daily/
- Relationship profiles → ~/life/people/
- Life threads and themes → ~/life/threads/, ~/life/themes/
- Technical knowledge → ~/wiki/
- Calendar and tasks → Google Calendar + Taskwarrior

Phileas captures what falls through the cracks: the small facts, preferences,
and patterns from conversations that aren't significant enough for a daily entry
but matter for continuity across sessions.

## Future: Indexing ~/life and ~/wiki

Eventually, Phileas could also **index** the existing files to make them
searchable via the same MCP interface. This would let Claude query one
unified memory layer instead of reading files + calling MCP separately.

But that's a later optimization. Start with conversation memory first.
