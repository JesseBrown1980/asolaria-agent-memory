# asolaria-agent-memory

Clean-room re-engineering of the researched agent-memory + bidirectional-brain methods into **usable, std-only Rust** for Asolaria's newer systems (HBP hot-path, `json=0`, `sha16` content-addressing). Built by studying the *mechanism* in the papers and writing a better/simpler version from scratch — **no copied source**.

**Status: `cargo build` clean · `cargo test` → 38 passed, 0 failed** (cargo 1.95.0, zero external crates).

## Modules (each re-engineers a researched method)

| module | re-engineers | from | tests |
|---|---|---|---|
| `typed_bitemporal_store` | typed (episodic/semantic/procedural) memory + bi-temporal edges; conflict **invalidates, never deletes**; total-audit `as_of(world, system)` | CoALA `2309.02427` + Zep `2501.13956` | 7 |
| `hopfield_recall` | modern-Hopfield associative recall = attention: store forward, recall a full pattern from a **partial cue in one step** (`Xᵀ·softmax(β·X·ξ)`), LSE energy, masked recall | `2008.02217` | 5 |
| `predictive_coding` | Rao-Ballard local learning: top-down predictions, bottom-up errors, **local weight updates (no global backprop)** — the "learns as it runs" primitive | Rao-Ballard 1999 | ✓ |
| `recall_first_loop` | MemoryVLA-style retrieval **inside** the loop, bounded-latency: retrieve→condition→decide→consolidate | `2508.19236` | ✓ |
| `memory_os_paging` | MemGPT LLM-as-OS: bounded working set + **evict-to-cube on memory pressure**, recall-back on demand | `2310.08560` | ✓ |
| `dual_system_router` | dual-system control surface: low-rate planner emits one **latent**, high-rate workers consume it (maps to the omnidispatcher planner + fast named-agents) | GR00T `2503.14734` / π0 `2410.24164` | ✓ |

## Honest boundaries
- Each module is **compile-verified** under the default toolchain (cargo 1.95.0); this is scoped evidence — not the owning-repo CI on a pinned toolchain.
- These are **faithful re-engineerings of the mechanisms**, simplified for clarity (e.g. the bitemporal store is a functional-key edge log, not a full knowledge graph; `sha16` is a 64-bit identity key, **not** compression — full payload retained; Shannon still caps everything).
- Out of scope (documented in each module header): LLM-driven extraction from raw text, graph traversal/community detection, on-device gradient-quality proof, real-time hz-bounded actuator control.
- The line held: this converges on the **shape** of the researched methods; it does **not** claim to beat their world models or Shannon.
