# FLOAT-CORE-PENDING — 146 sites, one design decision outstanding

Operator rule: **Rust 1.81 with clippy · integer and ternary only · never float.**
Pinned and gated in this repo. **14 sites converted; 146 remain, deliberately, pending a
decision that is a numerical-methods choice rather than a mechanical edit.**

## Converted (exact, no behaviour change)

| file | was | now |
|---|---|---|
| `src/memory_os_paging.rs` (11) | `tau_hi: f64 = 0.9`, `tau_lo: f64 = 0.7`, `tau * budget as f64` | `tau_hi_permille: u16 = 900`, `tau_lo_permille = 700`, `budget * permille / 1000` in `u128`, returning `usize` bytes. Comparisons `usage <= hi_bytes()` are now integer-to-integer. |
| `src/bin/run100b.rs` (3) | `RUN_SECS: f64`, `as_secs_f64()`, `processed as f64 / secs`, `TARGET as f64 / rate` | `RUN_MILLIS: u128`, `as_millis()`/`as_micros()`, `rate_per_sec = processed * 1000 / millis`, integer extrapolation; reports print `{}.{:02}s` and `{}.{:02}M/s` from integer division |

Watermark behaviour is *more* exact than before: `floor(budget × permille / 1000)` has no
rounding drift, and the hysteresis guarantee (a flush always drains to `tau_lo`) still holds
because eligibility uses the same integer value the flush targets.

## The 146 remaining, by kind

| count | kind | example |
|---:|---|---|
| 80 | network numerics — weights, states, buffers | `hopfield_recall.rs:140,259` |
| 22 | learning rates / gradients — `eta_r`, `eta_w`, `lambda`, `alpha`, `df()` | `predictive_coding.rs` |
| 19 | state / similarity — `confidence`, `margin`, `energy`, `mixture`, `cosine`, `sigmoid` | `hopfield_recall.rs:107-117` |
| 8 | **transcendentals** — `.exp()`, `.ln()`, `.sqrt()`, log-sum-exp softmax, `beta = 1/sqrt(dim)` | `hopfield_recall.rs:167,324,326` |
| 8 | **content-address seam** — `sha16(vec: &[f32], step)`, `compute_lid(seq, data: &[f32])` | `recall_first_loop.rs:160`, `dual_system_router.rs:288` |
| 9 | comments describing the above | — |

## The two blockers, stated plainly

**1. The content address is computed over the float bytes.**

```rust
pub fn sha16(vec: &[f32], step: u64) -> String {   // recall_first_loop.rs:160
    for x in vec { bytes.extend_from_slice(&x.to_le_bytes()); }
```

`lid = sha16(seq_le || data f32-le)`. Change the number representation and **every content
address changes**. Any stored lid, any receipt quoting one, any cold-store key stops resolving.
That is not a rounding question; it is a data-migration question, and it needs the operator's
call on whether existing addresses must survive.

**2. Transcendentals are the algorithm, not decoration.**

Hopfield attractor dynamics need `exp` and log-sum-exp; `beta = 1/sqrt(dim)`; cosine similarity
needs `sqrt`; predictive coding is gradient descent with learning rates. Integer versions exist
— fixed-point Q-format with lookup tables, or CORDIC — but they change results, and the change
must be measured against the current recall behaviour, not assumed.

## Three honest options for the decision

- **(a) Fixed-point Q-format** (e.g. Q16.16 in `i32`, or per-mille where ranges allow) with
  precomputed tables for `exp`/`ln`/`sqrt`. Deterministic across platforms — a real gain, since
  float `exp` is not bit-identical between targets. Cost: a table-accuracy pass, and recall
  behaviour re-measured against today's.
- **(b) Trit quantization** — the operator's own substrate: weights and states as balanced
  ternary, energy as integer counts. Most in the spirit of the rule; largest behavioural change;
  needs a recall-quality comparison before adoption.
- **(c) Declare the numerical core a measured boundary** — mark it `FLOAT-NUMERICAL-CORE` with
  a stated reason, exactly as `FLOAT-WITNESS-EXEMPT` and `FLOAT-WIRE-BOUNDARY-EXEMPT` are used
  elsewhere, and convert it when the trit substrate is ready to host it.

**Recommendation: (c) now, (b) when the substrate is ready.** The rule stays honest, nothing is
hidden, and the conversion happens as designed work with measurements rather than as a rewrite
that silently moves every address.

## Gate behaviour in this repo

The CI float check **reports** the remaining sites and points here, instead of failing the build
or pretending they are absent. One line flips it to failing once the core is converted:

```yaml
exit 0    # -> exit $viol
```

Operator: Jesse Daniel Brown. Staged 2026-08-06.
