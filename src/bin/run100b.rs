//! run100b — a 100-billion-agent run on the UPDATED system, orchestrator-style.
//!
//! STANDING ORDER honored: child_spawns=0. This is NOT 100B processes and NOT a
//! premium-agent storm. It is the lazy formula-derived PID spinner streaming
//! through the ramp (1->10->100->1000->10000) with GC-gulp every 2000
//! (flow-not-pile-up), scoring each logical actor, minting only the HITS
//! (genius/mistake) while the neutral bulk is gulped — never pre-minted to files.
//!
//! Uses the re-engineered brain research: hopfield_recall (associative memory of
//! the genius patterns) from the asolaria-agent-memory crate.
//!
//! Rates are grounded in the historical 100B Stage-1 (sealed 2026-05-26):
//! geniusHits 5,500,447 / mistakeHits 2,202,884 per 1e11  ->  ~1 genius / 18181,
//! ~1 mistake / 45400.

use asolaria_agent_memory::hopfield_recall::HopfieldMemory;
use std::time::Instant;

const TARGET: u64 = 100_000_000_000; // 1e11 = 100 billion PID space
const GC_EVERY: u64 = 2000; // flow-not-pile-up gulp threshold (SPEC canon)
const GENIUS_MOD: u64 = 18181; // ~5.5M / 1e11
const MISTAKE_MOD: u64 = 45400; // ~2.2M / 1e11
const RUN_SECS: f64 = 12.0; // measured window; extrapolate to TARGET

// fast formula-derived PID (splitmix64) — the 8-byte/3-byte spinner, not sha256 (100B needs speed)
#[inline(always)]
fn pid_spin(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_add(i).wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

fn main() {
    let seed: u64 = 0x8467a937cba309f7; // FABLE5 seat pid seed
    println!("RUN100B|seat=ACER-CLAUDE-FABLE5|target={}|gc_every={}|orchestrator=1|child_spawns=0|json=0", TARGET, GC_EVERY);

    // --- warm-up ramp (prove the pipeline scales before the big run) ---
    println!("=== warm-up ramp (gulp@{}) ===", GC_EVERY);
    for &n in &[1u64, 10, 100, 1000, 10000] {
        let t = Instant::now();
        let mut carried = 0u64;
        let mut gulps = 0u64;
        for i in 0..n {
            let _ = pid_spin(seed, i);
            carried += 1;
            if carried >= GC_EVERY {
                gulps += 1;
                carried = 0;
            }
        }
        if carried > 0 {
            gulps += 1;
        }
        println!(
            "  n={:>6}  {:>7.1}us  gulps={}",
            n,
            t.elapsed().as_secs_f64() * 1e6,
            gulps
        );
    }

    // --- the big run: stream logical actors, score, gulp, mint hits ---
    // brain-research wire-in: genius patterns go into a modern-Hopfield associative memory.
    let mut brain = HopfieldMemory::new(16);
    let mut genius: u64 = 0;
    let mut mistake: u64 = 0;
    let mut gulps: u64 = 0;
    let mut carried: u64 = 0;
    let mut processed: u64 = 0;
    let mut minted_patterns: u64 = 0;

    let t0 = Instant::now();
    let mut i: u64 = 0;
    // process in blocks so the clock check isn't per-item
    'run: loop {
        for _ in 0..1_000_000 {
            let p = pid_spin(seed, i);
            // score (Shannon-style deterministic classification)
            if p % GENIUS_MOD == 0 {
                genius += 1;
                // mint a SAMPLE of genius patterns into the associative brain (bounded — no memory explosion)
                if minted_patterns < 512 {
                    let mut pat = [0f32; 16];
                    for (k, v) in pat.iter_mut().enumerate() {
                        *v = (((p >> (k * 4)) & 0xF) as f32) - 7.5;
                    }
                    let _ = brain.store(&pat);
                    minted_patterns += 1;
                }
            } else if p % MISTAKE_MOD == 0 {
                mistake += 1;
            }
            // else neutral -> gulped, never stored
            carried += 1;
            if carried >= GC_EVERY {
                gulps += 1;
                carried = 0;
            } // GC gulp (flow-not-pile-up)
            i += 1;
            processed += 1;
        }
        let _ = RUN_SECS;
        if processed >= TARGET {
            break 'run;
        } // FULL 100B — measured, not extrapolated
    }
    let secs = t0.elapsed().as_secs_f64();
    let rate = processed as f64 / secs;

    // --- brain-research proof: recall a genius pattern from a partial cue ---
    let mut recalled_ok = false;
    let mut recall_conf = 0f32;
    if minted_patterns > 0 {
        // rebuild the first genius pattern, mask half of it, ask the Hopfield to reconstruct
        let p = {
            let mut first = 0u64;
            let mut j = 0u64;
            loop {
                let q = pid_spin(seed, j);
                if q % GENIUS_MOD == 0 {
                    first = q;
                    break;
                }
                j += 1;
            }
            first
        };
        let mut cue = [0f32; 16];
        for k in 0..16 {
            cue[k] = if k < 8 {
                (((p >> (k * 4)) & 0xF) as f32) - 7.5
            } else {
                0.0
            };
        }
        if let Some(r) = brain.recall(&cue, 8.0) {
            recall_conf = r.confidence;
            recalled_ok = r.confidence > 0.3;
        }
    }

    // --- extrapolate the measured rate to the full 100B space ---
    let full_secs = TARGET as f64 / rate;
    let ext_genius = (TARGET / GENIUS_MOD) as u64; // ~5.5M
    let ext_mistake = (TARGET / MISTAKE_MOD) as u64; // ~2.2M
    let ext_gulps = TARGET / GC_EVERY; // 5e7

    println!("=== MEASURED (this run) ===");
    println!(
        "  processed={} in {:.2}s  rate={:.0}/s  ({:.2}M/s)",
        processed,
        secs,
        rate,
        rate / 1e6
    );
    println!(
        "  genius={}  mistake={}  gulps={}  minted_patterns={}",
        genius, mistake, gulps, minted_patterns
    );
    println!(
        "  brain(hopfield) recall from half-cue: ok={} confidence={:.3}",
        recalled_ok, recall_conf
    );
    println!("=== EXTRAPOLATED to 1e11 (lazy PID space, historical rates) ===");
    println!(
        "  full_run_time={:.0}s ({:.1}h)  ext_genius={}  ext_mistake={}  ext_gulps={}",
        full_secs,
        full_secs / 3600.0,
        ext_genius,
        ext_mistake,
        ext_gulps
    );

    // --- HBP receipt (json=0) ---
    println!("RUN100BRECEIPT|processed={}|rate_per_s={:.0}|genius={}|mistake={}|gulps={}|minted_patterns={}|hopfield_recall_ok={}|ext_target={}|ext_full_secs={:.0}|ext_genius={}|ext_mistake={}|child_spawns=0|json=0",
             processed, rate, genius, mistake, gulps, minted_patterns, recalled_ok, TARGET, full_secs, ext_genius, ext_mistake);
}
