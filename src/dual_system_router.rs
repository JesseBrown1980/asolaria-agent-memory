//! # dual_system_router — the seam between a slow planner and a fast worker
//!
//! ## What this is (clean-room, WHITE-ROOM-RULES)
//! GR00T N1 (arXiv:2503.14734), Helix (Figure AI, 2025), and pi0 (arXiv:2410.24164)
//! independently converged on ONE structure for dual-rate robot control. Strip each to
//! its smallest common core and the same three-rule seam remains:
//!
//! 1. **ZERO-ORDER HOLD** — the fast lane (worker, ~100-200 Hz) NEVER blocks on the slow
//!    lane (planner, ~7-10 Hz). It reads whatever latent `z` was last published and holds
//!    it constant until replaced, so staleness is bounded by one planner period.
//!    [MEASURED: Helix explicit "latest-wins" async latent; GR00T implicit.]
//! 2. **ACTION CHUNKING (receding horizon)** — each worker tick emits `H` future actions;
//!    only a prefix is consumed before the next chunk supersedes it. The chunk must cover
//!    at least one planner period so a slow-ish worker still drives a smooth loop:
//!    `H * ctrl_period >= 1/planner_hz`. [MEASURED: H=16 GR00T, H=50 pi0.]
//! 3. **INFORMATION BOTTLENECK BY CONSTRUCTION** — `I(action; goal)` flows ONLY through
//!    the latent `z`. Seam bandwidth = `worker_hz * dim * 32` bits/s. No side channel, no
//!    shared scratch, no RPC schema: the single-slot mailbox IS the API.
//!    [MEASURED across all three papers.]
//!
//! Everything neural in the papers (VLM planner, flow-matching decoder, cross-attention)
//! is *implementation behind the seam*, not the mechanism. This module reifies only the
//! mechanism, std-only, deterministic.
//!
//! ## What Asolaria ADDS (all tagged DESIGN)
//! - **LOGICAL TIME, not wall-clock** — both lanes run off `u64` tick counters, so the
//!   whole router is deterministic, `E=0` testable, and replayable from receipts. Wall
//!   clock enters only at an outer harness, never in this contract.
//! - **CONTENT-ADDRESSED SEAM** — `lid = sha16(seq_le || z f32-le)`, and every chunk
//!   carries `cid = sha16(seq_le || lid || action bytes)`, so the causal chain
//!   `goal -> z -> chunk` is verifiable after the fact by replaying HBP rows. `sha16` is a
//!   pure-Rust SHA-256 truncated to 8 bytes (KAT-verified below), clean-room reimplemented.
//! - **RECOVER-OR-HOLD GATE** — the honest boundary the papers lack: if the seam is absent
//!   or `age_ticks > R + slack`, the worker REFUSES to act (emits a Hold receipt) instead
//!   of silently driving on a dead latent.
//! - **HBP HOT-PATH RECEIPTS** — every tick emits a `|`-delimited row ending `|json=0`.
//!   Rows are the primary output; JSON never appears on this path.
//!
//! ## Honest boundary (SHANNON — do NOT re-inflate)
//! `z` is a *lossy shadow* of planner state: `H(planner_state | z) > 0`. The worker cannot
//! reconstruct information not in `z`, and a goal change cannot reach actions faster than
//! one worker period. Seam bandwidth `worker_hz * dim * 32` bit/s is a hard ceiling. This
//! module reproduces only the STRUCTURAL contract — NOT the papers' trained-network results
//! (no "200 Hz humanoid control", no end-to-end training; the differentiable cross-attention
//! seam was deliberately DROPPED). The reference `Planner`/`Worker` impls are original
//! deterministic stand-ins that prove the seam is testable with zero neural machinery.

use std::sync::Arc;

// ===========================================================================
// sha16 — pure-Rust SHA-256 truncated to 8 bytes (clean-room, KAT-verified)
// ===========================================================================

/// Pure-Rust SHA-256. Reimplemented from FIPS 180-4; no external crates, no copied source.
mod sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    /// Full 32-byte SHA-256 digest of `msg`.
    pub fn digest(msg: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];

        // Pre-process: append 0x80, pad with zeros to 56 mod 64, then 64-bit big-endian length.
        let bit_len: u64 = (msg.len() as u64).wrapping_mul(8);
        let mut data: Vec<u8> = Vec::with_capacity(msg.len() + 72);
        data.extend_from_slice(msg);
        data.push(0x80);
        while data.len() % 64 != 56 {
            data.push(0x00);
        }
        data.extend_from_slice(&bit_len.to_be_bytes());

        for block in data.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, wi) in w.iter_mut().enumerate().take(16) {
                let j = i * 4;
                *wi = u32::from_be_bytes([block[j], block[j + 1], block[j + 2], block[j + 3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }

            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
                (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }

            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }

        let mut out = [0u8; 32];
        for (i, word) in h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// Truncated SHA-256: the first 8 bytes of the digest of the concatenation of `parts`.
/// This is the seam's content-address primitive (DESIGN). Same pattern as
/// asolaria-hbi-hbp — clean-room reimplemented, never copied.
pub fn sha16(parts: &[&[u8]]) -> [u8; 8] {
    let mut buf: Vec<u8> = Vec::new();
    for p in parts {
        buf.extend_from_slice(p);
    }
    let full = sha256::digest(&buf);
    let mut out = [0u8; 8];
    out.copy_from_slice(&full[..8]);
    out
}

/// Lowercase hex of an 8-byte content-address (16 chars). Used in HBP rows.
pub fn hex16(id: &[u8; 8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(16);
    for &byte in id.iter() {
        s.push(LUT[(byte >> 4) as usize] as char);
        s.push(LUT[(byte & 0x0f) as usize] as char);
    }
    s
}

// ===========================================================================
// Errors + rate configuration (fail-fast constructor invariants)
// ===========================================================================

/// Construction-time rejections. The coverage inequality and rate sanity are enforced here
/// so a misconfigured seam is impossible to build, not a runtime surprise (DESIGN).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouterError {
    /// A rate field was zero, or the worker lane is not faster than the planner lane.
    BadRate(&'static str),
    /// Chunk horizon too small to cover one planner period: `horizon < ceil(worker/planner)`.
    /// This is MEASURED-motivated (GR00T H=16, pi0 H=50 both exceed their ratio) and made a
    /// hard invariant (DESIGN).
    CoverageViolation { horizon: usize, required: u64 },
    /// The latent produced by a `Planner` did not match the router's configured `dim`.
    DimMismatch { got: usize, want: usize },
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::BadRate(m) => write!(f, "bad rate: {m}"),
            RouterError::CoverageViolation { horizon, required } => write!(
                f,
                "chunk coverage violated: horizon {horizon} < required {required}"
            ),
            RouterError::DimMismatch { got, want } => {
                write!(f, "latent dim mismatch: got {got}, want {want}")
            }
        }
    }
}

impl std::error::Error for RouterError {}

/// The two lane rates plus the chunk horizon and staleness slack.
///
/// - `planner_hz` (f1): slow lane, MEASURED ~7-10 Hz in the papers.
/// - `worker_hz`  (f2): fast lane, MEASURED ~100-200 Hz in the papers.
/// - `horizon`    (H): actions emitted per worker tick, MEASURED 16 (GR00T) / 50 (pi0).
/// - `slack_ticks`: DESIGN grace beyond the rate ratio before the Hold gate fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateCfg {
    pub planner_hz: u32,
    pub worker_hz: u32,
    pub horizon: usize,
    pub slack_ticks: u64,
}

impl RateCfg {
    /// `R = ceil(worker_hz / planner_hz)` — how many worker ticks pass per planner period,
    /// i.e. the maximum zero-order-hold age a fresh latent should ever reach.
    pub fn rate_ratio(&self) -> u64 {
        let w = self.worker_hz as u64;
        let p = self.planner_hz.max(1) as u64;
        w.div_ceil(p)
    }

    /// The Hold gate threshold in worker ticks: `R + slack`. `age_ticks > bound => Hold`.
    pub fn staleness_bound(&self) -> u64 {
        self.rate_ratio() + self.slack_ticks
    }

    /// Enforce fail-fast invariants (rate sanity + coverage inequality).
    pub fn validate(&self) -> Result<(), RouterError> {
        if self.planner_hz == 0 {
            return Err(RouterError::BadRate("planner_hz == 0"));
        }
        if self.worker_hz == 0 {
            return Err(RouterError::BadRate("worker_hz == 0"));
        }
        if self.worker_hz < self.planner_hz {
            return Err(RouterError::BadRate(
                "worker_hz < planner_hz (fast lane not faster)",
            ));
        }
        if self.horizon == 0 {
            return Err(RouterError::BadRate("horizon == 0"));
        }
        // Coverage: H * ctrl_period >= 1/planner_hz. With ctrl_period = 1/worker_hz this is
        // H >= worker_hz/planner_hz = R. Enforced as a constructor invariant, not a doc note.
        let required = self.rate_ratio();
        if (self.horizon as u64) < required {
            return Err(RouterError::CoverageViolation {
                horizon: self.horizon,
                required,
            });
        }
        Ok(())
    }
}

// ===========================================================================
// The seam payload: an immutable, content-addressed Latent
// ===========================================================================

/// The single compact value coupling the two lanes — the latent `z`.
///
/// Immutable once constructed: fields are private and there are no setters, so `lid` can
/// never drift from `data`. A new value means a new `seq` and a new `lid` (latest-wins).
#[derive(Debug, Clone, PartialEq)]
pub struct Latent {
    seq: u64,
    planner_hz: u32,
    data: Vec<f32>,
    lid: [u8; 8],
}

impl Latent {
    /// Build a latent and compute its content-address `lid = sha16(seq_le || data f32-le)`.
    pub fn new(seq: u64, planner_hz: u32, data: Vec<f32>) -> Latent {
        let lid = Latent::compute_lid(seq, &data);
        Latent {
            seq,
            planner_hz,
            data,
            lid,
        }
    }

    fn compute_lid(seq: u64, data: &[f32]) -> [u8; 8] {
        let mut bytes: Vec<u8> = Vec::with_capacity(8 + data.len() * 4);
        for &x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        sha16(&[&seq.to_le_bytes(), &bytes])
    }

    /// Logical planner sequence number (monotonic; the seam's version).
    pub fn seq(&self) -> u64 {
        self.seq
    }
    /// Latent dimensionality `d` — one factor of the seam bandwidth `worker_hz * d * 32`.
    pub fn dim(&self) -> usize {
        self.data.len()
    }
    /// The planner rate this latent was produced at.
    pub fn planner_hz(&self) -> u32 {
        self.planner_hz
    }
    /// Read-only view of the latent vector.
    pub fn data(&self) -> &[f32] {
        &self.data
    }
    /// The 8-byte content-address of this latent.
    pub fn lid(&self) -> [u8; 8] {
        self.lid
    }
    /// Re-derive the lid and check it matches — proves immutability held (auditing).
    pub fn verify_lid(&self) -> bool {
        Latent::compute_lid(self.seq, &self.data) == self.lid
    }
}

// ===========================================================================
// The single-slot atomic mailbox (latest-wins, O(1) swap, no torn reads)
// ===========================================================================

/// One entry: the published latent plus the worker tick at which it was published (for age).
#[derive(Debug, Clone)]
struct SlotEntry {
    latent: Arc<Latent>,
    publish_tick: u64,
}

/// The seam itself: a single slot holding at most one immutable latent.
///
/// Publish is an O(1) swap of a whole `Arc<Latent>` — the worker never mutates in place and
/// never takes a lock the planner holds across compute. This single-threaded slot is the
/// contract; a threaded harness wraps it in `RwLock<Option<Arc<Latent>>>` with the same O(1)
/// writer swap (the immutable `Arc` is exactly what makes that swap torn-read-free).
#[derive(Debug, Clone, Default)]
pub struct LatentSlot {
    entry: Option<SlotEntry>,
}

impl LatentSlot {
    /// An empty seam (no latent yet published).
    pub fn new() -> LatentSlot {
        LatentSlot { entry: None }
    }

    /// Latest-wins publish: replace whatever was there. O(1), no in-place mutation.
    pub fn publish(&mut self, latent: Arc<Latent>, publish_tick: u64) {
        self.entry = Some(SlotEntry {
            latent,
            publish_tick,
        });
    }

    /// Read the held latent and its publish tick, if any. Cheap `Arc` clone, never blocks.
    pub fn read(&self) -> Option<(Arc<Latent>, u64)> {
        self.entry
            .as_ref()
            .map(|e| (Arc::clone(&e.latent), e.publish_tick))
    }

    /// Whether a latent has ever been published.
    pub fn is_armed(&self) -> bool {
        self.entry.is_some()
    }
}

// ===========================================================================
// The worker's output: a receding-horizon action chunk, or a Hold
// ===========================================================================

/// One control step's action vector.
pub type Action = Vec<f32>;

/// H future actions produced from a held latent, content-addressed by `cid`.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    worker_tick: u64,
    latent_seq: u64,
    latent_lid: [u8; 8],
    age_ticks: u64,
    actions: Vec<Action>,
    cid: [u8; 8],
}

impl Chunk {
    fn new(
        worker_tick: u64,
        latent_seq: u64,
        latent_lid: [u8; 8],
        age_ticks: u64,
        actions: Vec<Action>,
    ) -> Chunk {
        let cid = Chunk::compute_cid(latent_seq, &latent_lid, &actions);
        Chunk {
            worker_tick,
            latent_seq,
            latent_lid,
            age_ticks,
            actions,
            cid,
        }
    }

    /// `cid = sha16(latent_seq_le || latent_lid || action f32-le bytes)`. Binds the chunk to
    /// the exact latent it was produced from, so `goal -> z -> chunk` is replay-verifiable.
    fn compute_cid(latent_seq: u64, latent_lid: &[u8; 8], actions: &[Action]) -> [u8; 8] {
        let mut abytes: Vec<u8> = Vec::new();
        for a in actions {
            for &x in a {
                abytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        sha16(&[&latent_seq.to_le_bytes(), latent_lid, &abytes])
    }

    /// Worker tick at which this chunk was emitted.
    pub fn worker_tick(&self) -> u64 {
        self.worker_tick
    }
    /// The seq of the latent this chunk was driven from.
    pub fn latent_seq(&self) -> u64 {
        self.latent_seq
    }
    /// The content-address of the latent this chunk was driven from (the causal link).
    pub fn latent_lid(&self) -> [u8; 8] {
        self.latent_lid
    }
    /// Zero-order-hold age in worker ticks (0 == fresh this planner period).
    pub fn age_ticks(&self) -> u64 {
        self.age_ticks
    }
    /// The horizon H = number of actions in this chunk.
    pub fn horizon(&self) -> usize {
        self.actions.len()
    }
    /// The action steps.
    pub fn actions(&self) -> &[Action] {
        &self.actions
    }
    /// This chunk's content-address.
    pub fn cid(&self) -> [u8; 8] {
        self.cid
    }
    /// Re-derive the cid from the stored latent link + actions and check it matches.
    pub fn verify_cid(&self) -> bool {
        Chunk::compute_cid(self.latent_seq, &self.latent_lid, &self.actions) == self.cid
    }
}

/// Why the worker refused to act (the honest boundary the papers lack).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HoldReason {
    /// No latent has ever been published — the seam is not armed.
    NoSeam,
    /// The held latent is older than `R + slack` worker ticks: driving on it would be acting
    /// on a dead seam. `age_ticks > bound`.
    Stale { age_ticks: u64, bound: u64 },
}

/// The result of a single worker tick.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerStep {
    /// A fresh-enough seam: here is the receding-horizon chunk to consume.
    Drive(Chunk),
    /// The recover-or-Hold gate fired: refuse to act, emit a Hold receipt.
    Hold(HoldReason),
}

// ===========================================================================
// The two trait seats (System-2 planner, System-1 worker)
// ===========================================================================

/// System-2 seat (slow). In Asolaria the omnidispatcher sits here; in the papers a VLM.
pub trait Planner {
    /// Produce the latent `z` of length exactly `dim` from observation + goal at logical
    /// sequence `seq`. Must be pure w.r.t. its inputs for `E=0` replay.
    fn plan(&mut self, seq: u64, obs: &[f32], goal: &[f32], dim: usize) -> Vec<f32>;
}

/// System-1 seat (fast). In Asolaria a named fast-agent; in the papers a flow-matching
/// decoder. Reads the held latent (never blocks) and emits `horizon` actions.
pub trait Worker {
    /// Produce exactly `horizon` action vectors from the held latent `z` and current `state`.
    fn act(&mut self, worker_tick: u64, z: &Latent, state: &[f32], horizon: usize) -> Vec<Action>;
}

// ===========================================================================
// Deterministic reference impls (no neural machinery — proves the seam is testable)
// ===========================================================================

fn unit_from_bytes(id: &[u8; 8]) -> f32 {
    let u = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
    // Map u32 -> [-1, 1) deterministically (f64 intermediate for precision).
    ((u as f64 / u32::MAX as f64) * 2.0 - 1.0) as f32
}

/// Bounded squash x/(1+|x|) in (-1, 1) — std-only, no libm transcendentals.
fn squash(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// A deterministic stand-in for the VLM planner: a sha16-seeded linear projection of
/// `obs || goal` into `dim` bounded floats. Original clean-room code (DESIGN), not derived
/// from any GR00T/Helix/pi0 weights — its only job is to prove the seam contract runs with
/// zero learned parameters.
#[derive(Debug, Default, Clone)]
pub struct DeterministicPlanner;

impl Planner for DeterministicPlanner {
    fn plan(&mut self, seq: u64, obs: &[f32], goal: &[f32], dim: usize) -> Vec<f32> {
        let mut inputs: Vec<f32> = Vec::with_capacity(obs.len() + goal.len());
        inputs.extend_from_slice(obs);
        inputs.extend_from_slice(goal);
        let mut z = Vec::with_capacity(dim);
        for j in 0..dim {
            let mut acc = 0.0f32;
            for (k, &v) in inputs.iter().enumerate() {
                // Deterministic pseudo-weight seeded by (seq, j, k).
                let seed = sha16(&[
                    &seq.to_le_bytes(),
                    &(j as u64).to_le_bytes(),
                    &(k as u64).to_le_bytes(),
                ]);
                acc += unit_from_bytes(&seed) * v;
            }
            // A small seq/j-dependent bias so distinct goals map to distinct latents.
            let bias = unit_from_bytes(&sha16(&[&seq.to_le_bytes(), &(j as u64).to_le_bytes()]));
            z.push(squash(acc + 0.1 * bias));
        }
        z
    }
}

/// A deterministic stand-in for the flow-matching worker: each of the `horizon` steps is an
/// affine blend of the latent and the current state, with a mild per-step decay so the
/// receding horizon is visible. Original clean-room code (DESIGN).
#[derive(Debug, Clone)]
pub struct DeterministicWorker {
    /// Latent influence.
    pub alpha: f32,
    /// State influence.
    pub beta: f32,
}

impl Default for DeterministicWorker {
    fn default() -> Self {
        DeterministicWorker {
            alpha: 0.7,
            beta: 0.3,
        }
    }
}

impl Worker for DeterministicWorker {
    fn act(&mut self, _worker_tick: u64, z: &Latent, state: &[f32], horizon: usize) -> Vec<Action> {
        let zdat = z.data();
        let adim = state.len().max(1);
        let mut chunk = Vec::with_capacity(horizon);
        for h in 0..horizon {
            // Later steps in the horizon decay toward state-hold (receding-horizon shape).
            let decay = 1.0 / (1.0 + h as f32);
            let mut action = Vec::with_capacity(adim);
            for i in 0..adim {
                let zi = if zdat.is_empty() {
                    0.0
                } else {
                    zdat[i % zdat.len()]
                };
                let si = state.get(i).copied().unwrap_or(0.0);
                action.push(squash(self.alpha * zi * decay + self.beta * si));
            }
            chunk.push(action);
        }
        chunk
    }
}

// ===========================================================================
// HBP hot-path receipts (rows are primary output; JSON never on this path)
// ===========================================================================

/// Planner publish receipt.
/// `DSR|role=planner|seq=..|lid=..|dim=..|hz=..|json=0`
pub fn hbp_planner_row(seq: u64, lid: &[u8; 8], dim: usize, planner_hz: u32) -> String {
    format!(
        "DSR|role=planner|seq={}|lid={}|dim={}|hz={}|json=0",
        seq,
        hex16(lid),
        dim,
        planner_hz
    )
}

/// Worker drive receipt.
/// `DSR|role=worker|tick=..|seq=..|latent_lid=..|age_ticks=..|h=..|cid=..|json=0`
pub fn hbp_worker_row(chunk: &Chunk) -> String {
    format!(
        "DSR|role=worker|tick={}|seq={}|latent_lid={}|age_ticks={}|h={}|cid={}|json=0",
        chunk.worker_tick(),
        chunk.latent_seq(),
        hex16(&chunk.latent_lid()),
        chunk.age_ticks(),
        chunk.horizon(),
        hex16(&chunk.cid())
    )
}

/// Worker Hold (refusal) receipt — refusals are auditable too, never a silent drop.
/// `DSR|role=worker|tick=..|hold=stale|age_ticks=..|bound=..|json=0`
/// `DSR|role=worker|tick=..|hold=no_seam|json=0`
pub fn hbp_hold_row(worker_tick: u64, reason: &HoldReason) -> String {
    match reason {
        HoldReason::NoSeam => {
            format!("DSR|role=worker|tick={worker_tick}|hold=no_seam|json=0")
        }
        HoldReason::Stale { age_ticks, bound } => {
            format!(
                "DSR|role=worker|tick={worker_tick}|hold=stale|age_ticks={age_ticks}|bound={bound}|json=0"
            )
        }
    }
}

// ===========================================================================
// The router: couples the two lanes through the one slot, in logical time
// ===========================================================================

/// The dual-system router. Owns the single seam slot and the two logical tick counters, and
/// enforces zero-order-hold + the recover-or-Hold gate. Fully deterministic: no wall clock,
/// no threads, replayable from its HBP receipts.
#[derive(Debug)]
pub struct DualRouter {
    cfg: RateCfg,
    dim: usize,
    slot: LatentSlot,
    /// Next planner sequence number to assign (monotonic logical time).
    planner_seq: u64,
    /// Current worker tick (monotonic logical time).
    worker_tick: u64,
    /// Captured HBP rows, in emission order (primary audit output).
    receipts: Vec<String>,
}

impl DualRouter {
    /// Build a router, enforcing the coverage inequality and rate sanity up front.
    pub fn new(cfg: RateCfg, dim: usize) -> Result<DualRouter, RouterError> {
        cfg.validate()?;
        if dim == 0 {
            return Err(RouterError::BadRate("dim == 0"));
        }
        Ok(DualRouter {
            cfg,
            dim,
            slot: LatentSlot::new(),
            planner_seq: 0,
            worker_tick: 0,
            receipts: Vec::new(),
        })
    }

    /// The configured rates.
    pub fn cfg(&self) -> &RateCfg {
        &self.cfg
    }
    /// Latent dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Current worker tick.
    pub fn worker_clock(&self) -> u64 {
        self.worker_tick
    }
    /// Next planner sequence number.
    pub fn planner_clock(&self) -> u64 {
        self.planner_seq
    }
    /// Seam bandwidth ceiling in bits/sec: `worker_hz * dim * 32` (SHANNON hard ceiling).
    pub fn seam_bandwidth_bits_per_s(&self) -> u64 {
        self.cfg.worker_hz as u64 * self.dim as u64 * 32
    }
    /// All captured HBP receipt rows so far.
    pub fn receipts(&self) -> &[String] {
        &self.receipts
    }

    /// SLOW LANE. Run the planner, publish a fresh immutable latent into the seam (latest-wins),
    /// and emit a planner HBP row. Returns the published latent (shared, immutable).
    pub fn planner_publish<P: Planner>(
        &mut self,
        planner: &mut P,
        obs: &[f32],
        goal: &[f32],
    ) -> Result<Arc<Latent>, RouterError> {
        let seq = self.planner_seq;
        let z = planner.plan(seq, obs, goal, self.dim);
        if z.len() != self.dim {
            return Err(RouterError::DimMismatch {
                got: z.len(),
                want: self.dim,
            });
        }
        let latent = Arc::new(Latent::new(seq, self.cfg.planner_hz, z));
        // O(1) swap of a whole immutable value — the fast lane never sees a torn read.
        self.slot.publish(Arc::clone(&latent), self.worker_tick);
        self.receipts.push(hbp_planner_row(
            latent.seq(),
            &latent.lid(),
            latent.dim(),
            latent.planner_hz(),
        ));
        self.planner_seq += 1;
        Ok(latent)
    }

    /// FAST LANE. One worker tick. NEVER blocks on the planner: it reads whatever latent is
    /// held (zero-order hold), applies the recover-or-Hold gate, and either drives a chunk or
    /// refuses. Always advances the worker clock and always emits exactly one HBP row.
    pub fn worker_tick<W: Worker>(&mut self, worker: &mut W, state: &[f32]) -> WorkerStep {
        let now = self.worker_tick;
        let step = match self.slot.read() {
            None => {
                let reason = HoldReason::NoSeam;
                self.receipts.push(hbp_hold_row(now, &reason));
                WorkerStep::Hold(reason)
            }
            Some((latent, publish_tick)) => {
                // Staleness in logical worker ticks since publish (bounded by one planner period).
                let age = now.saturating_sub(publish_tick);
                let bound = self.cfg.staleness_bound();
                if age > bound {
                    let reason = HoldReason::Stale {
                        age_ticks: age,
                        bound,
                    };
                    self.receipts.push(hbp_hold_row(now, &reason));
                    WorkerStep::Hold(reason)
                } else {
                    let actions = worker.act(now, &latent, state, self.cfg.horizon);
                    let chunk = Chunk::new(now, latent.seq(), latent.lid(), age, actions);
                    self.receipts.push(hbp_worker_row(&chunk));
                    WorkerStep::Drive(chunk)
                }
            }
        };
        self.worker_tick += 1;
        step
    }
}

// ===========================================================================
// Tests — exercise the real mechanism, deterministically (E=0, no wall clock)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // sha16 is the seam's content-address primitive; a wrong SHA-256 would silently corrupt
    // every lid/cid. Pin it against the FIPS 180-4 known-answer tests (first 8 bytes).
    #[test]
    fn sha256_known_answer_tests() {
        // SHA-256("") = e3b0c442 98fc1c14 ...
        assert_eq!(hex16(&sha16(&[b""])), "e3b0c44298fc1c14");
        // SHA-256("abc") = ba7816bf 8f01cfea ...
        assert_eq!(hex16(&sha16(&[b"abc"])), "ba7816bf8f01cfea");
        // Concatenation must equal hashing the joined bytes.
        assert_eq!(sha16(&[b"ab", b"c"]), sha16(&[b"abc"]));
    }

    #[test]
    fn constructor_enforces_coverage_and_rate_sanity() {
        // R = ceil(200/10) = 20, so horizon must be >= 20.
        let bad = RateCfg {
            planner_hz: 10,
            worker_hz: 200,
            horizon: 8, // < 20
            slack_ticks: 0,
        };
        match DualRouter::new(bad, 16) {
            Err(RouterError::CoverageViolation { horizon, required }) => {
                assert_eq!(horizon, 8);
                assert_eq!(required, 20);
            }
            other => panic!("expected CoverageViolation, got {other:?}"),
        }

        // Fast lane must actually be faster than the slow lane.
        let inverted = RateCfg {
            planner_hz: 100,
            worker_hz: 10,
            horizon: 4,
            slack_ticks: 0,
        };
        assert!(matches!(
            DualRouter::new(inverted, 8),
            Err(RouterError::BadRate(_))
        ));

        // A paper-shaped config (GR00T-ish: 10 Hz / 120 Hz, H=16 >= ceil(120/10)=12) is valid.
        let good = RateCfg {
            planner_hz: 10,
            worker_hz: 120,
            horizon: 16,
            slack_ticks: 2,
        };
        let r = DualRouter::new(good, 32).expect("valid config");
        assert_eq!(r.cfg().rate_ratio(), 12);
        assert_eq!(r.cfg().staleness_bound(), 14);
        // Seam bandwidth ceiling = 120 * 32 * 32.
        assert_eq!(r.seam_bandwidth_bits_per_s(), 120 * 32 * 32);
    }

    #[test]
    fn zero_order_hold_then_stale_hold_gate() {
        // planner 10 Hz, worker 40 Hz -> R = 4; slack 1 -> bound = 5.
        let cfg = RateCfg {
            planner_hz: 10,
            worker_hz: 40,
            horizon: 4,
            slack_ticks: 1,
        };
        let mut router = DualRouter::new(cfg, 6).unwrap();
        let mut planner = DeterministicPlanner;
        let mut worker = DeterministicWorker::default();

        // Before any publish, the seam is not armed: worker must Hold(NoSeam), not act.
        let s0 = router.worker_tick(&mut worker, &[0.0, 0.0, 0.0]);
        assert_eq!(s0, WorkerStep::Hold(HoldReason::NoSeam));

        // Publish one latent at worker_tick == 1 (the NoSeam tick advanced the clock).
        let published = router
            .planner_publish(&mut planner, &[0.1, -0.2, 0.3], &[1.0, 0.0])
            .unwrap();
        assert_eq!(published.dim(), 6);
        assert!(published.verify_lid());

        // Zero-order hold: several worker ticks with NO new publish all drive the SAME latent
        // (same latent_lid), with age rising by exactly 1 each tick, until the gate fires.
        let mut last_age = None;
        let mut drove = 0;
        let mut stale_hit = false;
        for _ in 0..8 {
            match router.worker_tick(&mut worker, &[0.5, -0.5, 0.25]) {
                WorkerStep::Drive(chunk) => {
                    // ZOH: it is the same published latent, held constant.
                    assert_eq!(chunk.latent_lid(), published.lid());
                    assert_eq!(chunk.latent_seq(), published.seq());
                    assert_eq!(chunk.horizon(), 4);
                    // The causal chain goal -> z -> chunk is re-derivable.
                    assert!(chunk.verify_cid());
                    // Age is monotonically increasing by 1 (logical ticks).
                    if let Some(prev) = last_age {
                        assert_eq!(chunk.age_ticks(), prev + 1);
                    }
                    last_age = Some(chunk.age_ticks());
                    // Never drive past the bound.
                    assert!(chunk.age_ticks() <= router.cfg().staleness_bound());
                    drove += 1;
                }
                WorkerStep::Hold(HoldReason::Stale { age_ticks, bound }) => {
                    // Recover-or-Hold: refuse to act on the dead seam.
                    assert!(age_ticks > bound);
                    assert_eq!(bound, 5);
                    stale_hit = true;
                    break;
                }
                other => panic!("unexpected step: {other:?}"),
            }
        }
        assert!(drove >= 1, "should have driven at least one fresh chunk");
        assert!(
            stale_hit,
            "the Hold gate must eventually fire on a stale seam"
        );

        // Every emitted receipt is a hot-path row ending |json=0, with NO JSON braces.
        assert!(!router.receipts().is_empty());
        for row in router.receipts() {
            assert!(row.ends_with("|json=0"), "row not hot-path: {row}");
            assert!(row.starts_with("DSR|role="), "row not a DSR row: {row}");
            assert!(
                !row.contains('{') && !row.contains('}'),
                "JSON leaked: {row}"
            );
        }
        // Exactly one Hold(stale) receipt was recorded (refusals are auditable, not silent).
        let stale_rows = router
            .receipts()
            .iter()
            .filter(|r| r.contains("hold=stale"))
            .count();
        assert_eq!(stale_rows, 1);
    }

    #[test]
    fn latest_wins_republish_resets_age_and_swaps_lid() {
        let cfg = RateCfg {
            planner_hz: 10,
            worker_hz: 30,
            horizon: 3,
            slack_ticks: 0,
        };
        let mut router = DualRouter::new(cfg, 4).unwrap();
        let mut planner = DeterministicPlanner;
        let mut worker = DeterministicWorker::default();
        let state = [0.2, -0.1];

        // Publish seq 0, drive once.
        let z0 = router
            .planner_publish(&mut planner, &[0.0, 1.0], &[1.0, 0.0])
            .unwrap();
        let step0 = router.worker_tick(&mut worker, &state);
        let cid0 = match step0 {
            WorkerStep::Drive(ref c) => {
                assert_eq!(c.latent_lid(), z0.lid());
                c.cid()
            }
            _ => panic!("expected Drive"),
        };

        // Publish seq 1 with a DIFFERENT goal -> different latent -> latest-wins swap.
        let z1 = router
            .planner_publish(&mut planner, &[0.0, 1.0], &[0.0, 1.0])
            .unwrap();
        assert_ne!(
            z0.lid(),
            z1.lid(),
            "distinct goals must yield distinct lids"
        );
        assert_eq!(z1.seq(), z0.seq() + 1);

        // Next worker tick now drives the NEW latent, and age resets (published this tick range).
        match router.worker_tick(&mut worker, &state) {
            WorkerStep::Drive(c) => {
                assert_eq!(c.latent_lid(), z1.lid(), "worker must see latest latent");
                assert_ne!(c.cid(), cid0, "new latent must change the chunk cid");
                assert!(c.age_ticks() <= router.cfg().staleness_bound());
                assert!(c.verify_cid());
            }
            other => panic!("expected Drive on fresh latent, got {other:?}"),
        }

        // Immutability audit: both published latents still verify.
        assert!(z0.verify_lid() && z1.verify_lid());
    }

    #[test]
    fn hbp_rows_have_exact_shape() {
        let lid = sha16(&[b"seed-latent"]);
        let planner_row = hbp_planner_row(7, &lid, 32, 10);
        assert_eq!(
            planner_row,
            format!(
                "DSR|role=planner|seq=7|lid={}|dim=32|hz=10|json=0",
                hex16(&lid)
            )
        );

        let hold_no = hbp_hold_row(3, &HoldReason::NoSeam);
        assert_eq!(hold_no, "DSR|role=worker|tick=3|hold=no_seam|json=0");
        let hold_stale = hbp_hold_row(
            9,
            &HoldReason::Stale {
                age_ticks: 12,
                bound: 5,
            },
        );
        assert_eq!(
            hold_stale,
            "DSR|role=worker|tick=9|hold=stale|age_ticks=12|bound=5|json=0"
        );

        // A driven chunk's worker row must carry the cid and end |json=0.
        let chunk = Chunk::new(2, 7, lid, 1, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        let wr = hbp_worker_row(&chunk);
        assert!(wr.contains("role=worker"));
        assert!(wr.contains(&format!("cid={}", hex16(&chunk.cid()))));
        assert!(wr.ends_with("|json=0"));
        assert!(chunk.verify_cid());
    }
}
