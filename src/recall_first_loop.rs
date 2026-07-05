//! recall_first_loop — a bounded, deterministic, recall-before-decide working
//! memory bank rendered as a per-step act loop.
//!
//! # What this is
//! Asolaria's recall-first law made into a constant-cost per-step loop. A
//! memory-less agent is non-Markovian-blind: two visually identical
//! observations with different histories look the same and force a wrong
//! action. This module cures that with a FIXED-SIZE bank that lives *inside*
//! the loop and does four load-bearing moves every step:
//!
//!   1. RETRIEVE by content **and** time — score `s_i = cos(q, m_i) +
//!      lambda*phi(now - t_i)`. Time is carried in the SCORE (keys), never
//!      baked into the stored vector (values carry no positional signal).
//!   2. GATE the blend instead of adding — `x_tilde = g*H + (1-g)*obs`, with
//!      `g = sigmoid(alpha*(mean_topk_score - s0))`. The loop learns/decides
//!      *when* to trust memory instead of unconditionally summing it in.
//!   3. WRITE BACK the fused `x_tilde` (the interpretation the policy actually
//!      used), never the raw observation — that is what makes recall
//!      decision-disambiguating rather than a plain frame buffer.
//!   4. FORGET by MERGE, not FIFO — on overflow, merge the most cosine-similar
//!      *temporally adjacent* pair via a count-weighted running mean. Preserves
//!      temporal order and salient transitions; redundancy-aware.
//!
//! The capacity `L` is the real-time guarantee: every step is O(L*d)
//! regardless of episode length.
//!
//! # Provenance tags
//! * MEASURED-in-paper: the *mechanism ranking* — gating beats addition,
//!   time-PE in the key helps, adjacent-merge beats FIFO, and L is a tuned
//!   (non-monotonic) budget. Those orderings come from the source ablations.
//! * MEASURED-in-paper (NOT us): concrete scores (71.9% etc.), latency numbers,
//!   and specific L optima belong to the paper's *learned* system on *its*
//!   benchmarks. This deterministic std-only re-engineering inherits the
//!   MECHANISM, not the numbers. Our loop's performance is UNVERIFIED until we
//!   benchmark it ourselves — no receipt or doc here claims those scores.
//! * DESIGN (ours, Asolaria improvements over the paper):
//!     - deterministic cos+time scoring with softmax-over-top-k (auditable,
//!       replayable) replacing learned cross-attention;
//!     - `k <= L` as a second, tighter latency knob the paper lacks;
//!     - pluggable [`Gate`] trait so a learned gate slots in later;
//!     - count-weighted merge (unbiased under repeated merges) vs plain 1/2-1/2;
//!     - cached adjacent-pair cosines => overflow-merge is O(d) amortized
//!       instead of a full O(L*d) rescan;
//!     - sha16 content-addressing of every entry AND query for replayable
//!       provenance;
//!     - HBP hot-path journal row per step (`...|json=0`);
//!     - [`StepBudget`] enforced mechanically (degrade k, never block, record
//!       `k_used` so degradation is visible not silent).
//!
//! # Shannon honesty
//! The merge step AVERAGES frames: it is LOSSY by design —
//! `H(original frames | merged bank) > 0`. This bank is a bounded lossy
//! working-memory *summary*. It does NOT give lossless recall, 0-loss
//! compression, or code-rate-1.0. Lossless recovery is a DIFFERENT mechanism
//! (a content-addressed store that retains the full object). Do not conflate.
//!
//! `sha16` here is a 64-bit prefix locator (first 8 bytes of SHA-256): a
//! journal/provenance id with ~2^32 birthday-collision scale. It is NOT an
//! equality proof, dedup oracle, or security boundary. Correctness compares
//! vectors, never hashes.
//!
//! std-only. No external crates. f32 math, `std::f32` transcendentals,
//! `std::time::Instant` for the budget clock.

use std::time::Instant;

// ---------------------------------------------------------------------------
// SHA-256 (std-only) — used only to derive sha16 content addresses.
// Correct, self-contained; verified against the "abc" KAT in the tests.
// ---------------------------------------------------------------------------
mod sha256 {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
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

    /// Full SHA-256 digest of `msg`.
    pub fn digest(msg: &[u8]) -> [u8; 32] {
        let mut h = H0;
        let bit_len = (msg.len() as u64).wrapping_mul(8);
        let mut data = msg.to_vec();
        data.push(0x80);
        while data.len() % 64 != 56 {
            data.push(0);
        }
        data.extend_from_slice(&bit_len.to_be_bytes());

        for chunk in data.chunks_exact(64) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    chunk[4 * i],
                    chunk[4 * i + 1],
                    chunk[4 * i + 2],
                    chunk[4 * i + 3],
                ]);
            }
            for i in 16..64 {
                let s0 =
                    w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
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
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
        }
        out
    }
}

/// Content address = first 8 bytes of `SHA-256(le_bytes(vec) || le_bytes(step))`
/// rendered as 16 lowercase hex chars ("sha16"). Provenance locator, NOT an
/// equality proof (see module Shannon-honesty note).
pub fn sha16(vec: &[f32], step: u64) -> String {
    let mut bytes = Vec::with_capacity(vec.len() * 4 + 8);
    for x in vec {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    bytes.extend_from_slice(&step.to_le_bytes());
    let h = sha256::digest(&bytes);
    let mut s = String::with_capacity(16);
    for b in &h[..8] {
        // std-only hex, no crate.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

const ZERO_ADDR: &str = "0000000000000000";

// ---------------------------------------------------------------------------
// Math helpers (guarded, std-only)
// ---------------------------------------------------------------------------

/// Cosine similarity with a hard zero-norm / non-finite guard returning 0.0.
/// A NaN must never enter the bank — it would poison every future retrieval.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 || !denom.is_finite() {
        return 0.0;
    }
    let c = dot / denom;
    if c.is_finite() {
        c
    } else {
        0.0
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Numerically stable logistic.
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Recency kernel phi(dt) in [0,1]: 1 at dt=0, decaying with scale `tau`.
/// This is the TIME half of content+time addressing — load-bearing, default ON.
#[inline]
fn phi(dt: u64, tau: f32) -> f32 {
    if tau <= 0.0 {
        return 0.0;
    }
    (-(dt as f32) / tau).exp()
}

// ---------------------------------------------------------------------------
// Pluggable gate — DESIGN: the deterministic default; a learned gate can slot
// in by implementing this trait without touching the memory core.
// ---------------------------------------------------------------------------

/// Decides how much to trust recalled memory. Returns g in [0,1].
pub trait Gate {
    fn gate(&self, mean_topk_score: f32, cfg: &RecallConfig) -> f32;
}

/// `g = sigmoid(alpha*(mean_topk_score - s0))`. Replaces the paper's gate MLP.
/// Calibrate `s0` so g stays off-saturation — fused write-back + a saturating
/// gate is a positive-feedback loop; watch `gate_milli` in receipts for drift
/// toward always-trust-memory.
pub struct SigmoidGate;
impl Gate for SigmoidGate {
    fn gate(&self, mean_topk_score: f32, cfg: &RecallConfig) -> f32 {
        sigmoid(cfg.alpha * (mean_topk_score - cfg.s0))
    }
}

// ---------------------------------------------------------------------------
// Orthogonal action head — the memory core never knows what decides.
// ---------------------------------------------------------------------------

/// The action head. Matches the paper's orthogonality claim: memory and policy
/// are decoupled. `decide` consumes the fused `x_tilde`.
pub trait Decide {
    type Action;
    fn decide(&mut self, x_tilde: &[f32]) -> Self::Action;
}

// ---------------------------------------------------------------------------
// Config & budget
// ---------------------------------------------------------------------------

/// Hard, mechanically-enforced per-step budget. Enforcement DEGRADES (halves k)
/// rather than blocks; degradation is recorded in the receipt (`k_used`) so a
/// "real-time" claim can never silently become an over-claim.
#[derive(Clone, Debug)]
pub struct StepBudget {
    /// Hard ceiling on top-k regardless of `RecallConfig::k`.
    pub max_k: usize,
    /// Hard ceiling on bank size regardless of `RecallConfig::capacity_l`.
    pub max_bank: usize,
    /// Soft time budget in microseconds. Exceeding it halves k for NEXT step.
    pub max_us: u128,
}

impl StepBudget {
    /// A permissive budget (won't trigger degradation in normal use).
    pub fn relaxed(max_k: usize, max_bank: usize) -> Self {
        StepBudget {
            max_k,
            max_bank,
            max_us: u128::MAX,
        }
    }
}

/// All tunables. `k` and `capacity_l` are BUDGETS, not "bigger is better":
/// the paper measured L and k as non-monotonic (an over-large L both hurt
/// quality and eroded the latency bound). Defaults below are DESIGN starting
/// points and are UNVERIFIED for any particular task.
#[derive(Clone, Debug)]
pub struct RecallConfig {
    /// Vector dimensionality d.
    pub dim: usize,
    /// Bank capacity L (the real-time guarantee: work is O(L*d)/step).
    pub capacity_l: usize,
    /// Top-k retrieved per step (k <= L; a second, tighter latency knob).
    pub k: usize,
    /// Weight on the time term in the score. Default ON (do not drop it).
    pub lambda: f32,
    /// Recency decay scale for phi.
    pub tau: f32,
    /// Gate slope.
    pub alpha: f32,
    /// Gate threshold (trust-memory midpoint).
    pub s0: f32,
    /// Hard budget.
    pub budget: StepBudget,
}

impl RecallConfig {
    /// DESIGN defaults keyed only on dimensionality; tune per task.
    pub fn new(dim: usize, capacity_l: usize, k: usize) -> Self {
        assert!(dim >= 1, "dim must be >= 1");
        assert!(capacity_l >= 1, "capacity_l must be >= 1");
        assert!(k >= 1, "k must be >= 1");
        RecallConfig {
            dim,
            capacity_l,
            k,
            lambda: 0.25,
            tau: 8.0,
            alpha: 4.0,
            s0: 0.5,
            budget: StepBudget::relaxed(k, capacity_l),
        }
    }
}

// ---------------------------------------------------------------------------
// Bank entry & receipt
// ---------------------------------------------------------------------------

/// One temporally-ordered slot. `vec` is the FUSED representation the policy
/// used (never a raw observation, never time-embedded). `count` tracks how many
/// original frames were merged in (for unbiased count-weighted merging).
#[derive(Clone, Debug)]
pub struct MemEntry {
    pub vec: Vec<f32>,
    /// Earliest timestep represented by this slot (kept on merge).
    pub t: u64,
    /// Number of original frames merged into this slot.
    pub count: u32,
    /// sha16 content address of (vec, t).
    pub addr: String,
}

/// Replayable per-step provenance. Everything here except `us` is a pure
/// function of the observation stream (deterministic). `us` is the sole
/// wall-clock field.
#[derive(Clone, Debug)]
pub struct StepReceipt {
    pub step: u64,
    /// Number of entries actually combined this step (post budget degradation).
    pub k_used: usize,
    /// Bank size after write-back + any overflow merge.
    pub bank: usize,
    /// Whether an overflow merge happened this step.
    pub merged: bool,
    /// Gate value * 1000, clamped 0..=1000. The trust dial, for drift monitoring.
    pub gate_milli: u32,
    /// sha16 of the query (observation, step).
    pub q_addr: String,
    /// sha16 of the top-scored entry recalled ("0"*16 if bank was empty).
    pub top_addr: String,
    /// Wall-clock microseconds for this step (the ONLY nondeterministic field).
    pub us: u128,
}

impl StepReceipt {
    /// HBP hot-path journal row. JSON is never emitted here.
    pub fn to_hbp(&self) -> String {
        format!(
            "RECALLSTEP|step={}|k={}|bank={}|merged={}|gate_milli={}|q={}|top={}|us={}|json=0",
            self.step,
            self.k_used,
            self.bank,
            self.merged as u8,
            self.gate_milli,
            self.q_addr,
            self.top_addr,
            self.us,
        )
    }

    /// Deterministic subset of [`to_hbp`](Self::to_hbp) with the wall-clock
    /// `us` omitted — byte-identical across replays of the same obs stream.
    pub fn canonical_row(&self) -> String {
        format!(
            "RECALLSTEP|step={}|k={}|bank={}|merged={}|gate_milli={}|q={}|top={}|json=0",
            self.step,
            self.k_used,
            self.bank,
            self.merged as u8,
            self.gate_milli,
            self.q_addr,
            self.top_addr,
        )
    }
}

// ---------------------------------------------------------------------------
// The bank
// ---------------------------------------------------------------------------

/// Fixed-size recall bank living inside the act loop.
pub struct RecallBank<G: Gate = SigmoidGate> {
    entries: Vec<MemEntry>,
    /// Cached adjacent-pair cosines: `adj_cos[i] = cos(entries[i], entries[i+1])`.
    /// Length is always `entries.len().saturating_sub(1)`. This is what makes
    /// overflow-merge O(d) amortized instead of a full O(L*d) rescan.
    adj_cos: Vec<f32>,
    cfg: RecallConfig,
    gate: G,
    /// Current step counter (also the "now" used by phi).
    step: u64,
    /// Adaptive top-k after budget pressure; starts at effective k.
    cur_k: usize,
    /// Scratch buffers reused across steps (no allocator churn).
    score_buf: Vec<f32>,
    idx_buf: Vec<usize>,
}

impl RecallBank<SigmoidGate> {
    /// Build with the default deterministic sigmoid gate.
    pub fn new(cfg: RecallConfig) -> Self {
        Self::with_gate(cfg, SigmoidGate)
    }
}

impl<G: Gate> RecallBank<G> {
    /// Build with a custom [`Gate`].
    pub fn with_gate(cfg: RecallConfig, gate: G) -> Self {
        let cap = cfg.capacity_l.min(cfg.budget.max_bank).max(1);
        let start_k = cfg.k.min(cfg.budget.max_k).max(1);
        RecallBank {
            entries: Vec::with_capacity(cap + 1),
            adj_cos: Vec::with_capacity(cap),
            cfg,
            gate,
            step: 0,
            cur_k: start_k,
            score_buf: Vec::new(),
            idx_buf: Vec::new(),
        }
    }

    /// Effective bank capacity = min(configured L, hard max_bank).
    #[inline]
    pub fn effective_capacity(&self) -> usize {
        self.cfg.capacity_l.min(self.cfg.budget.max_bank).max(1)
    }

    /// Upper bound on top-k = min(configured k, hard max_k).
    #[inline]
    fn effective_k_ceiling(&self) -> usize {
        self.cfg.k.min(self.cfg.budget.max_k).max(1)
    }

    /// Current number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Read-only view of the temporally-ordered bank.
    pub fn entries(&self) -> &[MemEntry] {
        &self.entries
    }

    /// The core loop step: RETRIEVE (content+time) -> GATE -> FUSE ->
    /// WRITE-BACK the fused vector -> FORGET-by-merge on overflow.
    /// Returns the fused `x_tilde` and a replayable [`StepReceipt`].
    ///
    /// Guaranteed bounded: never allocates unboundedly, never blocks, and the
    /// bank never exceeds [`effective_capacity`](Self::effective_capacity).
    pub fn step(&mut self, obs: &[f32]) -> (Vec<f32>, StepReceipt) {
        assert_eq!(obs.len(), self.cfg.dim, "obs dim mismatch");
        let t0 = Instant::now();
        let now = self.step;
        let dim = self.cfg.dim;

        let q_addr = sha16(obs, now);

        // --- 1. RETRIEVE: score every entry by content + time. -------------
        let n = self.entries.len();
        self.score_buf.clear();
        self.score_buf.reserve(n);
        for e in &self.entries {
            let dt = now.saturating_sub(e.t);
            let s = cosine(obs, &e.vec) + self.cfg.lambda * phi(dt, self.cfg.tau);
            self.score_buf.push(s);
        }

        // Hard top-k (k bounded by cur_k, max_k ceiling, and bank size).
        let k_eff = self.cur_k.min(self.effective_k_ceiling()).min(n);

        // Partial-ish selection via full sort of indices by score desc. n is
        // bounded by L, so this stays within the O(L*d) step budget (scoring,
        // at O(n*d), dominates the O(n log n) sort whenever d >= log n).
        self.idx_buf.clear();
        self.idx_buf.extend(0..n);
        {
            let scores = &self.score_buf;
            self.idx_buf.sort_by(|&a, &b| {
                scores[b]
                    .partial_cmp(&scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // --- 2. GATE + FUSE. ------------------------------------------------
        let mut fused = vec![0.0f32; dim];
        let (g, top_addr);
        if k_eff == 0 {
            // Empty bank: nothing to recall, trust the observation fully.
            fused.copy_from_slice(obs);
            g = 0.0;
            top_addr = ZERO_ADDR.to_string();
        } else {
            let sel = &self.idx_buf[..k_eff];
            // softmax over the top-k RAW scores with max-subtraction.
            let mut max_s = f32::NEG_INFINITY;
            let mut sum_s = 0.0f32;
            for &j in sel {
                let s = self.score_buf[j];
                if s > max_s {
                    max_s = s;
                }
                sum_s += s;
            }
            let mut wsum = 0.0f32;
            let mut h = vec![0.0f32; dim];
            for &j in sel {
                let w = (self.score_buf[j] - max_s).exp();
                wsum += w;
                let v = &self.entries[j].vec;
                for d in 0..dim {
                    h[d] += w * v[d];
                }
            }
            if wsum > 0.0 && wsum.is_finite() {
                for d in 0..dim {
                    h[d] /= wsum;
                }
            }
            let mean_topk = sum_s / (k_eff as f32);
            let gate_val = self.gate.gate(mean_topk, &self.cfg).clamp(0.0, 1.0);
            // x_tilde = g*H + (1-g)*obs  (element-wise blend, NOT addition).
            for d in 0..dim {
                fused[d] = gate_val * h[d] + (1.0 - gate_val) * obs[d];
            }
            g = gate_val;
            top_addr = self.entries[sel[0]].addr.clone();
        }
        let gate_milli = (g * 1000.0).round().clamp(0.0, 1000.0) as u32;

        // --- 3. WRITE BACK the fused representation (never the raw obs). ----
        let addr = sha16(&fused, now);
        self.push_entry(MemEntry {
            vec: fused.clone(),
            t: now,
            count: 1,
            addr,
        });

        // --- 4. FORGET by merge on overflow (adjacent, count-weighted). -----
        let cap = self.effective_capacity();
        let mut merged = false;
        while self.entries.len() > cap {
            self.merge_most_similar_adjacent();
            merged = true;
        }

        // --- Budget enforcement: degrade k, never block; record k_used. -----
        let us = t0.elapsed().as_micros();
        if us > self.cfg.budget.max_us {
            self.cur_k = (self.cur_k / 2).max(1);
        } else if self.cur_k < self.effective_k_ceiling() {
            // AIMD-style gentle recovery so we don't oscillate hard.
            self.cur_k += 1;
        }

        let receipt = StepReceipt {
            step: now,
            k_used: k_eff,
            bank: self.entries.len(),
            merged,
            gate_milli,
            q_addr,
            top_addr,
            us,
        };
        self.step += 1;
        (fused, receipt)
    }

    /// Convenience: recall + fuse + hand the fused vector to an action head.
    pub fn act<D: Decide>(&mut self, obs: &[f32], decider: &mut D) -> (D::Action, StepReceipt) {
        let (x, r) = self.step(obs);
        (decider.decide(&x), r)
    }

    // --- internal: append and maintain the trailing adjacency cosine. ------
    fn push_entry(&mut self, e: MemEntry) {
        if let Some(last) = self.entries.last() {
            let c = cosine(&last.vec, &e.vec);
            self.adj_cos.push(c);
        }
        self.entries.push(e);
        debug_assert_eq!(self.adj_cos.len(), self.entries.len().saturating_sub(1));
    }

    // --- internal: merge the most cosine-similar temporally-adjacent pair. --
    // Count-weighted running mean; keep earliest t; sum counts (saturating).
    // Only <=2 adjacency cosines change => O(d) amortized, not a full rescan.
    fn merge_most_similar_adjacent(&mut self) {
        debug_assert!(self.entries.len() >= 2);
        // argmax over the cached adjacency cosines.
        let mut bi = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &c) in self.adj_cos.iter().enumerate() {
            if c > bv {
                bv = c;
                bi = i;
            }
        }
        let a = bi;
        let b = bi + 1;
        let ca = self.entries[a].count as f32;
        let cb = self.entries[b].count as f32;
        let denom = ca + cb;
        let dim = self.cfg.dim;
        let mut nv = vec![0.0f32; dim];
        for d in 0..dim {
            nv[d] = (ca * self.entries[a].vec[d] + cb * self.entries[b].vec[d]) / denom;
        }
        let nt = self.entries[a].t.min(self.entries[b].t); // earliest kept
        let ncount = self.entries[a].count.saturating_add(self.entries[b].count);
        let naddr = sha16(&nv, nt);
        self.entries[a] = MemEntry {
            vec: nv,
            t: nt,
            count: ncount,
            addr: naddr,
        };
        self.entries.remove(b);

        // Repair adjacency cache: drop the merged pair's cos, recompute the
        // (at most two) neighbours that now touch the merged slot.
        self.adj_cos.remove(bi);
        if bi > 0 {
            self.adj_cos[bi - 1] = cosine(&self.entries[bi - 1].vec, &self.entries[bi].vec);
        }
        if bi < self.adj_cos.len() {
            self.adj_cos[bi] = cosine(&self.entries[bi].vec, &self.entries[bi + 1].vec);
        }
        debug_assert_eq!(self.adj_cos.len(), self.entries.len().saturating_sub(1));
    }
}

// ===========================================================================
// Tests — exercise the real mechanism, not stubs.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-observation generator (std-only, no rand crate).
    fn obs_stream(dim: usize, seed: u64, step: u64) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        let mut s = seed ^ step.wrapping_mul(0x9E3779B97F4A7C15);
        for d in 0..dim {
            // xorshift64*
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545F4914F6CDD1D);
            v[d] = ((r >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        }
        v
    }

    #[test]
    fn sha256_known_answer() {
        // SHA-256("abc") = ba7816bf8f01cfea...; sha16 takes the first 8 bytes.
        let h = sha256::digest(b"abc");
        let hex: String = h[..8].iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "ba7816bf8f01cfea");
    }

    #[test]
    fn bank_never_exceeds_capacity() {
        let l = 16;
        let dim = 12;
        let mut bank = RecallBank::new(RecallConfig::new(dim, l, 4));
        for step in 0..2000u64 {
            let obs = obs_stream(dim, 0xABCD, step);
            let (x, r) = bank.step(&obs);
            assert_eq!(x.len(), dim);
            // Hard invariant: constant memory regardless of episode length.
            assert!(bank.len() <= l, "bank {} exceeded L {}", bank.len(), l);
            assert_eq!(r.bank, bank.len());
            // Receipt fields must be present and finite-ish.
            assert!(r.gate_milli <= 1000);
            assert_eq!(r.q_addr.len(), 16);
            assert_eq!(r.top_addr.len(), 16);
            // No NaN may enter the bank.
            for e in bank.entries() {
                assert!(e.vec.iter().all(|x| x.is_finite()));
            }
        }
    }

    #[test]
    fn count_weighted_merge_is_exact_running_mean() {
        // Build a bank whose overflow forces merges, then check that a merged
        // slot equals the count-weighted mean of the frames folded into it.
        // We verify the merge math directly against a brute-force weighted mean.
        let dim = 3;
        // Three "frames" that will be adjacent & similar enough to merge in order.
        // Force merges by capacity 2 so every push beyond the first merges.
        let mut cfg = RecallConfig::new(dim, 2, 2);
        cfg.lambda = 0.0; // isolate content merging from time in this check
        let mut bank = RecallBank::new(cfg);

        // Feed 4 identical-direction vectors of increasing magnitude; because
        // cosine(adjacent)~1 they will keep merging into a single slot.
        let frames = [
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
            vec![3.0, 0.0, 0.0],
            vec![4.0, 0.0, 0.0],
        ];
        // Note: the bank stores FUSED x_tilde, not raw frames. With an empty
        // bank the first fuse returns obs verbatim (g=0), so entry[0].vec == frame0.
        // Subsequent fuses blend, so we can't compare stored vecs to raw frames.
        // Instead we test the merge primitive on a controlled bank below.
        for f in &frames {
            bank.step(f);
        }
        assert!(bank.len() <= 2);

        // Direct, deterministic check of the count-weighted merge primitive:
        // merge slot A (count 2, mean [1, .]) with slot B (count 3, mean [6, .]);
        // result must be the count-weighted mean = (2*1 + 3*6)/5 = 4.0, count 5.
        let dim2 = 1;
        let mut cfg2 = RecallConfig::new(dim2, 2, 1);
        cfg2.lambda = 0.0;
        let mut b2: RecallBank = RecallBank::new(cfg2);
        // Hand-construct two entries via the internal push (through steps that
        // then get merged). Simpler: reach into the struct through a fresh bank.
        b2.entries.push(MemEntry {
            vec: vec![1.0],
            t: 0,
            count: 2,
            addr: sha16(&[1.0], 0),
        });
        b2.entries.push(MemEntry {
            vec: vec![6.0],
            t: 1,
            count: 3,
            addr: sha16(&[6.0], 1),
        });
        b2.adj_cos.push(cosine(&[1.0], &[6.0]));
        b2.merge_most_similar_adjacent();
        assert_eq!(b2.len(), 1);
        let m = &b2.entries()[0];
        let expected = (2.0 * 1.0 + 3.0 * 6.0) / 5.0; // 4.0
        assert!(
            (m.vec[0] - expected).abs() < 1e-6,
            "got {}, expected {}",
            m.vec[0],
            expected
        );
        assert_eq!(m.count, 5);
        assert_eq!(m.t, 0, "earliest timestep must be kept");
    }

    #[test]
    fn deterministic_receipt_replay() {
        // Same obs stream -> byte-identical canonical receipt rows (us excluded).
        let dim = 8;
        let run = || {
            let mut bank = RecallBank::new(RecallConfig::new(dim, 8, 3));
            let mut rows = Vec::new();
            for step in 0..200u64 {
                let obs = obs_stream(dim, 0x1234, step);
                let (_x, r) = bank.step(&obs);
                rows.push(r.canonical_row());
            }
            rows
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "canonical receipts must replay byte-identically");
        // And a spot-check that the HBP row has the required shape + suffix.
        assert!(a[0].starts_with("RECALLSTEP|step=0|"));
        assert!(a[0].ends_with("|json=0"));
    }

    #[test]
    fn budget_degrades_k_and_records_it_never_blocks() {
        // An impossibly tight time budget forces k to halve toward 1 while the
        // loop keeps running (never blocks) and records k_used every step.
        let dim = 10;
        let mut cfg = RecallConfig::new(dim, 32, 16);
        cfg.budget.max_us = 0; // every step is "over budget"
        let mut bank = RecallBank::new(cfg);
        // Prime the bank so there is something to recall.
        for step in 0..40u64 {
            bank.step(&obs_stream(dim, 0x55, step));
        }
        let (_x, r) = bank.step(&obs_stream(dim, 0x55, 40));
        // Under sustained pressure cur_k collapses to the floor of 1.
        assert!(r.k_used >= 1, "k_used must never fall below 1");
        assert!(
            r.k_used <= 2,
            "under a zero budget k should have degraded to ~1, got {}",
            r.k_used
        );
        // Bank still bounded, loop still alive.
        assert!(bank.len() <= 32);
    }

    #[test]
    fn zero_norm_cosine_is_guarded() {
        assert_eq!(cosine(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]), 0.0);
        assert_eq!(cosine(&[0.0, 0.0], &[0.0, 0.0]), 0.0);
        // A zero observation into an empty bank must not produce NaN.
        let mut bank = RecallBank::new(RecallConfig::new(3, 4, 2));
        let (x, r) = bank.step(&[0.0, 0.0, 0.0]);
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(r.gate_milli, 0, "empty bank => gate off");
    }

    #[test]
    fn gate_and_hbp_row_shape() {
        let dim = 4;
        let mut bank = RecallBank::new(RecallConfig::new(dim, 4, 2));
        let (_x, r) = bank.step(&[1.0, 0.0, 0.0, 0.0]);
        let row = r.to_hbp();
        assert!(row.contains("|gate_milli="));
        assert!(row.contains("|q="));
        assert!(row.ends_with("|json=0"));
        assert!(!row.contains("json=1"));
    }
}
