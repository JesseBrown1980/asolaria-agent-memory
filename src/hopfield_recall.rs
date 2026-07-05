//! # hopfield_recall — modern (continuous) Hopfield associative memory, std-only
//!
//! Clean-room re-engineering (WHITE-ROOM discipline: built from the *mechanism*,
//! not from any source) of the **modern Hopfield network** as content-addressable
//! recall.
//!
//! ## The one identity this whole file rests on (MEASURED — Ramsauer et al. 2020,
//! "Hopfield Networks is All You Need", arXiv:2008.02217)
//!
//! The entire associative memory collapses to a single update line:
//!
//! ```text
//!     xi_new = X^T · softmax(beta · X · xi)
//! ```
//!
//! i.e. **transformer attention with a single query IS content-addressable recall.**
//! - Storage is trivially appending a row to `X` (no training, no backprop, no
//!   learned weight matrix).
//! - Recall is one matvec + a stable softmax + one matvec: `O(N·d)`.
//!
//! Three paper-proven properties make it load-bearing (all tagged MEASURED here,
//! meaning *proven in the cited paper* — on-machine numbers are separate):
//! 1. **ONE-STEP RETRIEVAL** — a single update lands within ~`exp(-beta·Delta_i)`
//!    of the closest stored pattern when patterns are separated
//!    (`Delta_i = self-similarity − max cross-similarity`), so recall is
//!    effectively a single pass, not an iterative settle (paper Thm 4 / A.2).
//! 2. **EXPONENTIAL CAPACITY** — `N ~ c^((d-1)/4)` well-separated random spherical
//!    patterns are exactly retrievable (paper Thm 3), so a modest `d` holds a huge
//!    dictionary.
//! 3. **SAFE ITERATION** — the update is CCCP descent on a log-sum-exp energy, so
//!    energy `E` monotonically decreases and iterating can never diverge; `beta` is
//!    a single dial (large = sharp single-pattern recall, small = metastable
//!    cluster average).
//!
//! **Free bonus (falls out of the math):** the softmax weights `a_i` are already a
//! probability distribution over the store, so `a_max` is a native *confidence* and
//! `a_(1) − a_(2)` is a native *margin* — a recover-or-HOLD gate for zero extra cost.
//!
//! ## MEASURED vs DESIGN split (kept explicit per the claims-gate)
//! - **MEASURED (cited from the paper):** the update rule, LSE energy + its monotone
//!   descent, the one-step error bound, exponential capacity, the `beta` regime, and
//!   `beta = 1/sqrt(d)` as the transformer-scaling identity.
//! - **DESIGN (Asolaria engineering, NOT covered by the paper's bounds):** sha16
//!   content addressing, snap-to-stored-row (Path-1 recall-of-retained-object),
//!   masked/partial-cue recall, the confidence/margin recover-or-HOLD gate, and the
//!   HBP hot-path tuple row.
//!
//! ## SHANNON boundary (do not re-inflate)
//! This is **recall of a RETAINED object** (Path-1), *not* inversion of a lossy
//! shadow. A partial cue *selects* from a dictionary: information recovered is
//! `<= log2(N)` bits of selection plus what the cue itself carries; the returned
//! pattern's bytes come from the **STORE**, not the cue. Never phrase this as
//! "reconstructs data below entropy" or "lossless recovery from a lossy cue".
//!
//! Dependency surface: `std` + the in-house KAT-verified SHA-256 below. No ndarray,
//! nalgebra, rayon, serde, or external `sha2`.

use std::collections::HashMap;

/// A content address: the first 8 bytes of SHA-256 over a pattern's little-endian
/// `f32` bytes. **DESIGN** — an *address*, not a proof: 8 bytes has birthday
/// collisions near `2^32` entries, so [`HopfieldMemory::store`] verifies full-pattern
/// equality on an address hit before declaring a dedup.
pub type Sha16 = [u8; 8];

/// Errors from mutating the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HopfieldError {
    /// Pattern length did not match the memory's `dim`.
    DimMismatch { expected: usize, got: usize },
    /// Pattern contained a non-finite value (NaN/Inf). A poisoned row would corrupt
    /// every future recall, so we reject at the boundary.
    NonFinite { index: usize },
    /// A sha16 address collided but the stored bytes differ from the new pattern
    /// (astronomically unlikely; surfaced rather than silently overwriting).
    AddressCollision,
}

impl std::fmt::Display for HopfieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HopfieldError::DimMismatch { expected, got } => {
                write!(f, "dim mismatch: expected {expected}, got {got}")
            }
            HopfieldError::NonFinite { index } => {
                write!(f, "non-finite value at index {index}")
            }
            HopfieldError::AddressCollision => write!(f, "sha16 address collision"),
        }
    }
}

impl std::error::Error for HopfieldError {}

/// The result of a recall. Carries both the **snapped** stored row (byte-identical
/// to what was stored — Path-1 semantics) and the raw **mixture** `X^T·a`, so a
/// metastable blend is never hidden. `confidence`/`margin` drive a recover-or-HOLD
/// gate; `mixture_dist` is `||mixture − snapped||_2`, the honest distance between the
/// soft answer and the hard snap.
#[derive(Debug, Clone)]
pub struct Recall {
    /// Content address of the snapped stored pattern.
    pub id: Sha16,
    /// Row index of the snapped stored pattern.
    pub idx: usize,
    /// The stored row, byte-identical to what `store()` received (Path-1).
    pub snapped: Vec<f32>,
    /// Raw attention mixture `X^T·a` (DESIGN: exposed so blends are visible).
    pub mixture: Vec<f32>,
    /// `a_max` — softmax weight of the winner. **Not** calibrated P(correct).
    pub confidence: f32,
    /// `a_(1) − a_(2)` — gap between the top two softmax weights.
    pub margin: f32,
    /// Iterations performed (1 for a single-step recall; expected ~1 always).
    pub steps: usize,
    /// LSE energy at the final iterate (MEASURED: non-increasing across steps).
    pub energy: f32,
    /// The `beta` actually used (after `<=0` defaulting).
    pub beta: f32,
    /// `||mixture − snapped||_2`: how far the soft mixture sits from the hard snap.
    pub mixture_dist: f32,
}

impl Recall {
    /// Recover-or-HOLD gate (DESIGN, same grammar as NQPrismNexus recover-or-Hold):
    /// returns `true` (safe to act on the snapped row) only when both confidence and
    /// margin clear the caller's thresholds. Below either, the caller should HOLD
    /// rather than guess — near-duplicates can give high confidence on the wrong twin.
    pub fn accepts(&self, min_confidence: f32, min_margin: f32) -> bool {
        self.confidence >= min_confidence && self.margin >= min_margin
    }
}

/// A modern-Hopfield associative memory: a flat row-major matrix `X` of stored
/// patterns plus a content-address index. Recall is one MEASURED attention step.
#[derive(Debug, Clone)]
pub struct HopfieldMemory {
    dim: usize,
    /// Row-major `N * dim` flat store (no ndarray — a linear scan of dot products).
    data: Vec<f32>,
    /// Content address per row, index-aligned with `data`.
    ids: Vec<Sha16>,
    /// `address -> row index` for O(1) `get`.
    index: HashMap<Sha16, usize>,
    /// Default inverse-temperature; `<=0` on a call means "use transformer scaling".
    default_beta: f32,
}

/// Internal single-step attention result.
struct Attn {
    argmax: usize,
    confidence: f32,
    margin: f32,
    mixture: Vec<f32>,
}

impl HopfieldMemory {
    /// New empty memory over `dim`-dimensional patterns. Default `beta` is the
    /// transformer-scaling identity `1/sqrt(dim)` (MEASURED).
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        HopfieldMemory {
            dim,
            data: Vec::new(),
            ids: Vec::new(),
            index: HashMap::new(),
            default_beta: 1.0 / (dim as f32).sqrt(),
        }
    }

    /// New empty memory with an explicit default `beta` (`<=0` falls back to
    /// `1/sqrt(dim)`).
    pub fn with_beta(dim: usize, beta: f32) -> Self {
        let mut m = Self::new(dim);
        if beta > 0.0 {
            m.default_beta = beta;
        }
        m
    }

    /// Number of stored patterns `N`.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Pattern dimensionality `d`.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The active default `beta`.
    pub fn default_beta(&self) -> f32 {
        self.default_beta
    }

    /// Content address of a pattern (DESIGN): sha16 = first 8 bytes of SHA-256 over
    /// the little-endian `f32` bytes. Exact-bytes only — two semantically-equal
    /// patterns differing by one ULP hash differently.
    pub fn address(pattern: &[f32]) -> Sha16 {
        let mut bytes = Vec::with_capacity(pattern.len() * 4);
        for &v in pattern {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let full = sha256::digest(&bytes);
        let mut a = [0u8; 8];
        a.copy_from_slice(&full[..8]);
        a
    }

    /// Store a pattern; returns its content address. Idempotent: storing a
    /// byte-identical pattern again returns the same id and does not grow the store.
    ///
    /// Validates `dim` and rejects non-finite values (a NaN row poisons every
    /// recall). On an address hit, compares the **full** stored bytes before
    /// declaring a dedup (sha16 is an address, not a proof).
    pub fn store(&mut self, pattern: &[f32]) -> Result<Sha16, HopfieldError> {
        if pattern.len() != self.dim {
            return Err(HopfieldError::DimMismatch {
                expected: self.dim,
                got: pattern.len(),
            });
        }
        for (i, &v) in pattern.iter().enumerate() {
            if !v.is_finite() {
                return Err(HopfieldError::NonFinite { index: i });
            }
        }
        let id = Self::address(pattern);
        if let Some(&existing) = self.index.get(&id) {
            let row = self.row(existing);
            if row == pattern {
                return Ok(id); // exact dedup
            }
            return Err(HopfieldError::AddressCollision);
        }
        let idx = self.ids.len();
        self.data.extend_from_slice(pattern);
        self.ids.push(id);
        self.index.insert(id, idx);
        Ok(id)
    }

    /// O(1) fetch of a stored pattern by content address.
    pub fn get(&self, id: &Sha16) -> Option<&[f32]> {
        self.index.get(id).map(|&i| self.row(i))
    }

    /// The stored id at a row index.
    pub fn id_at(&self, idx: usize) -> Option<Sha16> {
        self.ids.get(idx).copied()
    }

    /// Borrow row `i` of the flat store.
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Resolve a caller `beta`: `<=0` means transformer scaling `1/sqrt(dim)`.
    fn resolve_beta(&self, beta: f32) -> f32 {
        if beta > 0.0 {
            beta
        } else {
            self.default_beta
        }
    }

    /// Dot product `X_i · xi` with an f64 accumulator for numerical honesty
    /// (f32 accumulation over large `d` drifts).
    fn dot_row(&self, i: usize, xi: &[f32]) -> f32 {
        let row = self.row(i);
        let mut acc = 0.0f64;
        for k in 0..self.dim {
            acc += row[k] as f64 * xi[k] as f64;
        }
        acc as f32
    }

    /// Masked dot: only dims where `mask[k]` is true contribute (DESIGN).
    fn dot_row_masked(&self, i: usize, xi: &[f32], mask: &[bool]) -> f32 {
        let row = self.row(i);
        let mut acc = 0.0f64;
        for k in 0..self.dim {
            if mask[k] {
                acc += row[k] as f64 * xi[k] as f64;
            }
        }
        acc as f32
    }

    /// LSE energy at `xi` (MEASURED — the CCCP objective, Ramsauer et al. eq. 6):
    ///
    /// ```text
    ///     E(xi) = -lse(beta, X·xi) + 0.5·||xi||^2
    /// ```
    ///
    /// where `lse(beta, z) = (1/beta)·log sum_i exp(beta·z_i)`, computed max-stably.
    /// The paper's additive constants (`beta^-1·log N`, `0.5·M^2`) are dropped: they
    /// do not affect the monotone-descent property that this function is used to
    /// witness. The update always decreases this `E`.
    pub fn energy(&self, xi: &[f32], beta: f32) -> f32 {
        assert_eq!(xi.len(), self.dim, "xi dim mismatch");
        let beta = self.resolve_beta(beta);
        let n = self.len();
        if n == 0 {
            return 0.0;
        }
        let mut z = Vec::with_capacity(n);
        let mut zmax = f32::NEG_INFINITY;
        for i in 0..n {
            let zi = self.dot_row(i, xi);
            if zi > zmax {
                zmax = zi;
            }
            z.push(zi);
        }
        // sum exp(beta*(z_i - zmax)) in f64, then lse = zmax + (1/beta) log(sum).
        let mut s = 0.0f64;
        for &zi in &z {
            s += ((beta * (zi - zmax)) as f64).exp();
        }
        let lse = zmax as f64 + s.ln() / beta as f64;
        let mut sq = 0.0f64;
        for &v in xi {
            sq += v as f64 * v as f64;
        }
        (-lse + 0.5 * sq) as f32
    }

    /// Core single attention step (MEASURED update; snap + gate are DESIGN).
    /// `mask = None` uses all dims; `Some(mask)` restricts logits to known dims and,
    /// if `rescale`, multiplies each logit by `dim/|known|` to keep the `beta` regime.
    fn attention(&self, xi: &[f32], beta: f32, mask: Option<(&[bool], bool)>) -> Attn {
        let n = self.len();
        let mut logits = Vec::with_capacity(n);
        let mut lmax = f32::NEG_INFINITY;
        let scale = match mask {
            Some((m, true)) => {
                let known = m.iter().filter(|&&b| b).count().max(1);
                self.dim as f32 / known as f32
            }
            _ => 1.0,
        };
        for i in 0..n {
            let raw = match mask {
                Some((m, _)) => self.dot_row_masked(i, xi, m),
                None => self.dot_row(i, xi),
            };
            let l = beta * scale * raw;
            if l > lmax {
                lmax = l;
            }
            logits.push(l);
        }
        // max-subtracted softmax (mandatory at large beta — naive exp overflows).
        // Accumulate exps in f64, then normalize into f32 probability weights.
        let mut exps = Vec::with_capacity(n);
        let mut sum = 0.0f64;
        for &l in &logits {
            let e = ((l - lmax) as f64).exp();
            exps.push(e);
            sum += e;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        let mut weights = Vec::with_capacity(n);
        let mut argmax = 0usize;
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        for (i, &e) in exps.iter().enumerate() {
            let p = (e * inv) as f32;
            weights.push(p);
            if p > top1 {
                top2 = top1;
                top1 = p;
                argmax = i;
            } else if p > top2 {
                top2 = p;
            }
        }
        if top2.is_infinite() {
            top2 = 0.0; // N == 1
        }
        // mixture = X^T a  (full-dim projection, regardless of masking).
        // f64 accumulator for numerical honesty, cast down at the end.
        let mut mix = vec![0.0f64; self.dim];
        for i in 0..n {
            let w = weights[i] as f64;
            if w == 0.0 {
                continue;
            }
            let row = self.row(i);
            for k in 0..self.dim {
                mix[k] += w * row[k] as f64;
            }
        }
        let mixture: Vec<f32> = mix.iter().map(|&x| x as f32).collect();
        Attn {
            argmax,
            confidence: top1,
            margin: top1 - top2,
            mixture,
        }
    }

    /// Build a [`Recall`] from a finished attention step at iterate `xi`.
    fn finish(&self, attn: Attn, steps: usize, beta: f32, xi: &[f32]) -> Recall {
        let idx = attn.argmax;
        let snapped = self.row(idx).to_vec();
        let mut d2 = 0.0f64;
        for k in 0..self.dim {
            let diff = attn.mixture[k] as f64 - snapped[k] as f64;
            d2 += diff * diff;
        }
        Recall {
            id: self.ids[idx],
            idx,
            snapped,
            mixture: attn.mixture,
            confidence: attn.confidence,
            margin: attn.margin,
            steps,
            energy: self.energy(xi, beta),
            beta,
            mixture_dist: (d2.sqrt()) as f32,
        }
    }

    /// **One-step recall** (MEASURED update, DESIGN snap): a single attention step,
    /// then snap to the argmax **stored** row (byte-identical, Path-1). The raw
    /// mixture and confidence/margin ride along. `beta <= 0` uses `1/sqrt(dim)`.
    /// Returns `None` on an empty store or a `dim` mismatch.
    pub fn recall(&self, cue: &[f32], beta: f32) -> Option<Recall> {
        if self.is_empty() || cue.len() != self.dim {
            return None;
        }
        let beta = self.resolve_beta(beta);
        let attn = self.attention(cue, beta, None);
        Some(self.finish(attn, 1, beta, cue))
    }

    /// **Masked / partial-cue recall (DESIGN — does NOT inherit the paper's one-step
    /// bound).** Logits are formed over known dims only (`mask[k] == true`), optionally
    /// rescaled by `dim/|known|` to preserve the `beta` regime. Validate empirically;
    /// masking changes the effective geometry.
    pub fn recall_masked(
        &self,
        cue: &[f32],
        mask: &[bool],
        beta: f32,
        rescale: bool,
    ) -> Option<Recall> {
        if self.is_empty() || cue.len() != self.dim || mask.len() != self.dim {
            return None;
        }
        let beta = self.resolve_beta(beta);
        let attn = self.attention(cue, beta, Some((mask, rescale)));
        Some(self.finish(attn, 1, beta, cue))
    }

    /// **Iterated recall** (MEASURED safe iteration): loop the update until
    /// `||delta||_inf < eps` or `max_steps`, asserting energy is non-increasing every
    /// step (the CCCP guarantee doubles as a built-in self-test). Expected to settle
    /// in ~1 step for well-separated patterns; a metastable settle shows up as extra
    /// steps and a small margin in the result. Snaps to the argmax stored row at the
    /// end.
    pub fn recall_iter(&self, cue: &[f32], beta: f32, eps: f32, max_steps: usize) -> Option<Recall> {
        if self.is_empty() || cue.len() != self.dim {
            return None;
        }
        let beta = self.resolve_beta(beta);
        let mut xi = cue.to_vec();
        let mut prev_e = self.energy(&xi, beta);
        let mut steps = 0usize;
        let mut last: Attn;
        loop {
            let attn = self.attention(&xi, beta, None);
            let mut delta_inf = 0.0f32;
            for k in 0..self.dim {
                let d = (attn.mixture[k] - xi[k]).abs();
                if d > delta_inf {
                    delta_inf = d;
                }
            }
            let e = self.energy(&attn.mixture, beta);
            // MEASURED guarantee: energy must not increase (fp tolerance).
            debug_assert!(
                e <= prev_e + 1e-3,
                "energy increased: {e} > {prev_e} (violates CCCP descent)"
            );
            steps += 1;
            xi = attn.mixture.clone();
            prev_e = e;
            last = attn;
            if delta_inf < eps || steps >= max_steps {
                break;
            }
        }
        Some(self.finish(last, steps, beta, &xi))
    }

    /// **HOT-PATH FIRST (DESIGN):** emit one HBP tuple row for a recall. This is the
    /// primary ledger form; JSON is cold debug only. Row grammar:
    ///
    /// ```text
    /// HOPRECALL|id=<sha16-hex>|idx=<i>|conf=<f>|margin=<f>|steps=<n>|energy=<f>|beta=<f>|dim=<d>|n=<N>|json=0
    /// ```
    ///
    /// `steps` and `margin` are in the row on purpose: a metastable settle is then
    /// visible in the ledger instead of being hidden by the argmax snap.
    pub fn hbp_row(&self, r: &Recall) -> String {
        let mut hex = String::with_capacity(16);
        for b in r.id {
            hex.push_str(&format!("{b:02x}"));
        }
        format!(
            "HOPRECALL|id={hex}|idx={idx}|conf={conf:.6}|margin={margin:.6}|steps={steps}|energy={energy:.6}|beta={beta:.6}|dim={dim}|n={n}|json=0",
            idx = r.idx,
            conf = r.confidence,
            margin = r.margin,
            steps = r.steps,
            energy = r.energy,
            beta = r.beta,
            dim = self.dim,
            n = self.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// In-house SHA-256 (pure Rust, std-only, KAT-verified in tests).
// FIPS 180-4. Used only for the sha16 content address.
// ---------------------------------------------------------------------------
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

    /// SHA-256 digest of an arbitrary byte slice.
    pub fn digest(msg: &[u8]) -> [u8; 32] {
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let ml_bits = (msg.len() as u64).wrapping_mul(8);
        let mut data = msg.to_vec();
        data.push(0x80);
        while data.len() % 64 != 56 {
            data.push(0);
        }
        data.extend_from_slice(&ml_bits.to_be_bytes());

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
        for i in 0..8 {
            out[4 * i..4 * i + 4].copy_from_slice(&h[i].to_be_bytes());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests — these prove the MEASURED claims on-machine (not the theorem).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic LCG so the tests are reproducible with zero deps.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_f32(&mut self) -> f32 {
            // Numerical Recipes LCG constants.
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (self.0 >> 33) as u32; // top bits
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0 // ~uniform in [-1, 1)
        }
    }

    /// Build `n` unit-norm pseudo-random `dim`-vectors (near-orthogonal in high dim,
    /// i.e. well-separated — the regime the capacity theorem is proven for).
    fn make_patterns(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg::new(seed);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
            let norm = (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()).sqrt() as f32;
            for x in v.iter_mut() {
                *x /= norm;
            }
            out.push(v);
        }
        out
    }

    #[test]
    fn sha256_known_answer_vectors() {
        // Empty string.
        let e = sha256::digest(b"");
        let mut hex = String::new();
        for b in e {
            hex.push_str(&format!("{b:02x}"));
        }
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // "abc".
        let a = sha256::digest(b"abc");
        let mut hex2 = String::new();
        for b in a {
            hex2.push_str(&format!("{b:02x}"));
        }
        assert_eq!(
            hex2,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn store_dedup_and_get_roundtrip() {
        let dim = 60;
        let mut mem = HopfieldMemory::new(dim);
        let pats = make_patterns(8, dim, 0xA5A5);
        let mut ids = Vec::new();
        for p in &pats {
            ids.push(mem.store(p).unwrap());
        }
        assert_eq!(mem.len(), 8);
        // Storing the same pattern again is idempotent (exact-bytes dedup).
        let again = mem.store(&pats[3]).unwrap();
        assert_eq!(again, ids[3]);
        assert_eq!(mem.len(), 8, "dedup must not grow the store");
        // get() returns byte-identical bytes.
        assert_eq!(mem.get(&ids[3]).unwrap(), pats[3].as_slice());
        // dim mismatch and NaN are rejected at the boundary.
        assert!(matches!(
            mem.store(&vec![0.0; dim - 1]),
            Err(HopfieldError::DimMismatch { .. })
        ));
        let mut bad = pats[0].clone();
        bad[7] = f32::NAN;
        assert!(matches!(
            mem.store(&bad),
            Err(HopfieldError::NonFinite { index: 7 })
        ));
    }

    #[test]
    fn one_step_exact_recall_from_masked_cue() {
        // MEASURED claim, measured on-machine: a corrupted cue of a well-separated
        // pattern recalls the EXACT stored row (byte-identical) in a single step.
        let dim = 60;
        let mut mem = HopfieldMemory::new(dim);
        let pats = make_patterns(8, dim, 0x1234_5678);
        let ids: Vec<_> = pats.iter().map(|p| mem.store(p).unwrap()).collect();

        for target in 0..pats.len() {
            // Corrupt: zero out the second half of the cue (partial cue).
            let mut cue = pats[target].clone();
            for k in (dim / 2)..dim {
                cue[k] = 0.0;
            }
            let r = mem.recall(&cue, 20.0).unwrap();
            // snapped row is byte-identical to the stored target (Path-1).
            assert_eq!(r.idx, target, "argmax snapped to the wrong pattern");
            assert_eq!(r.id, ids[target]);
            assert_eq!(r.snapped, pats[target], "recall must be bit-identical");
            assert_eq!(r.steps, 1);
            // native confidence/margin fall out of the softmax.
            assert!(r.confidence > 0.0 && r.confidence <= 1.0);
            assert!(r.margin >= 0.0);
            // HBP hot-path row is well-formed and ends in |json=0.
            let row = mem.hbp_row(&r);
            assert!(row.starts_with("HOPRECALL|id="));
            assert!(row.ends_with("|json=0"), "row = {row}");
        }
    }

    #[test]
    fn iterated_recall_energy_is_monotone_nonincreasing() {
        // MEASURED safe-iteration guarantee, witnessed on-machine: energy never
        // increases across update steps (CCCP descent).
        let dim = 48;
        let mut mem = HopfieldMemory::new(dim);
        let pats = make_patterns(6, dim, 0xBEEF);
        for p in &pats {
            mem.store(p).unwrap();
        }
        // Start from a blurred blend of two patterns so iteration does real work.
        let mut cue = vec![0.0f32; dim];
        for k in 0..dim {
            cue[k] = 0.5 * pats[1][k] + 0.5 * pats[4][k];
        }
        let beta = mem.default_beta();

        // Manually walk the updates and record the energy trace.
        let mut xi = cue.clone();
        let mut energies = vec![mem.energy(&xi, beta)];
        for _ in 0..10 {
            let r = mem.recall(&xi, beta).unwrap(); // one step
            // advance along the mixture (soft update), not the hard snap
            xi = r.mixture.clone();
            energies.push(mem.energy(&xi, beta));
        }
        for w in energies.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-3,
                "energy increased across a step: {} -> {}",
                w[0],
                w[1]
            );
        }

        // The library's own iterated recall completes and snaps to a stored row.
        let r = mem.recall_iter(&cue, beta, 1e-5, 32).unwrap();
        assert!(r.steps >= 1);
        assert!(mem.get(&r.id).is_some());
        assert_eq!(r.snapped, mem.get(&r.id).unwrap());
    }

    #[test]
    fn masked_recall_is_design_but_functional() {
        // DESIGN path (no inherited paper bound): partial-cue recall with an explicit
        // mask still selects the right well-separated pattern.
        let dim = 60;
        let mut mem = HopfieldMemory::new(dim);
        let pats = make_patterns(5, dim, 0x0F0F);
        let ids: Vec<_> = pats.iter().map(|p| mem.store(p).unwrap()).collect();

        let target = 2;
        // Known = first 40 dims; the rest are garbage we tell the recaller to ignore.
        let mut mask = vec![true; dim];
        for k in 40..dim {
            mask[k] = false;
        }
        let mut cue = pats[target].clone();
        for k in 40..dim {
            cue[k] = 999.0; // poisoned, but masked out
        }
        let r = mem.recall_masked(&cue, &mask, 20.0, true).unwrap();
        assert_eq!(r.idx, target);
        assert_eq!(r.id, ids[target]);
        assert_eq!(r.snapped, pats[target]);
    }
}
