//! # predictive_coding — a std-only Rao & Ballard predictive-coding network
//!
//! Clean-room implementation (WHITE-ROOM discipline: built from the *mechanism*,
//! no source copied) of hierarchical predictive coding.
//!
//! ## The one genius (MEASURED — Rao & Ballard, *Nature Neuroscience* 1999)
//! Replace a single **global** backprop pass with a lattice of purely **local**
//! error-minimization loops. Two unit types make this work:
//!   * representation units `r[l]` (the causes / latents), and
//!   * error units `e[l-1] = r[l-1] - f(W[l] · r[l])` (what the layer below did
//!     that the top-down prediction failed to explain).
//!
//! The signal that travels *up the wire* is the **error**, never the raw
//! activity. That single choice yields the three properties that are the real
//! win, all of which this module implements faithfully:
//!
//! 1. **LOCALITY** — inference moves `r[l]` on exactly two adjacent error terms
//!    (the error of the layer it predicts, pulled up through `W[l]^T`, and its
//!    own error against the top-down prediction from above). Learning is a
//!    Hebbian outer product `dW = (error below) ⊗ (settled activity above)`.
//!    Nothing global travels more than one layer.
//! 2. **SETTLE-THEN-LEARN** — two clean phases. Freeze `W`, relax `r` to a fixed
//!    point (`infer`), then take **one** Hebbian `W` step (`learn`). Same energy
//!    descent at two timescales.
//! 3. **ONE ENERGY** — both `r` and `W` updates descend a single
//!    precision-weighted squared-error functional
//!    `E = Σ_l π[l] · ‖e[l]‖² + λ‖r_top‖² + α·Σ‖W‖²`.
//!    Falling `E` is the entire training signal — no separate optimizer.
//!
//! ## What is MEASURED vs DESIGN
//! * **MEASURED (from the paper):** the local error/representation update rules,
//!   settle-then-learn, the single energy functional, and per-level Gaussian
//!   *variance* (here promoted to per-layer precision `π`) — Rao & Ballard use
//!   exactly this to reproduce extra-classical receptive-field / end-stopping.
//! * **DESIGN (Asolaria simplifications / additions):** the *fixed-count
//!   deterministic settle* (→ bit-reproducible state), per-layer precision as a
//!   *first-class API knob*, and content-addressing via `state_sha16()` +
//!   `hbp_row()` on the HBP hot path. These are engineering choices, not claims
//!   about the brain.
//!
//! ## Honest boundaries (do NOT over-claim)
//! * The top latent `r_top` is a **lossy** code of `x`. `predict(infer(x))` is an
//!   *approximate* reconstruction whose error floors at the residual entropy of
//!   the source under the model. This is a learned **lossy generative model** —
//!   never lossless, never "code-rate-1.0", never compression below entropy.
//! * On a linear net this descends the *same quadratic energy* as PCA/least
//!   squares; it is **not** a general SGD-killer. The honest win is LOCALITY +
//!   biological plausibility + determinism/addressability, not accuracy.
//!
//! Zero external crates. `#![forbid(unsafe_code)]`-safe (no unsafe used).

// ===========================================================================
// Activation (nonlinearity behind an enum; Identity keeps the linear fast path)
// ===========================================================================

/// Element-wise nonlinearity `f` applied to a top-down prediction `W·r`.
///
/// `Identity` makes `f' = 1`, killing every `·f'` term so the math is exact and
/// the net descends a clean quadratic energy — this is the recommended default
/// and the regime in which the four test-gates below hold as equalities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// `f(x) = x`, `f'(x) = 1`. Linear fast path (default).
    Identity,
    /// `f(x) = tanh(x)`, `f'(x) = 1 - tanh(x)^2`.
    Tanh,
    /// `f(x) = 1/(1+e^-x)`, `f'(x) = f(1-f)`.
    Logistic,
}

impl Act {
    /// Apply the nonlinearity to a pre-activation value.
    #[inline]
    pub fn f(self, x: f64) -> f64 {
        match self {
            Act::Identity => x,
            Act::Tanh => x.tanh(),
            Act::Logistic => 1.0 / (1.0 + (-x).exp()),
        }
    }
    /// Derivative `f'` **as a function of the pre-activation** `x`.
    #[inline]
    pub fn df(self, x: f64) -> f64 {
        match self {
            Act::Identity => 1.0,
            Act::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Act::Logistic => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
        }
    }
    /// Stable one-byte tag for content-addressing.
    #[inline]
    fn tag(self) -> u8 {
        match self {
            Act::Identity => 0,
            Act::Tanh => 1,
            Act::Logistic => 2,
        }
    }
}

// ===========================================================================
// Config
// ===========================================================================

/// Hyper-parameters for a [`PcNet`]. Defaults are deliberately conservative so
/// the settle is stable and `E` is monotone non-increasing across `learn`.
#[derive(Clone, Copy, Debug)]
pub struct PcConfig {
    /// Fixed number of settle sweeps per `infer`. Fixed (not early-stopped) →
    /// bit-reproducible state → valid content address (DESIGN).
    pub infer_iters: usize,
    /// Representation learning rate (settle step on `r`).
    pub eta_r: f64,
    /// Weight learning rate (Hebbian step on `W`).
    pub eta_w: f64,
    /// Gaussian prior coefficient on the top latent (`λ‖r_top‖²`). Keeps the
    /// top layer from running away when there is no layer above to constrain it.
    pub lambda: f64,
    /// Weight-decay coefficient (`α‖W‖²`).
    pub alpha: f64,
    /// Sparsity coefficient on hidden/top activity, using the Cauchy log-prior
    /// `Σ log(1+r²)` (derivative `2r/(1+r²)`). Default `0.0` → inert. When > 0
    /// its exact gradient is added so `E` still descends.
    pub sparsity: f64,
    /// Top-down nonlinearity.
    pub act: Act,
}

impl Default for PcConfig {
    fn default() -> Self {
        PcConfig {
            infer_iters: 40,
            eta_r: 0.1,
            eta_w: 0.01,
            lambda: 1e-3,
            alpha: 1e-5,
            sparsity: 0.0,
            act: Act::Identity,
        }
    }
}

// ===========================================================================
// The network
// ===========================================================================

/// A hierarchical predictive-coding network.
///
/// Weights are stored as one flat `Vec<f64>` per layer with explicit
/// `(rows, cols) = (sizes[i], sizes[i+1])`, so `W·x` and `W^T·y` are the *same*
/// loop with swapped index order (no transpose buffer). `r[0]` is always the
/// clamped input; `r[L]` is the top latent.
#[derive(Clone, Debug)]
pub struct PcNet {
    /// Layer widths `[n_0, n_1, …, n_L]` (input first, top latent last).
    sizes: Vec<usize>,
    /// `weights[i]` predicts `r[i]` from `r[i+1]`; row-major, len `sizes[i]*sizes[i+1]`.
    weights: Vec<Vec<f64>>,
    /// `precisions[i]` weights the level-`i` error `e[i]`; len `= sizes.len()-1`.
    precisions: Vec<f64>,
    cfg: PcConfig,
}

impl PcNet {
    /// Build a net.
    ///
    /// * `sizes` — `[n_0 … n_L]`, at least two entries, all non-zero.
    /// * `precisions` — one per weight layer (`sizes.len()-1`).
    /// * `seed` — seeds the deterministic LCG init so the whole net is
    ///   reproducible and content-addressable.
    ///
    /// # Panics
    /// If `sizes.len() < 2`, any size is `0`, or
    /// `precisions.len() != sizes.len()-1`.
    pub fn new(sizes: Vec<usize>, precisions: Vec<f64>, cfg: PcConfig, seed: u64) -> Self {
        assert!(
            sizes.len() >= 2,
            "need at least an input and one latent layer"
        );
        assert!(sizes.iter().all(|&s| s > 0), "layer sizes must be non-zero");
        assert_eq!(
            precisions.len(),
            sizes.len() - 1,
            "one precision per weight layer"
        );
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15; // splittable-ish seed mix
        let mut weights = Vec::with_capacity(sizes.len() - 1);
        for i in 0..sizes.len() - 1 {
            let rows = sizes[i];
            let cols = sizes[i + 1];
            // He-ish scale: (u-0.5)*sqrt(1/fan_in), fan_in = cols (input to matvec).
            let scale = (1.0 / cols as f64).sqrt();
            let mut w = Vec::with_capacity(rows * cols);
            for _ in 0..rows * cols {
                let u = next_unit(&mut state);
                w.push((u - 0.5) * scale);
            }
            weights.push(w);
        }
        PcNet {
            sizes,
            weights,
            precisions,
            cfg,
        }
    }

    /// Number of weight layers `L` (`= sizes.len()-1`).
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.sizes.len() - 1
    }

    /// Layer widths `[n_0 … n_L]`.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    // --- linear algebra (flat W, shared by forward and transpose) -----------

    /// `out = W · x` where `W` is `rows×cols` row-major, `x` len `cols`.
    fn matvec(w: &[f64], rows: usize, cols: usize, x: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; rows];
        for r in 0..rows {
            let base = r * cols;
            let mut acc = 0.0;
            for c in 0..cols {
                acc += w[base + c] * x[c];
            }
            out[r] = acc;
        }
        out
    }

    /// `out = W^T · y` reusing the SAME flat `W` (rows×cols), `y` len `rows`,
    /// result len `cols`. Same memory, swapped index order — no transpose copy.
    fn matvec_t(w: &[f64], rows: usize, cols: usize, y: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; cols];
        for r in 0..rows {
            let base = r * cols;
            let yr = y[r];
            for c in 0..cols {
                out[c] += w[base + c] * yr;
            }
        }
        out
    }

    // --- the settle (inference) ---------------------------------------------

    /// Settle the representation stack for input `x` with weights **frozen**.
    ///
    /// Returns the settled `r` stack (`r[0]` clamped to `x`, `r[L]` the top
    /// latent). Pure w.r.t. `self` (weights unchanged). Deterministic: a fixed
    /// `infer_iters` sweeps in a fixed reduction order → bit-reproducible.
    pub fn infer(&self, x: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(x.len(), self.sizes[0], "input width mismatch");
        let l = self.num_layers();
        // r[0] = x (clamped). Hidden/top start at zero (deterministic).
        let mut r: Vec<Vec<f64>> = self.sizes.iter().map(|&n| vec![0.0; n]).collect();
        r[0].copy_from_slice(x);

        for _ in 0..self.cfg.infer_iters {
            // ---- top-down sweep: pre-activations p[i] and errors e[i] -------
            let mut pre = Vec::with_capacity(l); // p[i] = W[i]·r[i+1]
            let mut err = Vec::with_capacity(l); // e[i] = r[i] - f(p[i])
            for i in 0..l {
                let rows = self.sizes[i];
                let cols = self.sizes[i + 1];
                let p = Self::matvec(&self.weights[i], rows, cols, &r[i + 1]);
                let mut e = vec![0.0; rows];
                for k in 0..rows {
                    e[k] = r[i][k] - self.cfg.act.f(p[k]);
                }
                pre.push(p);
                err.push(e);
            }

            // ---- bottom-up sweep: update each hidden/top r[m] (r[0] pinned) --
            for m in 1..=l {
                let below = m - 1; // error level this layer predicts
                let rows = self.sizes[below];
                let cols = self.sizes[m];
                // modulated error: (f'(p) .* e) for the pull-up transpose term
                let mut me = vec![0.0; rows];
                for k in 0..rows {
                    me[k] = self.cfg.act.df(pre[below][k]) * err[below][k];
                }
                let pull = Self::matvec_t(&self.weights[below], rows, cols, &me);
                let pi_below = self.precisions[below];
                for j in 0..cols {
                    let mut g = pi_below * pull[j];
                    if m < l {
                        // top-down error from the layer above (level m exists)
                        g -= self.precisions[m] * err[m][j];
                    } else {
                        // top layer: Gaussian prior pull instead of top-down error
                        g -= self.cfg.lambda * r[m][j];
                    }
                    if self.cfg.sparsity != 0.0 {
                        let rj = r[m][j];
                        g -= self.cfg.sparsity * (2.0 * rj / (1.0 + rj * rj));
                    }
                    r[m][j] += self.cfg.eta_r * g;
                }
            }
        }
        r
    }

    /// Convenience: the settled **top latent** `r[L]` for input `x`.
    pub fn latent(&self, x: &[f64]) -> Vec<f64> {
        let r = self.infer(x);
        r.into_iter().last().expect("non-empty stack")
    }

    /// Top-down prediction of the **input** from a settled stack:
    /// `f(W[0] · r[1])`. This is the model's reconstruction of `x` given the
    /// settled first hidden layer. (LOSSY — see module docs.)
    pub fn predict(&self, stack: &[Vec<f64>]) -> Vec<f64> {
        let rows = self.sizes[0];
        let cols = self.sizes[1];
        let p = Self::matvec(&self.weights[0], rows, cols, &stack[1]);
        p.iter().map(|&v| self.cfg.act.f(v)).collect()
    }

    /// Settle then reconstruct: `predict(infer(x))`. Approximate, not exact.
    pub fn reconstruct(&self, x: &[f64]) -> Vec<f64> {
        let r = self.infer(x);
        self.predict(&r)
    }

    // --- learning ------------------------------------------------------------

    /// One **settle-then-learn** step on a single sample `x`.
    ///
    /// Settles `r` (weights frozen), then takes exactly one local Hebbian step
    /// per layer:
    /// `W[i] += η_W · ( π[i] · (f'(p[i]).*e[i]) ⊗ r[i+1] − α·W[i] )`,
    /// and returns the scalar energy `E` computed from the settled state.
    /// **Falling `E` across calls is the entire learning signal.**
    pub fn learn(&mut self, x: &[f64]) -> f64 {
        // PHASE 1: settle with frozen weights (must come first — a W step on
        // un-settled r would use stale errors and corrupt the local gradient).
        let r = self.infer(x);
        let l = self.num_layers();

        // Recompute pre-activations & errors from the settled state.
        let mut pre = Vec::with_capacity(l);
        let mut err = Vec::with_capacity(l);
        for i in 0..l {
            let rows = self.sizes[i];
            let cols = self.sizes[i + 1];
            let p = Self::matvec(&self.weights[i], rows, cols, &r[i + 1]);
            let mut e = vec![0.0; rows];
            for k in 0..rows {
                e[k] = r[i][k] - self.cfg.act.f(p[k]);
            }
            pre.push(p);
            err.push(e);
        }

        // PHASE 2: one Hebbian outer-product step per weight layer.
        for i in 0..l {
            let rows = self.sizes[i];
            let cols = self.sizes[i + 1];
            let pi = self.precisions[i];
            let eta = self.cfg.eta_w;
            let alpha = self.cfg.alpha;
            let w = &mut self.weights[i];
            for rr in 0..rows {
                let me = self.cfg.act.df(pre[i][rr]) * err[i][rr];
                let base = rr * cols;
                for cc in 0..cols {
                    let g = pi * me * r[i + 1][cc] - alpha * w[base + cc];
                    w[base + cc] += eta * g;
                }
            }
        }

        // Energy from the settled state (errors above were pre-W-step; recompute
        // after the step so the returned E reflects the updated weights).
        self.energy(&r)
    }

    /// Precision-weighted total energy for a given settled stack under the
    /// **current** weights:
    /// `E = Σ_i π[i]‖e[i]‖² + λ‖r[L]‖² + s·Σ log(1+r²) + α·Σ‖W‖²`.
    pub fn energy(&self, stack: &[Vec<f64>]) -> f64 {
        let l = self.num_layers();
        let mut e_total = 0.0;
        for i in 0..l {
            let rows = self.sizes[i];
            let cols = self.sizes[i + 1];
            let p = Self::matvec(&self.weights[i], rows, cols, &stack[i + 1]);
            let mut ss = 0.0;
            for k in 0..rows {
                let e = stack[i][k] - self.cfg.act.f(p[k]);
                ss += e * e;
            }
            e_total += self.precisions[i] * ss;
        }
        // Gaussian prior on the top latent.
        let top = &stack[l];
        e_total += self.cfg.lambda * top.iter().map(|&v| v * v).sum::<f64>();
        // Optional sparsity prior on all hidden/top layers.
        if self.cfg.sparsity != 0.0 {
            let mut s = 0.0;
            for m in 1..=l {
                for &v in &stack[m] {
                    s += (1.0 + v * v).ln();
                }
            }
            e_total += self.cfg.sparsity * s;
        }
        // Weight decay.
        if self.cfg.alpha != 0.0 {
            let mut wn = 0.0;
            for w in &self.weights {
                for &v in w {
                    wn += v * v;
                }
            }
            e_total += self.cfg.alpha * wn;
        }
        e_total
    }

    // --- Asolaria hot-path: content-addressing + HBP row --------------------

    /// Content address of `(weights + precisions + sizes + act-tag)`: pure-std
    /// SHA-256 over a canonical little-endian byte serialization, truncated to
    /// 16 hex chars. Stable across runs with the same seed (DESIGN: this is what
    /// makes the settled net slot into recall / HBI addressing).
    pub fn state_sha16(&self) -> String {
        let mut buf: Vec<u8> = Vec::new();
        for &s in &self.sizes {
            buf.extend_from_slice(&(s as u64).to_le_bytes());
        }
        buf.push(self.cfg.act.tag());
        for &p in &self.precisions {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        for w in &self.weights {
            for &v in w {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        let digest = sha256(&buf);
        let mut hex = String::with_capacity(16);
        for b in &digest[..8] {
            hex.push_str(&format!("{:02x}", b));
        }
        hex
    }

    /// Emit the HBP hot-path tuple row for a learn/infer boundary. Ends with
    /// `|json=0` (JSON stays cold/debug only).
    ///
    /// Grammar: `PC|layers=<L>|iters=<n>|energy=<E>|sha16=<h>|json=0`.
    pub fn hbp_row(&self, energy: f64) -> String {
        format!(
            "PC|layers={}|iters={}|energy={:.6}|sha16={}|json=0",
            self.num_layers(),
            self.cfg.infer_iters,
            energy,
            self.state_sha16()
        )
    }
}

// ===========================================================================
// Deterministic seeded RNG (LCG) — zero-dep, reproducible → content-addressable
// ===========================================================================

/// Advance an LCG and return the next uniform in `[0,1)` using the high bits
/// (low bits of an LCG have short periods; the high bits are well-distributed).
#[inline]
fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // take the top 31 bits → [0, 2^31) → [0,1)
    let hi = (*state >> 33) as u32;
    hi as f64 / (1u64 << 31) as f64
}

// ===========================================================================
// SHA-256 (FIPS 180-4) — pure std, no crate. Used only for content-addressing.
// ===========================================================================

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 over a byte slice → 32-byte digest. Standard FIPS 180-4.
fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bitlen = (msg.len() as u64).wrapping_mul(8);
    let mut data = msg.to_vec();
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in data.chunks(64) {
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
                .wrapping_add(SHA_K[i])
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

// ===========================================================================
// Tests — the four honest, checkable claims (+ SHA KAT + HBP format).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic low-rank Gaussian-ish source: x = B·z, B is n×k random,
    /// z is a k-dim small code. Built from the same LCG so tests are reproducible.
    fn low_rank_dataset(n: usize, k: usize, count: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut st = seed;
        // basis B (n×k)
        let mut b = vec![0.0f64; n * k];
        for v in b.iter_mut() {
            *v = next_unit(&mut st) - 0.5;
        }
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            let z: Vec<f64> = (0..k).map(|_| next_unit(&mut st) - 0.5).collect();
            let mut x = vec![0.0; n];
            for r in 0..n {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += b[r * k + c] * z[c];
                }
                x[r] = acc;
            }
            data.push(x);
        }
        data
    }

    fn mean_recon_err(net: &PcNet, data: &[Vec<f64>]) -> f64 {
        let mut total = 0.0;
        for x in data {
            let xh = net.reconstruct(x);
            let mut ss = 0.0;
            for k in 0..x.len() {
                let d = x[k] - xh[k];
                ss += d * d;
            }
            total += ss;
        }
        total / data.len() as f64
    }

    /// GATE 1 (MEASURED mechanism): energy is monotonically non-increasing over
    /// `learn()` calls. A non-falling E would mean a wrong transpose, a phase
    /// bug, or a diverging settle — none of which we ship as "trained".
    #[test]
    fn energy_is_monotone_nonincreasing() {
        let data = low_rank_dataset(8, 3, 12, 0xC0FFEE);
        let mut net = PcNet::new(vec![8, 4], vec![1.0], PcConfig::default(), 42);
        let mut prev = f64::INFINITY;
        for epoch in 0..60 {
            let mut e_epoch = 0.0;
            for x in &data {
                e_epoch = net.learn(x); // last sample's settled E
            }
            // Compare epoch-to-epoch on a fixed probe sample for a clean signal.
            let probe = net.energy(&net.infer(&data[0]));
            assert!(
                probe <= prev + 1e-6,
                "energy rose at epoch {}: {} > {}",
                epoch,
                probe,
                prev
            );
            prev = probe;
            let _ = e_epoch;
        }
    }

    /// GATE 2 (LOCALITY / purity): `infer` does not mutate weights and is a pure
    /// function of `(x, weights)` — two calls give identical stacks and the
    /// content address is unchanged.
    #[test]
    fn infer_is_pure_and_deterministic() {
        let net = PcNet::new(vec![6, 3], vec![1.0], PcConfig::default(), 7);
        let before = net.state_sha16();
        let x = vec![0.3, -0.1, 0.7, 0.0, -0.4, 0.2];
        let a = net.infer(&x);
        let b = net.infer(&x);
        let after = net.state_sha16();
        assert_eq!(before, after, "infer must not mutate weights");
        assert_eq!(a.len(), b.len());
        for (la, lb) in a.iter().zip(b.iter()) {
            assert_eq!(la, lb, "infer must be deterministic");
        }
    }

    /// GATE 3 (the honest win, stated LOSSILY): reconstruction improves after
    /// training on a low-rank source. Not lossless — just lower residual.
    #[test]
    fn reconstruction_improves_after_training() {
        let data = low_rank_dataset(8, 3, 16, 0xABCD1234);
        let mut net = PcNet::new(vec![8, 3], vec![1.0], PcConfig::default(), 99);
        let before = mean_recon_err(&net, &data);
        for _ in 0..300 {
            for x in &data {
                net.learn(x);
            }
        }
        let after = mean_recon_err(&net, &data);
        assert!(
            after < before,
            "training did not reduce reconstruction error: before={} after={}",
            before,
            after
        );
    }

    /// GATE 4 (DESIGN, content-addressing): same seed → identical `state_sha16`,
    /// and identical training keeps them identical. If this fails the
    /// "content-addressable" claim is false.
    #[test]
    fn state_sha16_is_reproducible() {
        let n1 = PcNet::new(vec![5, 4, 2], vec![1.0, 1.0], PcConfig::default(), 2024);
        let n2 = PcNet::new(vec![5, 4, 2], vec![1.0, 1.0], PcConfig::default(), 2024);
        assert_eq!(n1.state_sha16(), n2.state_sha16(), "seeded init must match");

        let data = low_rank_dataset(5, 2, 8, 555);
        let mut a = n1;
        let mut b = n2;
        for _ in 0..20 {
            for x in &data {
                a.learn(x);
                b.learn(x);
            }
        }
        assert_eq!(
            a.state_sha16(),
            b.state_sha16(),
            "identical training must yield identical content address"
        );
        assert_eq!(a.state_sha16().len(), 16, "address is 16 hex chars");
    }

    /// SHA-256 known-answer test: sha256("abc") starts with ba7816bf8f01cfea.
    /// Guards the content-addressing primitive itself.
    #[test]
    fn sha256_known_answer() {
        let d = sha256(b"abc");
        let mut hex = String::new();
        for byte in &d[..8] {
            hex.push_str(&format!("{:02x}", byte));
        }
        assert_eq!(hex, "ba7816bf8f01cfea");
        // empty string KAT prefix
        let e = sha256(b"");
        assert_eq!(e[0], 0xe3);
        assert_eq!(e[1], 0xb0);
    }

    /// HBP hot-path row: correct grammar, ends with `|json=0`, no JSON braces.
    #[test]
    fn hbp_row_is_hotpath() {
        let net = PcNet::new(vec![4, 2], vec![1.0], PcConfig::default(), 1);
        let e = net.energy(&net.infer(&[0.1, 0.2, 0.3, 0.4]));
        let row = net.hbp_row(e);
        assert!(row.starts_with("PC|layers=1|"), "row: {}", row);
        assert!(row.ends_with("|json=0"), "row: {}", row);
        assert!(row.contains("|sha16="), "row: {}", row);
        assert!(
            !row.contains('{') && !row.contains('}'),
            "no JSON on hot path"
        );
    }

    /// Nonlinear regime opt-in still runs and the energy is finite (smoke test;
    /// no accuracy claim — that would be UNVERIFIED without a benchmark).
    #[test]
    fn nonlinear_act_runs() {
        let mut cfg = PcConfig::default();
        cfg.act = Act::Tanh;
        cfg.eta_r = 0.05;
        let data = low_rank_dataset(6, 2, 8, 321);
        let mut net = PcNet::new(vec![6, 3], vec![1.0], cfg, 5);
        let mut e = f64::NAN;
        for _ in 0..30 {
            for x in &data {
                e = net.learn(x);
            }
        }
        assert!(e.is_finite(), "nonlinear energy must stay finite: {}", e);
    }
}
