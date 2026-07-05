//! # memory_os_paging — a deterministic, content-addressed memory pager
//!
//! Clean-room re-engineering of the *mechanism* in MemGPT (Packer et al.,
//! arXiv:2310.08560), NOT its code. Studied the paper, kept the smallest core,
//! rebuilt it stronger in Rust `std`-only (zero external crates).
//!
//! ## The essential genius (MEASURED — from the paper)
//! MemGPT turns a hard context-window *ceiling* into a *latency hierarchy* by
//! treating an LLM's fixed context like RAM and an external store like disk:
//!   1. **EVICT = MOVE, NEVER DROP.** A bounded hot set (RAM-analog) spills its
//!      oldest entries to an unbounded external store (disk-analog), so the full
//!      entropy `H(X)` is *always retained somewhere*; the budget only bounds the
//!      HOT slice.
//!   2. **LEAVE A LOG-SIZE STAND-IN.** Each flush leaves an `O(1)` digest behind
//!      so the hot set never loses the *ability to find* what it demoted — only
//!      the bytes themselves.
//!   3. **DEMAND PAGE-IN IS A FIRST-CLASS OP.** Recall re-admits cold bytes at
//!      the queue tail with a fresh arrival seq, which may itself evict other
//!      cold items — exactly like OS demand paging. The controller decides when
//!      to page, not the storage layer.
//!   4. **HYSTERESIS.** Evict from `tau_hi` down to `tau_lo` (defaults 0.9/0.7):
//!      each flush frees `>= (tau_hi - tau_lo) * B` bytes, giving amortized
//!      `O(1)` evictions per insert and killing thrash.
//!
//! ## What we built BETTER (DESIGN — Asolaria re-engineering)
//! * **No LLM in the loop.** MemGPT's warn-then-flush-with-LLM-summary is
//!   replaced by pure code: hysteresis watermarks + a digest that is a
//!   *content-addressed pointer-of-pointers* whose BYTES are the concatenated
//!   evicted `sha16` ids (digest id = `sha16` of that concatenation, chained to
//!   the previous digest). Cheaper, reproducible, `E=0`, testable with no model.
//! * **std-only substrate.** `VecDeque` + `HashMap` + a running usage counter,
//!   plus an in-repo SHA-256 verified against FIPS 180-4 KAT vectors.
//! * **One store, not two.** Content addressing collapses MemGPT's recall-vs-
//!   archival split into a single `sha16`-cube map; dedup falls out for free.
//! * **HBP journal is the paging log.** Every op emits a hot-path tuple row
//!   `MEMPAGE|op=..|..|json=0`, drained by [`MemoryPager::hbp_rows`]. JSON is
//!   cold debug only, never on the hot path.
//! * **Safe page-in.** `get` re-hashes every cube on load and rejects a mismatch
//!   (recover-or-Hold, never serve wrong bytes silently).
//!
//! ## Honest boundaries (Shannon)
//! This is **NOT compression** and **NOT free context extension**. The hot set
//! holds `<= B` bytes; full entropy lives in the external store and recall costs
//! a read. The digest is an `O(1)` POINTER — `H(X | digest) > 0`. "Lossless"
//! applies only to the *system as a whole* (evict = move), and only up to the
//! stated fsync boundary. `sha16` (64-bit) is an ADDRESS, not a MAC: accidental
//! collision is negligible below ~1e8 items but adversarially findable — hence
//! mandatory re-hash-on-load. MemGPT's benchmark numbers are MEASURED for an
//! LLM-in-the-loop system; this deterministic pager is DESIGN until its own
//! tests (below) run.

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

// ===========================================================================
// SHA-256 (pure Rust, in-repo, KAT-verified) — the addressing substrate.
// ===========================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Compute the SHA-256 digest of `data`. Pure `std`, no external crates.
/// Verified against FIPS 180-4 known-answer vectors in the test module.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Padding: append 0x80, zero-pad to 56 mod 64, then 64-bit big-endian bitlen.
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    for chunk in msg.chunks(64) {
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
                .wrapping_add(SHA256_K[i])
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

/// A content address: the first 8 bytes (64 bits) of SHA-256, rendered as 16
/// lowercase hex on the wire. An ADDRESS, not a MAC — see the module-level
/// Shannon note. Always re-hash on load and reject mismatches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Sha16([u8; 8]);

impl Sha16 {
    /// Content-address arbitrary bytes.
    pub fn of(bytes: &[u8]) -> Sha16 {
        let full = sha256(bytes);
        let mut id = [0u8; 8];
        id.copy_from_slice(&full[..8]);
        Sha16(id)
    }

    /// The raw 8 bytes (used when concatenating ids into a digest cube).
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// 16 lowercase hex characters.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(16);
        for b in &self.0 {
            s.push(hex_nibble(b >> 4));
            s.push(hex_nibble(b & 0x0f));
        }
        s
    }

    /// Parse 16 lowercase/uppercase hex characters back into a [`Sha16`].
    pub fn from_hex(s: &str) -> Option<Sha16> {
        if s.len() != 16 {
            return None;
        }
        let bytes = s.as_bytes();
        let mut out = [0u8; 8];
        for i in 0..8 {
            let hi = hex_val(bytes[2 * i])?;
            let lo = hex_val(bytes[2 * i + 1])?;
            out[i] = (hi << 4) | lo;
        }
        Some(Sha16(out))
    }
}

fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        _ => (b'a' + (n - 10)) as char,
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

// ===========================================================================
// HBP hot-path row (tuple, ends |json=0). JSON is cold debug only.
// ===========================================================================

/// Build one HBP hot-path tuple row: `MEMPAGE|op=<op>|k=v|...|json=0`.
/// This is the audit + replay grammar; the JSON lane never touches it.
pub fn hbp_row(op: &str, fields: &[(&str, &str)]) -> String {
    let mut s = String::from("MEMPAGE|op=");
    s.push_str(op);
    for (k, v) in fields {
        s.push('|');
        s.push_str(k);
        s.push('=');
        s.push_str(v);
    }
    s.push_str("|json=0");
    s
}

// ===========================================================================
// The pager.
// ===========================================================================

/// Result of [`MemoryPager::insert`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InsertOutcome {
    /// Newly admitted to the hot set (bytes retained hot; usage charged).
    Stored(Sha16),
    /// Identical bytes were already known (hot or cold); usage NOT re-charged.
    Deduped(Sha16),
    /// Too large to ever be hot-eligible; retained cold-only with a HOLD row.
    Held(Sha16),
}

impl InsertOutcome {
    /// The content address, regardless of variant.
    pub fn id(&self) -> Sha16 {
        match self {
            InsertOutcome::Stored(id) | InsertOutcome::Deduped(id) | InsertOutcome::Held(id) => *id,
        }
    }
}

/// Result of [`MemoryPager::get`] (demand recall).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Recall {
    /// Served from the hot set (FIFO: a hot hit does NOT refresh arrival seq).
    Hit(Vec<u8>),
    /// Demand-paged in from the cold store (re-admitted, or passed through if
    /// oversize). Bytes were re-hash-verified before return.
    PagedIn(Vec<u8>),
    /// No such id in hot or cold.
    NotFound,
    /// Found cold but the re-hash did not match the id — recover-or-Hold; bytes
    /// are refused rather than served wrong.
    Corrupt,
}

#[derive(Clone, Copy)]
struct HotEntry {
    id: Sha16,
    seq: u64,
    bytes: usize,
}

#[derive(Clone)]
struct Digest {
    id: Sha16,
    prev: Option<Sha16>,
    covers: Vec<Sha16>,
}

/// A deterministic, content-addressed memory pager.
///
/// Hot set = a FIFO-by-arrival working set bounded to `budget` bytes. Cold store
/// = an unbounded content-addressed `sha16` -> bytes map (the "disk"). Eviction
/// moves the oldest hot bytes to cold and leaves a chained digest pointer.
pub struct MemoryPager {
    budget: usize,
    tau_hi: f64,
    tau_lo: f64,

    hot: VecDeque<HotEntry>,       // arrival order (front = oldest)
    hot_bytes: HashMap<Sha16, Vec<u8>>,
    store: HashMap<Sha16, Vec<u8>>, // cold, unbounded, retained (also holds pinned digest cubes)
    usage: usize,                   // bytes currently in the hot set

    next_seq: u64,
    max_evict_per_op: usize,

    // Digest chain (pinned metadata — NOT counted against budget, never evicted).
    digests: Vec<Digest>,
    digest_head: Option<Sha16>,

    // Bounded HBP journal (GC discipline for long-running lanes).
    journal: VecDeque<String>,
    journal_cap: usize,
    journal_dropped: u64,
}

impl MemoryPager {
    /// New pager with default hysteresis watermarks (0.9 high / 0.7 low).
    /// `budget` is the hot-set byte ceiling; it is clamped to at least 1.
    pub fn new(budget: usize) -> MemoryPager {
        MemoryPager::with_watermarks(budget, 0.9, 0.7)
    }

    /// New pager with explicit watermarks. Requires `0 < tau_lo < tau_hi <= 1`.
    /// Invalid watermarks fall back to the 0.9/0.7 defaults.
    pub fn with_watermarks(budget: usize, tau_hi: f64, tau_lo: f64) -> MemoryPager {
        let (hi, lo) = if tau_lo > 0.0 && tau_lo < tau_hi && tau_hi <= 1.0 {
            (tau_hi, tau_lo)
        } else {
            (0.9, 0.7)
        };
        MemoryPager {
            budget: budget.max(1),
            tau_hi: hi,
            tau_lo: lo,
            hot: VecDeque::new(),
            hot_bytes: HashMap::new(),
            store: HashMap::new(),
            usage: 0,
            next_seq: 0,
            max_evict_per_op: 1024,
            digests: Vec::new(),
            digest_head: None,
            journal: VecDeque::new(),
            journal_cap: 4096,
            journal_dropped: 0,
        }
    }

    // ---- watermark helpers ----
    #[inline]
    fn hi_bytes(&self) -> f64 {
        self.tau_hi * self.budget as f64
    }
    #[inline]
    fn lo_bytes(&self) -> f64 {
        self.tau_lo * self.budget as f64
    }
    /// An item is hot-eligible iff it can fit under the low watermark; this is
    /// what guarantees a flush can always drain to `tau_lo` without livelock.
    #[inline]
    fn hot_eligible(&self, len: usize) -> bool {
        len <= self.budget && (len as f64) <= self.lo_bytes()
    }

    /// Insert bytes. Returns the content address and how it was handled.
    ///
    /// * New hot-eligible bytes are admitted at the tail (fresh arrival seq),
    ///   then a flush runs if the high watermark is crossed.
    /// * Identical bytes already known (hot OR cold) are [`InsertOutcome::Deduped`]
    ///   and NOT double-charged (content addressing gives dedup for free).
    /// * Oversize bytes are retained cold-only and reported [`InsertOutcome::Held`].
    pub fn insert(&mut self, bytes: &[u8]) -> InsertOutcome {
        let id = Sha16::of(bytes);
        let hex = id.to_hex();
        let blen = bytes.len().to_string();

        if self.hot_bytes.contains_key(&id) {
            self.emit("insert", &[("sha16", &hex), ("bytes", &blen), ("dedup", "1"), ("loc", "hot")]);
            return InsertOutcome::Deduped(id);
        }
        if self.store.contains_key(&id) {
            self.emit("insert", &[("sha16", &hex), ("bytes", &blen), ("dedup", "1"), ("loc", "cold")]);
            return InsertOutcome::Deduped(id);
        }

        if !self.hot_eligible(bytes.len()) {
            // Oversize: retain full entropy cold, but never admit to hot (avoids
            // a flush that could never reach tau_lo). Reads are pass-through.
            self.store.insert(id, bytes.to_vec());
            self.emit("hold", &[("reason", "oversize"), ("sha16", &hex), ("bytes", &blen)]);
            return InsertOutcome::Held(id);
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        self.hot.push_back(HotEntry { id, seq, bytes: bytes.len() });
        self.hot_bytes.insert(id, bytes.to_vec());
        self.usage += bytes.len();
        let seqs = seq.to_string();
        self.emit("insert", &[("sha16", &hex), ("bytes", &blen), ("seq", &seqs)]);

        self.maybe_flush(Some(id));
        InsertOutcome::Stored(id)
    }

    /// Demand-recall by content address.
    ///
    /// Hot hit -> [`Recall::Hit`] (FIFO: seq NOT refreshed). Cold hit ->
    /// re-hash-verify; on mismatch [`Recall::Corrupt`] (recover-or-Hold), else
    /// re-admit at the tail with a fresh seq (may cascade-evict others, but the
    /// just-admitted item is exempt from its own eviction) and return
    /// [`Recall::PagedIn`]. Oversize cold items are passed through without
    /// admission. Unknown id -> [`Recall::NotFound`].
    pub fn get(&mut self, id: &Sha16) -> Recall {
        let hex = id.to_hex();

        if let Some(b) = self.hot_bytes.get(id) {
            let out = b.clone();
            self.emit("get", &[("sha16", &hex), ("hit", "hot")]);
            return Recall::Hit(out);
        }

        let cold = match self.store.get(id) {
            Some(b) => b.clone(),
            None => {
                self.emit("get", &[("sha16", &hex), ("hit", "miss")]);
                return Recall::NotFound;
            }
        };

        // Recover-or-Hold: never serve wrong bytes silently.
        let recomputed = Sha16::of(&cold);
        if recomputed != *id {
            let got = recomputed.to_hex();
            self.emit("hold", &[("reason", "hash_mismatch"), ("sha16", &hex), ("got", &got)]);
            return Recall::Corrupt;
        }

        if !self.hot_eligible(cold.len()) {
            let clen = cold.len().to_string();
            self.emit("recall", &[("sha16", &hex), ("mode", "passthrough"), ("bytes", &clen)]);
            return Recall::PagedIn(cold);
        }

        let seq = self.next_seq;
        self.next_seq += 1;
        self.hot.push_back(HotEntry { id: *id, seq, bytes: cold.len() });
        self.hot_bytes.insert(*id, cold.clone());
        self.usage += cold.len();
        let seqs = seq.to_string();
        self.emit("recall", &[("sha16", &hex), ("mode", "pagein"), ("seq", &seqs)]);

        self.maybe_flush(Some(*id));
        Recall::PagedIn(cold)
    }

    /// Flush oldest hot entries to cold until usage <= low watermark, if the
    /// high watermark is exceeded. `exempt` (the just-touched id) is never its
    /// own victim. Each flush that evicts anything leaves one chained digest.
    fn maybe_flush(&mut self, exempt: Option<Sha16>) {
        if (self.usage as f64) <= self.hi_bytes() {
            return;
        }
        let target = self.lo_bytes();
        let mut evicted: Vec<Sha16> = Vec::new();
        let mut count = 0usize;

        while (self.usage as f64) > target && count < self.max_evict_per_op {
            let front_id = match self.hot.front() {
                Some(e) => e.id,
                None => break,
            };
            if Some(front_id) == exempt {
                // Oldest remaining is the just-admitted item; stop (it is the
                // only thing left and, being hot-eligible, is already <= target).
                break;
            }
            let entry = self.hot.pop_front().expect("front checked above");
            // Write-before-pop discipline (see `write_cube_atomic`): in a
            // fs-backed store the cube is durably renamed into place BEFORE this
            // in-memory move, so evict is a move, never a drop.
            if let Some(b) = self.hot_bytes.remove(&entry.id) {
                self.usage -= entry.bytes;
                self.store.insert(entry.id, b);
            }
            evicted.push(entry.id);
            let ehex = entry.id.to_hex();
            let ebytes = entry.bytes.to_string();
            let eseq = entry.seq.to_string();
            self.emit("evict", &[("sha16", &ehex), ("bytes", &ebytes), ("seq", &eseq)]);
            count += 1;
        }

        if !evicted.is_empty() {
            self.make_digest(&evicted);
        }
    }

    /// Build a content-addressed pointer-of-pointers over the evicted ids,
    /// chained to the previous digest. Pinned in the store; never evicted.
    fn make_digest(&mut self, evicted: &[Sha16]) {
        let mut content: Vec<u8> = Vec::new();
        if let Some(prev) = self.digest_head {
            content.extend_from_slice(prev.as_bytes());
        }
        for id in evicted {
            content.extend_from_slice(id.as_bytes());
        }
        let did = Sha16::of(&content);
        // Pin the digest cube (content-addressed, retained, not budget-charged).
        self.store.insert(did, content);
        self.digests.push(Digest {
            id: did,
            prev: self.digest_head,
            covers: evicted.to_vec(),
        });
        let prev_hex = self
            .digest_head
            .map(|p| p.to_hex())
            .unwrap_or_else(|| "none".to_string());
        self.digest_head = Some(did);

        let dhex = did.to_hex();
        let covers = evicted.len().to_string();
        self.emit("digest", &[("sha16", &dhex), ("covers", &covers), ("prev", &prev_hex)]);
    }

    /// Re-derive every digest from its recorded `prev + covers` and confirm the
    /// id matches — proves the paging history is tamper-evident (ReceiptChain).
    pub fn verify_digest_chain(&self) -> bool {
        let mut expected_prev: Option<Sha16> = None;
        for d in &self.digests {
            if d.prev != expected_prev {
                return false;
            }
            let mut content: Vec<u8> = Vec::new();
            if let Some(prev) = d.prev {
                content.extend_from_slice(prev.as_bytes());
            }
            for id in &d.covers {
                content.extend_from_slice(id.as_bytes());
            }
            if Sha16::of(&content) != d.id {
                return false;
            }
            expected_prev = Some(d.id);
        }
        expected_prev == self.digest_head
    }

    // ---- observation ----

    /// True if the id is known in hot OR cold.
    pub fn contains(&self, id: &Sha16) -> bool {
        self.hot_bytes.contains_key(id) || self.store.contains_key(id)
    }
    /// True if the id is currently resident in the hot set.
    pub fn is_hot(&self, id: &Sha16) -> bool {
        self.hot_bytes.contains_key(id)
    }
    /// Current hot-set byte usage.
    pub fn hot_usage(&self) -> usize {
        self.usage
    }
    /// Number of resident hot entries.
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }
    /// Number of distinct ids in the cold store (includes pinned digest cubes).
    pub fn cold_len(&self) -> usize {
        self.store.len()
    }
    /// The configured hot-set byte budget.
    pub fn budget(&self) -> usize {
        self.budget
    }
    /// Head of the digest chain, if any flush has occurred.
    pub fn digest_head(&self) -> Option<Sha16> {
        self.digest_head
    }

    // ---- HBP journal ----

    fn emit(&mut self, op: &str, fields: &[(&str, &str)]) {
        let row = hbp_row(op, fields);
        self.journal.push_back(row);
        // Bounded journal (GC discipline): drop-oldest past the cap. The digest
        // chain remains the durable, replayable summary of what was demoted.
        while self.journal.len() > self.journal_cap {
            self.journal.pop_front();
            self.journal_dropped += 1;
        }
    }

    /// Drain and return all pending HBP rows (the paging log). Each row ends
    /// `|json=0`. Draining keeps the journal bounded in a long-running lane.
    pub fn hbp_rows(&mut self) -> Vec<String> {
        self.journal.drain(..).collect()
    }

    /// Number of rows currently buffered (not yet drained).
    pub fn journal_len(&self) -> usize {
        self.journal.len()
    }
    /// Count of rows dropped by the bounded-journal GC.
    pub fn journal_dropped(&self) -> u64 {
        self.journal_dropped
    }

    // ---- test-only fault injection ----

    /// Overwrite a cold entry's bytes WITHOUT updating its id, simulating
    /// on-disk corruption/collision so [`get`] can be driven down the
    /// recover-or-Hold path. Returns false if the id is not cold.
    #[cfg(test)]
    fn tamper_cold_for_test(&mut self, id: &Sha16, new_bytes: Vec<u8>) -> bool {
        if self.store.contains_key(id) && !self.hot_bytes.contains_key(id) {
            self.store.insert(*id, new_bytes);
            true
        } else {
            false
        }
    }
}

// ===========================================================================
// Filesystem cube helpers (DESIGN — the atomic-spill discipline).
// ===========================================================================

/// Atomically write a content cube: write to a temp file, `sync_all`, then
/// `rename` into place. This is the write-before-pop primitive that makes an
/// eviction a MOVE rather than a drop.
///
/// Honest boundary: `rename` is atomic on the same volume, but a crash between
/// `rename` and a directory fsync can still lose the very last cube. This does
/// NOT claim "nothing is ever lost" beyond what the ordering guarantees.
pub fn write_cube_atomic(dir: &Path, id: &Sha16, bytes: &[u8]) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let hex = id.to_hex();
    let final_path = dir.join(format!("{}.cube", hex));
    let tmp_path = dir.join(format!("{}.cube.tmp", hex));
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, &final_path)?;
    Ok(final_path)
}

/// Read a cube and re-hash it, rejecting a mismatch (recover-or-Hold at the fs
/// boundary). Returns the verified bytes or an `InvalidData` error.
pub fn read_cube_verified(path: &Path, expected: &Sha16) -> io::Result<Vec<u8>> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let got = Sha16::of(&buf);
    if got != *expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "cube hash mismatch: expected {} got {}",
                expected.to_hex(),
                got.to_hex()
            ),
        ));
    }
    Ok(buf)
}

// ===========================================================================
// Tests — these actually exercise the mechanism.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn hex32(b: &[u8; 32]) -> String {
        let mut s = String::with_capacity(64);
        for x in b {
            s.push(hex_nibble(x >> 4));
            s.push(hex_nibble(x & 0x0f));
        }
        s
    }

    /// FIPS 180-4 known-answer vectors — proves the in-repo SHA-256 is correct,
    /// which is what makes every `sha16` address trustworthy.
    #[test]
    fn sha256_known_answer_vectors() {
        assert_eq!(
            hex32(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex32(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // sha16 = first 8 bytes of the "abc" digest.
        assert_eq!(Sha16::of(b"abc").to_hex(), "ba7816bf8f01cfea");
        // hex round-trip.
        let id = Sha16::of(b"abc");
        assert_eq!(Sha16::from_hex(&id.to_hex()), Some(id));
    }

    /// The core mechanism: pressure -> demote-oldest (evict = move) -> leave a
    /// verifiable digest -> recall on demand. Proves full entropy is retained
    /// and recall re-admits the exact bytes.
    #[test]
    fn evict_moves_and_recall_pages_back_in() {
        // budget 100, hi=90, lo=70. Each item is 30 bytes -> hot-eligible.
        let mut p = MemoryPager::new(100);
        let mut ids = Vec::new();
        for i in 0..6u8 {
            let payload = vec![i; 30];
            let out = p.insert(&payload);
            match out {
                InsertOutcome::Stored(id) | InsertOutcome::Deduped(id) => ids.push(id),
                InsertOutcome::Held(_) => panic!("30 bytes must be hot-eligible under budget 100"),
            }
        }

        // Hot set is bounded: usage never exceeds the high watermark.
        assert!(
            p.hot_usage() as f64 <= p.hi_bytes(),
            "hot usage {} must stay <= hi watermark {}",
            p.hot_usage(),
            p.hi_bytes()
        );
        // Eviction happened: at least one item is cold, and a digest exists.
        assert!(p.cold_len() >= 1, "expected some cold cubes after pressure");
        assert!(p.digest_head().is_some(), "a flush must leave a digest");
        assert!(p.verify_digest_chain(), "digest chain must verify");

        // FULL ENTROPY RETAINED: every id is still findable (hot or cold).
        for id in &ids {
            assert!(p.contains(id), "evict must MOVE, never drop: {}", id.to_hex());
        }

        // Find an id that was demoted to cold, then demand-recall it.
        let demoted = ids
            .iter()
            .copied()
            .find(|id| !p.is_hot(id))
            .expect("at least one item should have been evicted to cold");
        let want_byte = ids.iter().position(|x| *x == demoted).unwrap() as u8;
        match p.get(&demoted) {
            Recall::PagedIn(bytes) => {
                assert_eq!(bytes, vec![want_byte; 30], "recalled bytes must be exact");
                assert!(p.is_hot(&demoted), "page-in re-admits at the tail");
            }
            other => panic!("expected PagedIn from cold, got {:?}", other),
        }
    }

    /// Content addressing gives dedup for free: identical bytes are counted
    /// once and never double-charge the budget.
    #[test]
    fn dedup_does_not_double_charge() {
        let mut p = MemoryPager::new(1000);
        let a = p.insert(b"same-bytes");
        assert!(matches!(a, InsertOutcome::Stored(_)));
        let usage_after_first = p.hot_usage();
        let b = p.insert(b"same-bytes");
        assert!(matches!(b, InsertOutcome::Deduped(_)));
        assert_eq!(a.id(), b.id(), "identical bytes -> identical address");
        assert_eq!(
            p.hot_usage(),
            usage_after_first,
            "dedup must not re-charge usage"
        );
        assert_eq!(p.hot_len(), 1, "only one hot entry for duplicate bytes");
    }

    /// Every op emits a well-formed HBP hot-path row (`MEMPAGE|...|json=0`),
    /// and the standalone builder matches.
    #[test]
    fn hbp_rows_are_hot_path_tuples() {
        let mut p = MemoryPager::new(64);
        p.insert(b"row-check");
        let rows = p.hbp_rows();
        assert!(!rows.is_empty(), "insert must journal at least one row");
        for r in &rows {
            assert!(r.starts_with("MEMPAGE|op="), "row must be a MEMPAGE tuple: {}", r);
            assert!(r.ends_with("|json=0"), "hot-path row must end |json=0: {}", r);
            assert!(!r.contains('{'), "no JSON on the hot path: {}", r);
        }
        // Draining empties the bounded journal.
        assert_eq!(p.journal_len(), 0);
        // Standalone builder contract.
        let manual = hbp_row("evict", &[("sha16", "deadbeefdeadbeef"), ("bytes", "42")]);
        assert_eq!(manual, "MEMPAGE|op=evict|sha16=deadbeefdeadbeef|bytes=42|json=0");
    }

    /// Recover-or-Hold: a cold cube whose bytes no longer hash to its id is
    /// refused, never served silently wrong.
    #[test]
    fn corrupt_cold_cube_is_held_not_served() {
        // Small budget forces the first inserts cold quickly.
        let mut p = MemoryPager::new(60); // hi=54, lo=42; 30-byte items
        let id0 = p.insert(&vec![0u8; 30]).id();
        let _id1 = p.insert(&vec![1u8; 30]).id();
        let _id2 = p.insert(&vec![2u8; 30]).id();
        // id0 should be the oldest and demoted to cold.
        assert!(!p.is_hot(&id0), "oldest item should have been evicted");
        assert!(p.tamper_cold_for_test(&id0, vec![9u8; 30]), "tamper must land on a cold id");
        match p.get(&id0) {
            Recall::Corrupt => {}
            other => panic!("tampered cube must be Held/Corrupt, got {:?}", other),
        }
    }

    /// Oversize items are retained cold-only with a HOLD, and reads pass them
    /// through without ever entering (and thrashing) the hot set.
    #[test]
    fn oversize_item_is_held_and_passed_through() {
        let mut p = MemoryPager::new(100); // lo watermark = 70 bytes
        let big = vec![7u8; 80]; // > 70 -> not hot-eligible
        let out = p.insert(&big);
        let id = match out {
            InsertOutcome::Held(id) => id,
            other => panic!("oversize must be Held, got {:?}", other),
        };
        assert!(!p.is_hot(&id), "oversize never enters the hot set");
        assert!(p.contains(&id), "oversize is still retained cold (entropy kept)");
        match p.get(&id) {
            Recall::PagedIn(bytes) => {
                assert_eq!(bytes, big, "pass-through read returns exact bytes");
                assert!(!p.is_hot(&id), "pass-through must NOT admit oversize to hot");
            }
            other => panic!("expected pass-through PagedIn, got {:?}", other),
        }
    }

    /// The filesystem atomic-spill discipline: write a cube, verify-read it,
    /// then corrupt it on disk and confirm the verified read rejects it.
    #[test]
    fn fs_cube_write_verify_and_reject() {
        let dir = std::env::temp_dir().join(format!(
            "asolaria_mempage_test_{}_{}",
            std::process::id(),
            SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let bytes = b"content-addressed cube payload";
        let id = Sha16::of(bytes);

        let path = write_cube_atomic(&dir, &id, bytes).expect("atomic write");
        let read = read_cube_verified(&path, &id).expect("verified read of good cube");
        assert_eq!(read, bytes);

        // Corrupt on disk -> verified read must reject (recover-or-Hold).
        fs::write(&path, b"tampered").expect("overwrite");
        assert!(read_cube_verified(&path, &id).is_err(), "tampered cube must be rejected");

        // Wrong-id read of a good cube is also rejected.
        let other = Sha16::of(b"different");
        let good = write_cube_atomic(&dir, &id, bytes).expect("rewrite");
        assert!(read_cube_verified(&good, &other).is_err(), "wrong-id must be rejected");

        let _ = fs::remove_dir_all(&dir); // best-effort cleanup
    }

    // Unique suffix source so parallel fs tests never collide.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
}
