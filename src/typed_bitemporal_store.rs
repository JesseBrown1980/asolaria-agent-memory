//! # typed_bitemporal_store
//!
//! A zero-dependency, `std`-only, self-verifying **typed bitemporal edge store**.
//!
//! ## The winning core (SHANNON-minimal kernel)
//! "Hash the immutable, append the mutable." A record's identity
//! (`Sha16` = first 8 bytes / 16 hex of SHA-256) is computed over ONLY the
//! immutable payload — `kind ‖ subject ‖ predicate ‖ object ‖ t_valid` — and the
//! two mutable closure timestamps (`t_invalid`, `t_expired`) are NEVER hashed.
//! That one choice lets content-addressing and mutation coexist: "closing" a fact
//! never edits a row, it APPENDS a separate `INVAL` ledger row that references the
//! target `Sha16`. The store is therefore one append-only, hash-chained log of two
//! row kinds (immutable `PUT` + mutable `INVAL`), and "current state" is a pure JOIN
//! of a record with the latest closure for its id.
//!
//! ## Provenance tags
//! - **MEASURED** — bitemporality (4 timestamps, two independent axes),
//!   invalidate-never-delete conflict resolution, and typed lanes are the measured
//!   cores of Zep (arXiv:2501.13956) and CoALA (arXiv:2309.02427).
//! - **DESIGN** — the HBP tuple-row on-disk format, `Sha16` content addressing,
//!   the receipt-style hash chain, and the in-crate SHA-256 are Asolaria-native.
//!
//! ## Explicit scope / non-claims
//! - This is a faithful append-only LEDGER of *caller-supplied* tuples. It does NOT
//!   extract facts from text. Zep's LLM-driven fact/temporal extraction is
//!   **MEASURED-in-paper / NOT-BUILT** here.
//! - It is a functional-KEY edge log, NOT a knowledge graph: no multi-hop traversal,
//!   no community detection.
//! - `Sha16` is a 64-bit IDENTITY/dedupe key, NOT compression: the full immutable
//!   payload is retained; the hash cannot regenerate the fact. Collision probability
//!   is birthday-bounded `~n²/2^65` (negligible, NOT zero); `verify()` recomputes the
//!   full SHA-256 so a collision aliasing two facts is detectable.
//!
//! ## On-disk / on-wire format (HBP hot-path, `json=0`)
//! ```text
//! TYBT|op=put|kind=semantic|id=<sha16>|sub=<esc>|pred=<esc>|obj=<esc>|tv=<u64>|tc=<u64>|prev=<sha16>|json=0
//! TYBT|op=inval|target=<sha16>|ti=<u64>|te=<u64>|cause=<sha16>|prev=<sha16>|json=0
//! ```
//! `json=0` is literal: no serde, no serde_json, ever on this path.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ============================================================================
// SHA-256 (in-crate, pure std, KAT-verified in tests) — DESIGN
// ============================================================================

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

/// Full FIPS-180 SHA-256 over `data`, returning the 32-byte digest.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Padding: append 0x80, then zeros until len % 64 == 56, then the 64-bit big-endian bit length.
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = Vec::with_capacity(data.len() + 72);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
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

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

fn hex_nibble(c: u8) -> Result<u8, String> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(format!("invalid hex byte: {}", c as char)),
    }
}

// ============================================================================
// Sha16 — the 8-byte content-address / dedupe key — DESIGN
// ============================================================================

/// 8-byte content address: the first 8 bytes of a SHA-256 digest, rendered as
/// 16 lowercase hex chars. Identity/dedupe key ONLY — never a compressed payload.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Sha16(pub [u8; 8]);

impl Sha16 {
    /// The genesis / "no-parent" address (all zero). First row's `prev`.
    pub const ZERO: Sha16 = Sha16([0u8; 8]);

    /// Take the first 8 bytes of a full 32-byte SHA-256 digest.
    pub fn from_digest(d: &[u8; 32]) -> Sha16 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&d[..8]);
        Sha16(b)
    }

    /// Content-address a byte string directly.
    pub fn of(data: &[u8]) -> Sha16 {
        Sha16::from_digest(&sha256(data))
    }

    /// 16 lowercase hex chars.
    pub fn hex(&self) -> String {
        to_hex(&self.0)
    }

    /// Parse from 16 hex chars.
    pub fn from_hex(s: &str) -> Result<Sha16, String> {
        let b = s.as_bytes();
        if b.len() != 16 {
            return Err(format!("Sha16 must be 16 hex chars, got {}", b.len()));
        }
        let mut out = [0u8; 8];
        for i in 0..8 {
            out[i] = (hex_nibble(b[2 * i])? << 4) | hex_nibble(b[2 * i + 1])?;
        }
        Ok(Sha16(out))
    }
}

impl fmt::Display for Sha16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.hex())
    }
}

// ============================================================================
// Typed lanes (CoALA MEASURED) — one physical log, one `kind` discriminant
// ============================================================================

/// Memory lane. Only `Semantic` runs conflict/invalidation; `Episodic` is
/// append-only time-queried; `Procedural` is versioned. (CoALA MEASURED.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Episodic,
    Semantic,
    Procedural,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Episodic => "episodic",
            Kind::Semantic => "semantic",
            Kind::Procedural => "procedural",
        }
    }
    pub fn from_str(s: &str) -> Result<Kind, String> {
        match s {
            "episodic" => Ok(Kind::Episodic),
            "semantic" => Ok(Kind::Semantic),
            "procedural" => Ok(Kind::Procedural),
            other => Err(format!("unknown kind: {}", other)),
        }
    }
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ============================================================================
// Value escaping — makes row replay INJECTIVE (delimiter-injection safe) — DESIGN
// ============================================================================
//
// Caller strings (sub/pred/obj) may contain '|', '\n', '=' or '\\'. We escape
// those four bytes in VALUE fields only, so no crafted object can shift a field
// boundary or forge a row. (The hashed canon uses 0x1F separators instead, so the
// id is independently injection-proof.)

fn escape_val(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '|' => out.push_str("\\p"),
            '\n' => out.push_str("\\n"),
            '=' => out.push_str("\\e"),
            _ => out.push(c),
        }
    }
    out
}

fn unescape_val(s: &str) -> Result<String, String> {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('p') => out.push('|'),
                Some('n') => out.push('\n'),
                Some('e') => out.push('='),
                Some(other) => return Err(format!("bad escape: \\{}", other)),
                None => return Err("dangling escape at end of value".to_string()),
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

// ============================================================================
// Public data model
// ============================================================================

/// A caller-supplied immutable assertion. Its `Sha16` id is a pure function of
/// these five fields (world axis `t_valid` included).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub kind: Kind,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// World-valid start (caller-asserted world axis). See `t_created` for the system axis.
    pub t_valid: u64,
}

impl Fact {
    pub fn new(kind: Kind, subject: &str, predicate: &str, object: &str, t_valid: u64) -> Fact {
        Fact {
            kind,
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            t_valid,
        }
    }
    pub fn semantic(subject: &str, predicate: &str, object: &str, t_valid: u64) -> Fact {
        Fact::new(Kind::Semantic, subject, predicate, object, t_valid)
    }
    pub fn episodic(subject: &str, predicate: &str, object: &str, t_valid: u64) -> Fact {
        Fact::new(Kind::Episodic, subject, predicate, object, t_valid)
    }
    pub fn procedural(subject: &str, predicate: &str, object: &str, t_valid: u64) -> Fact {
        Fact::new(Kind::Procedural, subject, predicate, object, t_valid)
    }

    /// Compute the content address = hex16(SHA256(kind ‖ 0x1F ‖ s ‖ 0x1F ‖ p ‖ 0x1F ‖ o ‖ 0x1F ‖ tv)).
    /// 0x1F (unit separator) is used INSTEAD of '|' so a value containing '|'/' ' can
    /// never forge a different field boundary in the hashed payload.
    pub fn id(&self) -> Sha16 {
        const US: u8 = 0x1F;
        let mut buf = Vec::new();
        buf.extend_from_slice(self.kind.as_str().as_bytes());
        buf.push(US);
        buf.extend_from_slice(self.subject.as_bytes());
        buf.push(US);
        buf.extend_from_slice(self.predicate.as_bytes());
        buf.push(US);
        buf.extend_from_slice(self.object.as_bytes());
        buf.push(US);
        buf.extend_from_slice(self.t_valid.to_string().as_bytes());
        Sha16::of(&buf)
    }
}

/// An immutable `PUT` row: the fact plus the system-axis creation stamp and chain link.
#[derive(Clone, Debug)]
pub struct Record {
    pub id: Sha16,
    pub kind: Kind,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// World-valid start (world axis, caller-supplied).
    pub t_valid: u64,
    /// System/transaction creation stamp (system axis, monotone store counter).
    pub t_created: u64,
    /// Hash of the previous row (receipt-style chain).
    pub prev: Sha16,
}

/// A mutable `INVAL` closure row. Never edits the target — appended, then JOINed at read.
#[derive(Clone, Debug)]
pub struct Closure {
    pub target: Sha16,
    /// World-valid end (world axis).
    pub t_invalid: u64,
    /// System-transaction end (system axis, monotone counter at closure time).
    pub t_expired: u64,
    /// The new record that caused this closure (`Sha16::ZERO` for a manual invalidate).
    pub cause: Sha16,
    pub prev: Sha16,
}

/// One physical log row.
#[derive(Clone, Debug)]
pub enum Row {
    Put(Record),
    Inval(Closure),
}

impl Row {
    fn prev(&self) -> Sha16 {
        match self {
            Row::Put(r) => r.prev,
            Row::Inval(c) => c.prev,
        }
    }
}

/// A read-side view: a record JOINed with its latest closure (if any).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactView {
    pub id: Sha16,
    pub kind: Kind,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub t_valid: u64,
    pub t_created: u64,
    /// `None` while the world interval is still open.
    pub t_invalid: Option<u64>,
    /// `None` while the system interval is still open.
    pub t_expired: Option<u64>,
}

impl FactView {
    /// Is this record currently open (not closed on either axis)?
    pub fn is_open(&self) -> bool {
        self.t_invalid.is_none() && self.t_expired.is_none()
    }
}

/// Outcome of a `put`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PutResult {
    /// A brand-new live edge was appended.
    Inserted(Sha16),
    /// An identical (kind,s,p,o,t_valid) already exists — no work, no resurrection.
    Duplicate(Sha16),
    /// A functional-key predecessor (or predecessors) was closed and the new edge inserted.
    Superseded { new: Sha16, closed: Vec<Sha16> },
}

// ============================================================================
// The store
// ============================================================================

/// Typed bitemporal edge store over a single append-only, hash-chained HBP log.
///
/// The `rows` vector IS the source of truth; every other field is an EPHEMERAL
/// index rebuilt by replaying rows on `from_hbp`/`open`.
pub struct Store {
    rows: Vec<Row>,
    /// id -> index in `rows` (PUT rows).
    records: HashMap<Sha16, usize>,
    /// target id -> index in `rows` (latest INVAL closure for that id).
    closures: HashMap<Sha16, usize>,
    /// (subject, predicate) -> all record ids ever asserted for that functional key.
    fkey: HashMap<(String, String), Vec<Sha16>>,
    /// Predicates declared single-valued (only these auto-invalidate on conflict).
    functional: HashSet<String>,
    /// Strictly-monotone SYSTEM clock (NOT wall time). Next value to assign.
    sys_clock: u64,
    /// Hash of the last appended row (the chain head).
    last_hash: Sha16,
}

impl Default for Store {
    fn default() -> Self {
        Store::new()
    }
}

impl Store {
    pub fn new() -> Store {
        Store {
            rows: Vec::new(),
            records: HashMap::new(),
            closures: HashMap::new(),
            fkey: HashMap::new(),
            functional: HashSet::new(),
            sys_clock: 1, // start at 1 so system time 0 == "before anything is known"
            last_hash: Sha16::ZERO,
        }
    }

    /// Declare `predicate` FUNCTIONAL (single-valued). Only functional semantic
    /// predicates auto-close a predecessor on conflict; multi-valued predicates
    /// (e.g. `has_tag`, `member_of`) accumulate live edges. (Zep MEASURED, gated.)
    pub fn register_functional(&mut self, predicate: &str) {
        self.functional.insert(predicate.to_string());
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.functional.contains(predicate)
    }

    fn next_sys(&mut self) -> u64 {
        let t = self.sys_clock;
        self.sys_clock += 1;
        t
    }

    /// Assert a fact. Returns `Duplicate` for a byte-identical re-assertion (even if
    /// currently closed — no resurrection), `Superseded` if a functional predecessor
    /// was closed in the same append batch, else `Inserted`.
    pub fn put(&mut self, fact: Fact) -> PutResult {
        let id = fact.id();

        // CONTENT-ADDRESSED DEDUPE (DESIGN): identical id => same fact => no work.
        if self.records.contains_key(&id) {
            return PutResult::Duplicate(id);
        }

        let key = (fact.subject.clone(), fact.predicate.clone());
        let mut closed: Vec<Sha16> = Vec::new();

        // FUNCTIONAL-KEY CONFLICT (Zep MEASURED, per-predicate gated): if this
        // predicate is single-valued, close every currently-open predecessor for the
        // key so at most one live edge remains (invariant I2).
        if self.functional.contains(&fact.predicate) {
            if let Some(ids) = self.fkey.get(&key) {
                for old_id in ids.clone() {
                    if old_id != id && !self.closures.contains_key(&old_id) {
                        closed.push(old_id);
                    }
                }
            }
        }

        // --- APPEND THE NEW PUT ROW (atomic batch begins) ---
        let t_created = self.next_sys();
        let prev = self.last_hash;
        let rec = Record {
            id,
            kind: fact.kind,
            subject: fact.subject.clone(),
            predicate: fact.predicate.clone(),
            object: fact.object.clone(),
            t_valid: fact.t_valid,
            t_created,
            prev,
        };
        self.push_put(rec);

        // --- APPEND EACH PREDECESSOR CLOSURE IN THE SAME BATCH ---
        // MONOTONE CLOSURE: t_invalid = max(new.t_valid, old.t_valid) so a world
        // interval is never negative; t_expired = now (system axis).
        for old_id in &closed {
            let old_tv = match self.records.get(old_id) {
                Some(&idx) => match &self.rows[idx] {
                    Row::Put(r) => r.t_valid,
                    _ => 0,
                },
                None => 0,
            };
            let t_invalid = fact.t_valid.max(old_tv);
            let t_expired = self.next_sys();
            let prev = self.last_hash;
            let clo = Closure {
                target: *old_id,
                t_invalid,
                t_expired,
                cause: id,
                prev,
            };
            self.push_inval(clo);
        }

        if closed.is_empty() {
            PutResult::Inserted(id)
        } else {
            PutResult::Superseded { new: id, closed }
        }
    }

    /// Manually close an OPEN record on both axes at world time `t_invalid`.
    /// Returns `false` (monotone guard) if the id is unknown, already closed, or
    /// `t_invalid < t_valid`. History is never rewritten (invariant I1).
    pub fn invalidate(&mut self, id: Sha16, t_invalid: u64) -> bool {
        let idx = match self.records.get(&id) {
            Some(&i) => i,
            None => return false,
        };
        if self.closures.contains_key(&id) {
            return false; // already closed: reject re-closing
        }
        let t_valid = match &self.rows[idx] {
            Row::Put(r) => r.t_valid,
            _ => return false,
        };
        if t_invalid < t_valid {
            return false; // reject an interval that would run backward
        }
        let t_expired = self.next_sys();
        let prev = self.last_hash;
        let clo = Closure {
            target: id,
            t_invalid,
            t_expired,
            cause: Sha16::ZERO,
            prev,
        };
        self.push_inval(clo);
        true
    }

    // ---- internal append (chain + index maintenance) ----

    fn push_put(&mut self, rec: Record) {
        let line = record_hbp_row(&rec);
        self.last_hash = Sha16::of(line.as_bytes());
        let idx = self.rows.len();
        self.records.insert(rec.id, idx);
        self.fkey
            .entry((rec.subject.clone(), rec.predicate.clone()))
            .or_default()
            .push(rec.id);
        self.rows.push(Row::Put(rec));
    }

    fn push_inval(&mut self, clo: Closure) {
        let line = closure_hbp_row(&clo);
        self.last_hash = Sha16::of(line.as_bytes());
        let idx = self.rows.len();
        self.closures.insert(clo.target, idx);
        self.rows.push(Row::Inval(clo));
    }

    // ---- reads ----

    fn view_of(&self, idx: usize) -> Option<FactView> {
        let rec = match &self.rows[idx] {
            Row::Put(r) => r,
            _ => return None,
        };
        let (t_invalid, t_expired) = match self.closures.get(&rec.id) {
            Some(&cidx) => match &self.rows[cidx] {
                Row::Inval(c) => (Some(c.t_invalid), Some(c.t_expired)),
                _ => (None, None),
            },
            None => (None, None),
        };
        Some(FactView {
            id: rec.id,
            kind: rec.kind,
            subject: rec.subject.clone(),
            predicate: rec.predicate.clone(),
            object: rec.object.clone(),
            t_valid: rec.t_valid,
            t_created: rec.t_created,
            t_invalid,
            t_expired,
        })
    }

    /// Fetch a single record (with its closure JOINed) by id.
    pub fn get(&self, id: Sha16) -> Option<FactView> {
        self.records.get(&id).and_then(|&idx| self.view_of(idx))
    }

    /// Currently-live edges for `(subject, predicate)` — records with no closure.
    /// For a functional key this returns at most one edge (invariant I2).
    pub fn current(&self, subject: &str, predicate: &str) -> Vec<FactView> {
        let key = (subject.to_string(), predicate.to_string());
        let mut out = Vec::new();
        if let Some(ids) = self.fkey.get(&key) {
            for id in ids {
                if !self.closures.contains_key(id) {
                    if let Some(&idx) = self.records.get(id) {
                        if let Some(v) = self.view_of(idx) {
                            out.push(v);
                        }
                    }
                }
            }
        }
        out
    }

    /// Full version history for `(subject, predicate)`, open and closed, ordered by
    /// `t_valid` then `t_created`.
    pub fn history(&self, subject: &str, predicate: &str) -> Vec<FactView> {
        let key = (subject.to_string(), predicate.to_string());
        let mut out = Vec::new();
        if let Some(ids) = self.fkey.get(&key) {
            for id in ids {
                if let Some(&idx) = self.records.get(id) {
                    if let Some(v) = self.view_of(idx) {
                        out.push(v);
                    }
                }
            }
        }
        out.sort_by(|a, b| {
            a.t_valid
                .cmp(&b.t_valid)
                .then(a.t_created.cmp(&b.t_created))
        });
        out
    }

    /// BITEMPORAL RECONSTRUCTION (Zep MEASURED): every edge that was live at world
    /// time `w` **as the store believed it at system time `s`**. Pass `s = u64::MAX`
    /// for "latest belief". A closure only takes effect on the system axis once
    /// `t_expired <= s`, so `as_of` is total-audit: past beliefs are reconstructible.
    pub fn as_of(&self, w: u64, s: u64) -> Vec<FactView> {
        let mut out = Vec::new();
        for idx in 0..self.rows.len() {
            let rec = match &self.rows[idx] {
                Row::Put(r) => r,
                _ => continue,
            };
            // System axis: the record must be known by system time s.
            if rec.t_created > s {
                continue;
            }
            // Effective world-close is applied only if the closure had happened by s.
            let eff_t_invalid = match self.closures.get(&rec.id) {
                Some(&cidx) => match &self.rows[cidx] {
                    Row::Inval(c) if c.t_expired <= s => c.t_invalid,
                    _ => u64::MAX,
                },
                None => u64::MAX,
            };
            // World axis: t_valid <= w < effective t_invalid.
            if rec.t_valid <= w && w < eff_t_invalid {
                if let Some(v) = self.view_of(idx) {
                    out.push(v);
                }
            }
        }
        out
    }

    // ---- integrity: verify() is the product, not a debug aid ----

    /// Re-prove the three invariants against the raw rows:
    /// - **I3** (chain): each row's `prev` equals `Sha16` of the canonical previous row;
    /// - **I4** (addressing): every stored id recomputes from its immutable fields;
    /// - **I2** (functional uniqueness): at most one open edge per functional key.
    /// Any tampered field or row breaks I4 and every downstream I3 link.
    pub fn verify(&self) -> Result<(), String> {
        // I3: hash chain.
        let mut prev = Sha16::ZERO;
        for (i, row) in self.rows.iter().enumerate() {
            if row.prev() != prev {
                return Err(format!("I3 chain break at row {}: prev mismatch", i));
            }
            let line = row_hbp_row(row);
            prev = Sha16::of(line.as_bytes());
        }
        if prev != self.last_hash {
            return Err("I3 chain head mismatch vs last_hash".to_string());
        }

        // I4: recompute every content address from stored immutable fields.
        for (i, row) in self.rows.iter().enumerate() {
            if let Row::Put(r) = row {
                let recomputed = Fact {
                    kind: r.kind,
                    subject: r.subject.clone(),
                    predicate: r.predicate.clone(),
                    object: r.object.clone(),
                    t_valid: r.t_valid,
                }
                .id();
                if recomputed != r.id {
                    return Err(format!(
                        "I4 addressing break at row {}: id {} != recomputed {}",
                        i, r.id, recomputed
                    ));
                }
            }
        }

        // I2: at most one open edge per functional key.
        for (key, ids) in &self.fkey {
            if self.functional.contains(&key.1) {
                let open = ids
                    .iter()
                    .filter(|id| !self.closures.contains_key(id))
                    .count();
                if open > 1 {
                    return Err(format!(
                        "I2 violation: {} open edges for functional key ({}, {})",
                        open, key.0, key.1
                    ));
                }
            }
        }
        Ok(())
    }

    // ---- HBP hot-path I/O ----

    /// Stream every raw HBP row (each ending `|json=0`) for external re-verification.
    pub fn hbp_rows(&self) -> Vec<String> {
        self.rows.iter().map(row_hbp_row).collect()
    }

    /// The whole log as an HBP document (rows newline-joined, trailing newline).
    pub fn to_hbp(&self) -> String {
        let mut s = String::new();
        for r in &self.rows {
            s.push_str(&row_hbp_row(r));
            s.push('\n');
        }
        s
    }

    /// Rebuild a store by replaying an HBP document. Verifies the chain (I3) and
    /// recomputes ids (I4) as it goes; a broken tail row is reported as an error
    /// rather than silently trusted. Functional-predicate declarations are NOT stored
    /// in rows, so re-declare them (or call `register_functional`) before conflict use.
    pub fn from_hbp(text: &str) -> Result<Store, String> {
        let mut store = Store::new();
        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            store
                .replay_line(line)
                .map_err(|e| format!("line {}: {}", lineno + 1, e))?;
        }
        Ok(store)
    }

    fn replay_line(&mut self, line: &str) -> Result<(), String> {
        let row = parse_row(line)?;
        // I3: the row's prev must match our running chain head.
        if row.prev() != self.last_hash {
            return Err("chain break: prev does not match running head".to_string());
        }
        match &row {
            Row::Put(r) => {
                // I4: recompute the id from the immutable fields.
                let recomputed = Fact {
                    kind: r.kind,
                    subject: r.subject.clone(),
                    predicate: r.predicate.clone(),
                    object: r.object.clone(),
                    t_valid: r.t_valid,
                }
                .id();
                if recomputed != r.id {
                    return Err(format!(
                        "id mismatch: stored {} recomputed {}",
                        r.id, recomputed
                    ));
                }
                self.bump_clock(r.t_created);
                let idx = self.rows.len();
                self.records.insert(r.id, idx);
                self.fkey
                    .entry((r.subject.clone(), r.predicate.clone()))
                    .or_default()
                    .push(r.id);
            }
            Row::Inval(c) => {
                if !self.records.contains_key(&c.target) {
                    return Err(format!("INVAL targets unknown record {}", c.target));
                }
                self.bump_clock(c.t_expired);
            }
        }
        // Chain forward over the canonical serialization (must equal the input line).
        self.last_hash = Sha16::of(row_hbp_row(&row).as_bytes());
        self.rows.push(row);
        Ok(())
    }

    fn bump_clock(&mut self, seen: u64) {
        if seen >= self.sys_clock {
            self.sys_clock = seen + 1;
        }
    }

    /// Total row count (PUT + INVAL).
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Persist to `path` (HBP document) plus a `<path>.sha256` sidecar over the file bytes.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let doc = self.to_hbp();
        std::fs::write(path, &doc)?;
        let digest = to_hex(&sha256(doc.as_bytes()));
        std::fs::write(format!("{}.sha256", path), format!("{}\n", digest))?;
        Ok(())
    }

    /// Load from `path`, checking the `<path>.sha256` sidecar if present, then replay.
    pub fn open(path: &str) -> Result<Store, String> {
        let doc = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        let sidecar = format!("{}.sha256", path);
        if let Ok(expected) = std::fs::read_to_string(&sidecar) {
            let expected = expected.trim();
            let got = to_hex(&sha256(doc.as_bytes()));
            if !expected.is_empty() && expected != got {
                return Err(format!(
                    "sidecar mismatch: expected {} got {}",
                    expected, got
                ));
            }
        }
        Store::from_hbp(&doc)
    }
}

// ============================================================================
// HBP row serialization (the file IS the tuple rows) — DESIGN, `json=0`
// ============================================================================

/// Emit the canonical HBP row for any log row (String ending `|json=0`).
pub fn row_hbp_row(row: &Row) -> String {
    match row {
        Row::Put(r) => record_hbp_row(r),
        Row::Inval(c) => closure_hbp_row(c),
    }
}

/// Emit a PUT record as an HBP tuple row (ends `|json=0`).
pub fn record_hbp_row(r: &Record) -> String {
    format!(
        "TYBT|op=put|kind={}|id={}|sub={}|pred={}|obj={}|tv={}|tc={}|prev={}|json=0",
        r.kind.as_str(),
        r.id.hex(),
        escape_val(&r.subject),
        escape_val(&r.predicate),
        escape_val(&r.object),
        r.t_valid,
        r.t_created,
        r.prev.hex(),
    )
}

/// Emit an INVAL closure as an HBP tuple row (ends `|json=0`).
pub fn closure_hbp_row(c: &Closure) -> String {
    format!(
        "TYBT|op=inval|target={}|ti={}|te={}|cause={}|prev={}|json=0",
        c.target.hex(),
        c.t_invalid,
        c.t_expired,
        c.cause.hex(),
        c.prev.hex(),
    )
}

/// Parse one canonical HBP row back into a `Row`. Value fields are unescaped;
/// splitting on '|' is safe because escaped values encode '|' as `\p`.
pub fn parse_row(line: &str) -> Result<Row, String> {
    let mut fields: HashMap<&str, &str> = HashMap::new();
    let mut first = true;
    for tok in line.split('|') {
        if first {
            if tok != "TYBT" {
                return Err(format!("bad magic token: {:?}", tok));
            }
            first = false;
            continue;
        }
        let (k, v) = tok
            .split_once('=')
            .ok_or_else(|| format!("field without '=': {:?}", tok))?;
        fields.insert(k, v);
    }

    if fields.get("json") != Some(&"0") {
        return Err("missing or non-zero json marker (json must be 0)".to_string());
    }

    let get = |k: &str| -> Result<&str, String> {
        fields
            .get(k)
            .copied()
            .ok_or_else(|| format!("missing field: {}", k))
    };
    let get_u64 = |k: &str| -> Result<u64, String> {
        get(k)?
            .parse::<u64>()
            .map_err(|_| format!("field {} is not a u64", k))
    };

    match get("op")? {
        "put" => Ok(Row::Put(Record {
            id: Sha16::from_hex(get("id")?)?,
            kind: Kind::from_str(get("kind")?)?,
            subject: unescape_val(get("sub")?)?,
            predicate: unescape_val(get("pred")?)?,
            object: unescape_val(get("obj")?)?,
            t_valid: get_u64("tv")?,
            t_created: get_u64("tc")?,
            prev: Sha16::from_hex(get("prev")?)?,
        })),
        "inval" => Ok(Row::Inval(Closure {
            target: Sha16::from_hex(get("target")?)?,
            t_invalid: get_u64("ti")?,
            t_expired: get_u64("te")?,
            cause: Sha16::from_hex(get("cause")?)?,
            prev: Sha16::from_hex(get("prev")?)?,
        })),
        other => Err(format!("unknown op: {}", other)),
    }
}

// ============================================================================
// Tests — exercise the real mechanism
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_known_answer_vectors() {
        // FIPS-180 KATs: proves the in-crate SHA-256 (and thus every Sha16 id) is correct.
        assert_eq!(
            to_hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            to_hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            to_hex(&sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Sha16 = first 8 bytes / 16 hex chars.
        assert_eq!(Sha16::of(b"abc").hex(), "ba7816bf8f01cfea");
    }

    #[test]
    fn dedupe_and_functional_supersession() {
        let mut s = Store::new();
        s.register_functional("capital_of");

        let r1 = s.put(Fact::semantic("France", "capital_of", "Paris", 10));
        let id1 = match r1 {
            PutResult::Inserted(id) => id,
            other => panic!("expected Inserted, got {:?}", other),
        };

        // Byte-identical re-assertion => Duplicate, no work, no new row.
        let before = s.row_count();
        assert_eq!(
            s.put(Fact::semantic("France", "capital_of", "Paris", 10)),
            PutResult::Duplicate(id1)
        );
        assert_eq!(s.row_count(), before, "duplicate must not append a row");

        // A functional conflict at a later world time supersedes the predecessor.
        let r2 = s.put(Fact::semantic("France", "capital_of", "Lyon", 20));
        match r2 {
            PutResult::Superseded { new: _, closed } => {
                assert_eq!(closed, vec![id1], "old edge must be the closed one");
            }
            other => panic!("expected Superseded, got {:?}", other),
        }

        // current() sees exactly the new, live edge; history() keeps both.
        let cur = s.current("France", "capital_of");
        assert_eq!(cur.len(), 1);
        assert_eq!(cur[0].object, "Lyon");
        assert!(cur[0].is_open());

        let hist = s.history("France", "capital_of");
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].object, "Paris");
        assert_eq!(hist[0].t_invalid, Some(20)); // closed at new.t_valid
        assert!(hist[0].t_expired.is_some());
        assert_eq!(hist[1].object, "Lyon");

        s.verify().expect("invariants must hold after supersession");
    }

    #[test]
    fn multivalued_predicate_accumulates() {
        // has_tag is NOT registered functional -> a second value is an ADDITION.
        let mut s = Store::new();
        s.put(Fact::semantic("doc1", "has_tag", "rust", 1));
        s.put(Fact::semantic("doc1", "has_tag", "memory", 2));

        let cur = s.current("doc1", "has_tag");
        assert_eq!(cur.len(), 2, "multi-valued predicate must keep both live");
        s.verify().expect("non-functional keys never trip I2");
    }

    #[test]
    fn bitemporal_as_of_reconstructs_past() {
        let mut s = Store::new();
        s.register_functional("status");

        let r1 = s.put(Fact::semantic("svc", "status", "active", 100));
        let id1 = match r1 {
            PutResult::Inserted(id) => id,
            _ => unreachable!(),
        };
        let created1 = s.get(id1).unwrap().t_created;

        // Supersede: "active" world-closes at 200, "closed" opens at 200.
        s.put(Fact::semantic("svc", "status", "closed", 200));

        // Latest belief, world time inside the old interval => still "active".
        let v = s.as_of(150, u64::MAX);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].object, "active");

        // Latest belief, world time past supersession => "closed".
        let v = s.as_of(250, u64::MAX);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].object, "closed");

        // System axis: as the store believed at the instant only "active" was known,
        // the closure had not happened yet, so "active" is still open at w=150.
        let v = s.as_of(150, created1);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].object, "active");

        // Before any world validity => nothing was live.
        assert!(s.as_of(50, u64::MAX).is_empty());

        s.verify().unwrap();
    }

    #[test]
    fn hbp_roundtrip_and_injection_safety() {
        let mut s = Store::new();
        s.register_functional("owns");
        // An object crafted with delimiter/injection bytes must survive replay intact.
        let nasty = "a|b=c\nd\\e";
        s.put(Fact::semantic("u1", "owns", nasty, 5));
        s.put(Fact::episodic("u1", "logged_in", "session-7", 6));
        s.put(Fact::semantic("u1", "owns", "thing2", 9)); // supersedes the nasty one

        let doc = s.to_hbp();
        // Every emitted row is HBP hot-path (json=0), no JSON anywhere.
        for row in s.hbp_rows() {
            assert!(row.ends_with("|json=0"));
            assert!(!row.contains("{"));
        }

        let s2 = Store::from_hbp(&doc).expect("replay must succeed");
        s2.verify()
            .expect("replayed store must self-verify (I2/I3/I4)");
        assert_eq!(s2.row_count(), s.row_count());

        // The injection-laden object round-tripped byte-for-byte.
        let hist = s2.history("u1", "owns");
        assert!(hist.iter().any(|v| v.object == nasty));
        // Re-serialization is stable (canonical form is a fixed point).
        assert_eq!(s2.to_hbp(), doc);
    }

    #[test]
    fn tamper_breaks_verify() {
        let mut s = Store::new();
        s.put(Fact::semantic("k", "eq", "v1", 1));
        let mut doc = s.to_hbp();
        // Flip an object value in the raw log without recomputing its id/chain.
        assert!(doc.contains("obj=v1"));
        doc = doc.replace("obj=v1", "obj=v2");
        // from_hbp recomputes the id (I4) and rejects the forged row.
        assert!(
            Store::from_hbp(&doc).is_err(),
            "tampered log must not replay clean"
        );
    }

    #[test]
    fn manual_invalidate_is_monotone() {
        let mut s = Store::new();
        let r = s.put(Fact::semantic("a", "b", "c", 10));
        let id = match r {
            PutResult::Inserted(id) => id,
            _ => unreachable!(),
        };
        assert!(!s.invalidate(id, 5), "t_invalid < t_valid must be rejected");
        assert!(s.invalidate(id, 20), "valid close must succeed");
        assert!(
            !s.invalidate(id, 30),
            "re-closing a closed edge must be rejected"
        );
        assert!(
            s.current("a", "b").is_empty(),
            "closed edge is no longer current"
        );
        s.verify().unwrap();
    }
}
