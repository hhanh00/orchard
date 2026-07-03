//! Defines the `NoteFlavor` trait and marker types for Orchard protocol note flavors.
//!
//! `NoteFlavor` controls compile-time note sizes (Normal = 52-byte compact notes,
//! ZSA = 84-byte compact notes) and provides the byte array types used in
//! `Action`, `Bundle`, and `TransmittedNoteCiphertext`.
//!
//! Behavioral differences (hashing, note decryption, asset extraction) are
//! dispatched at runtime via `BundleVersion`, not through this trait.

use core::fmt;
use zcash_note_encryption::{
    note_bytes::{NoteBytes, NoteBytesData},
    AEAD_TAG_SIZE,
};

/// Size constants used by the note encryption domain.
const NOTE_VERSION_SIZE: usize = 1;
const NOTE_DIVERSIFIER_SIZE: usize = 11;
const NOTE_VALUE_SIZE: usize = 8;
const NOTE_RSEED_SIZE: usize = 32;
const ZSA_ASSET_SIZE: usize = 32;

const NOTE_VERSION_OFFSET: usize = 0;
const NOTE_DIVERSIFIER_OFFSET: usize = NOTE_VERSION_OFFSET + NOTE_VERSION_SIZE;
const NOTE_VALUE_OFFSET: usize = NOTE_DIVERSIFIER_OFFSET + NOTE_DIVERSIFIER_SIZE;
const NOTE_RSEED_OFFSET: usize = NOTE_VALUE_OFFSET + NOTE_VALUE_SIZE;

/// Vanilla compact note size (52 bytes).
pub(crate) const COMPACT_NOTE_SIZE_VANILLA: usize = NOTE_RSEED_OFFSET + NOTE_RSEED_SIZE;

/// ZSA compact note size (84 bytes = 52 + 32-byte asset).
pub(crate) const COMPACT_NOTE_SIZE_ZSA: usize = COMPACT_NOTE_SIZE_VANILLA + ZSA_ASSET_SIZE;

/// Memo size (512 bytes).
pub(crate) const MEMO_SIZE: usize = 512;

pub(crate) type Memo = [u8; MEMO_SIZE];

/// Version byte for normal (V2) notes.
pub(crate) const NOTE_VERSION_BYTE_V2: u8 = 0x02;

/// Version byte for ZSA (V3) notes.
pub(crate) const NOTE_VERSION_BYTE_V3: u8 = 0x03;

/// Compile-time Orchard protocol note flavor.
///
/// Controls note plaintext and ciphertext sizes. Behavioral differences are
/// dispatched at runtime via `BundleVersion`.
pub trait NoteFlavor: fmt::Debug + Clone + PartialEq + Eq {
    /// Size of the compact note (first N bytes of the encrypted note ciphertext).
    const COMPACT_NOTE_SIZE: usize;

    /// Size of the note plaintext = COMPACT_NOTE_SIZE + MEMO_SIZE.
    const NOTE_PLAINTEXT_SIZE: usize;

    /// Size of the full encrypted note ciphertext = NOTE_PLAINTEXT_SIZE + AEAD_TAG_SIZE.
    const ENC_CIPHERTEXT_SIZE: usize;

    /// Byte type for full note plaintext (note + memo).
    type NotePlaintextBytes: NoteBytes;

    /// Byte type for full encrypted note ciphertext.
    type NoteCiphertextBytes: NoteBytes;

    /// Byte type for compact note plaintext (note fields only, no memo).
    type CompactNotePlaintextBytes: NoteBytes;

    /// Byte type for compact note ciphertext (no AEAD tag; same size as compact plaintext).
    type CompactNoteCiphertextBytes: NoteBytes;
}

/// Normal Orchard flavor — 52-byte compact notes, zatoshi-only assets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NormalFlavor;

impl NoteFlavor for NormalFlavor {
    const COMPACT_NOTE_SIZE: usize = COMPACT_NOTE_SIZE_VANILLA;
    const NOTE_PLAINTEXT_SIZE: usize = COMPACT_NOTE_SIZE_VANILLA + MEMO_SIZE;
    const ENC_CIPHERTEXT_SIZE: usize = COMPACT_NOTE_SIZE_VANILLA + MEMO_SIZE + AEAD_TAG_SIZE;

    type NotePlaintextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_VANILLA + MEMO_SIZE }>;
    type NoteCiphertextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_VANILLA + MEMO_SIZE + AEAD_TAG_SIZE }>;
    type CompactNotePlaintextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_VANILLA }>;
    type CompactNoteCiphertextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_VANILLA }>;
}

/// ZSA Orchard flavor — 84-byte compact notes (52 + 32-byte asset), custom assets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ZsaFlavor;

impl NoteFlavor for ZsaFlavor {
    const COMPACT_NOTE_SIZE: usize = COMPACT_NOTE_SIZE_ZSA;
    const NOTE_PLAINTEXT_SIZE: usize = COMPACT_NOTE_SIZE_ZSA + MEMO_SIZE;
    const ENC_CIPHERTEXT_SIZE: usize = COMPACT_NOTE_SIZE_ZSA + MEMO_SIZE + AEAD_TAG_SIZE;

    type NotePlaintextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_ZSA + MEMO_SIZE }>;
    type NoteCiphertextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_ZSA + MEMO_SIZE + AEAD_TAG_SIZE }>;
    type CompactNotePlaintextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_ZSA }>;
    type CompactNoteCiphertextBytes = NoteBytesData<{ COMPACT_NOTE_SIZE_ZSA }>;
}

/// Version byte for a note plaintext (lead byte).
pub(crate) fn note_version_byte<F: NoteFlavor>() -> u8 {
    match F::COMPACT_NOTE_SIZE {
        COMPACT_NOTE_SIZE_VANILLA => NOTE_VERSION_BYTE_V2,
        COMPACT_NOTE_SIZE_ZSA => NOTE_VERSION_BYTE_V3,
        _ => unreachable!(),
    }
}

/// Convenience supertrait combining compile-time flavor sizing and circuit support.
///
/// Used in contexts where both note flavor and circuit generation are needed
/// (e.g., `ProvingKey::build`, `Builder::build`). The `OrchardCircuit` trait
/// is from `crate::circuit`.
#[cfg(feature = "circuit")]
pub trait OrchardFlavor: NoteFlavor + crate::circuit::OrchardCircuit {}

#[cfg(feature = "circuit")]
impl OrchardFlavor for NormalFlavor {}

#[cfg(feature = "circuit")]
impl OrchardFlavor for ZsaFlavor {}
