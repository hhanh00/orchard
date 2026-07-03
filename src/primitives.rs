//! Primitives used in the Orchard protocol.
//!
//! This module handles
//! - the encryption and decryption of notes,
//! - the commitments,
//! - the redpallas signatures.
//!
//! It includes functionality for handling both Normal and ZSA note flavors, with
//! compile-time size dispatch via the `NoteFlavor` generic parameter.

mod compact_action;
mod orchard_domain;
mod zcash_note_encryption_domain;
pub mod redpallas;

pub use {
    compact_action::CompactAction, orchard_domain::OrchardDomain,
};

#[cfg(feature = "test-dependencies")]
pub use compact_action::testing::fake_compact_action;
