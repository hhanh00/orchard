//! Proof of Balance
mod circuit;
mod interval;

pub use circuit::{
    create_proof, verify_proof, {Proof, ProofBalance, ProofBalancePublic},
};

use crate::tree::MerkleHashOrchard;
use bridgetree::{Hashable, Level};

/// Orchard hash of two nodes of the CMX tree
pub fn cmx_hash(level: u8, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let left = MerkleHashOrchard::from_bytes(left).unwrap();
    let right = MerkleHashOrchard::from_bytes(right).unwrap();
    let h = MerkleHashOrchard::combine(Level::new(level), &left, &right);
    h.to_bytes()
}

/// Empty Orchard CMX hash
pub fn empty_hash() -> [u8; 32] {
    MerkleHashOrchard::empty_leaf().to_bytes()
}
