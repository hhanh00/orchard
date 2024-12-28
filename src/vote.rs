//! Proof of Balance
mod ballot;
mod circuit;
mod count;
mod encryption;
mod errors;
mod interval;
mod logical;
mod path;
mod proof;

pub use errors::VoteError;
pub use ballot::{BallotBuilder, BallotEnvelope};
use blake2b_simd::Params;
pub use circuit::{Circuit as BallotCircuit, ElectionDomain};
pub use count::{CandidateCountEnvelope, Circuit as CountCircuit, CountBuilder, CandidateCount};
pub use encryption::{DecryptedVote, EncryptedVote};
use ff::FromUniformBytes;
use incrementalmerkletree::{Altitude, Hashable};
use pasta_curves::pallas;
pub use proof::{ProvingKey, VerifyingKey};
pub use path::{build_nf_ranges, calculate_merkle_paths};

use crate::tree::MerkleHashOrchard;

type Hash = [u8; 32];
const DEPTH: usize = 32;

/// Hash the given info byte string to get the election domain
pub fn domain(info: &[u8]) -> pallas::Base {
    let hash = Params::new()
        .hash_length(64)
        .personal(b"ZcashVote_domain")
        .to_state()
        .update(info)
        .finalize();
    pallas::Base::from_uniform_bytes(hash.as_bytes().try_into().unwrap())
}

/// Orchard hash of two nodes of the CMX tree
pub fn cmx_hash(level: u8, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let left = MerkleHashOrchard::from_bytes(left).unwrap();
    let right = MerkleHashOrchard::from_bytes(right).unwrap();
    let h = MerkleHashOrchard::combine(Altitude::from(level), &left, &right);
    h.to_bytes()
}

/// Empty Orchard CMX hash
pub fn empty_hash() -> [u8; 32] {
    MerkleHashOrchard::empty_leaf().to_bytes()
}
