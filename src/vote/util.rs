use blake2b_simd::Params;
use ff::FromUniformBytes as _;
use incrementalmerkletree::Hashable as _;
use pasta_curves::Fp;

use crate::tree::MerkleHashOrchard;

use super::VoteError;

/// Orchard hash of two nodes of the CMX tree
pub(crate) fn cmx_hash(level: u8, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let left = MerkleHashOrchard::from_bytes(left).unwrap();
    let right = MerkleHashOrchard::from_bytes(right).unwrap();
    let h = MerkleHashOrchard::combine(incrementalmerkletree::Level::from(level), &left, &right);
    h.to_bytes()
}

/// Empty Orchard CMX hash
pub(crate) fn empty_hash() -> [u8; 32] {
    MerkleHashOrchard::empty_leaf().to_bytes()
}

/// Hash the given info byte string to get the election domain
pub fn calculate_domain(info: &[u8]) -> Fp {
    let hash = Params::new()
        .hash_length(64)
        .personal(b"ZcashVote_domain")
        .to_state()
        .update(info)
        .finalize();
    Fp::from_uniform_bytes(hash.as_bytes().try_into().unwrap())
}

#[derive(Debug)]
pub(crate) struct CtOpt<T>(pub(crate) subtle::CtOption<T>);

impl<T> CtOpt<T> {
    pub fn to_result(self) -> Result<T, VoteError> {
        if self.0.is_none().into() {
            return Err(VoteError::InputError);
        }
        Ok(self.0.unwrap())
    }
}

pub(crate) fn as_byte256(h: &[u8]) -> [u8; 32] {
    let mut hh = [0u8; 32];
    hh.copy_from_slice(h);
    hh
}
