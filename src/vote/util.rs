use blake2b_simd::Params;
use ff::FromUniformBytes as _;
use incrementalmerkletree::Hashable as _;
use pasta_curves::Fp;
use zip32::ChildIndex;

use crate::{keys::SpendingKey, tree::MerkleHashOrchard, zip32::ExtendedSpendingKey};

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

/// Derive the secret key corresponding to a given question & answer
/// by index
pub fn derive_question_sk(seed: &[u8], coin_type: u32, question: usize, answer: usize) -> Result<SpendingKey, crate::zip32::Error> {
    let path = &[
        ChildIndex::hardened(32),
        ChildIndex::hardened(coin_type),
        ChildIndex::hardened(question as u32),
        ChildIndex::hardened(answer as u32),
    ];
    ExtendedSpendingKey::from_path(seed, path).map(|esk| esk.sk())
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
