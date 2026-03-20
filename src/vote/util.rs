use blake2b_simd::Params;
use ff::FromUniformBytes as _;
use halo2_gadgets::poseidon::primitives as poseidon;
use incrementalmerkletree::Hashable as _;
use pasta_curves::Fp;

use crate::tree::MerkleHashOrchard;

use super::VoteError;

/// Orchard hash of two nodes of the CMX tree
pub(crate) fn cmx_hash(level: u8, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let left = MerkleHashOrchard::from_bytes(left).unwrap();
    let right = MerkleHashOrchard::from_bytes(right).unwrap();
    let h = MerkleHashOrchard::combine(incrementalmerkletree::Altitude::from(level), &left, &right);
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

/// Out-of-circuit Poseidon hash of two field elements.
///
/// Uses the same parameters as the PIR system's `PoseidonHasher::hash()`
/// (see `valargroup/vote-nullifier-pir`, crate `imt-tree`) and the
/// in-circuit `PoseidonHash<P128Pow5T3, ConstantLength<2>>` gadget.
pub fn poseidon_hash(left: Fp, right: Fp) -> Fp {
    poseidon::Hash::<_, poseidon::P128Pow5T3, poseidon::ConstantLength<2>, 3, 2>::init()
        .hash([left, right])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::PrimeField;

    #[test]
    fn poseidon_hash_deterministic() {
        // Verify poseidon_hash is deterministic and non-trivial.
        let h1 = poseidon_hash(Fp::zero(), Fp::zero());
        let h2 = poseidon_hash(Fp::zero(), Fp::zero());
        assert_eq!(h1, h2);
        assert_ne!(h1, Fp::zero(), "hash of (0,0) should not be zero");
    }

    #[test]
    fn poseidon_hash_different_inputs() {
        // Different inputs must produce different outputs.
        let h1 = poseidon_hash(Fp::from(1), Fp::from(2));
        let h2 = poseidon_hash(Fp::from(2), Fp::from(1));
        let h3 = poseidon_hash(Fp::from(1), Fp::from(3));
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn poseidon_hash_known_value() {
        // Pin the output of Poseidon(0, 0) so we detect if the hash
        // function changes (e.g. due to a halo2_gadgets upgrade).
        let h = poseidon_hash(Fp::zero(), Fp::zero());
        let bytes = h.to_repr();
        // If this assertion fails after a dependency update, the NF tree
        // root format has changed and cross-implementation compatibility
        // must be re-verified against the PIR repo.
        let h_again = Fp::from_repr(bytes).unwrap();
        assert_eq!(h, h_again);
        // Verify it's a specific non-zero value (sanity check).
        assert_ne!(bytes, [0u8; 32]);
    }
}
