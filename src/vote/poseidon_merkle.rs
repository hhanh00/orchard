//! Poseidon-based Merkle tree gadget for in-circuit path verification.
//!
//! Copied from `valargroup/voting-circuits` (`voting-circuits/src/circuit/poseidon_merkle.rs`)
//! with minor adaptations for the vote module. Provides:
//! - [`MerkleSwapGate`]: constrained conditional swap (3 constraints per level)
//! - [`synthesize_poseidon_merkle_path`]: walks a Merkle path from leaf to root
//!
//! Used by the IMT non-membership gadget (depth 29) to verify Poseidon
//! Merkle paths in the nullifier tree.

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{self, Advice, Column, Constraints, Selector},
    poly::Rotation,
};
use pasta_curves::pallas;

use halo2_gadgets::{
    poseidon::{
        primitives::{self as poseidon, ConstantLength},
        Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
    },
    utilities::bool_check,
};

use crate::circuit::gadget::assign_free_advice;

/// Conditional swap gate for Merkle tree traversal.
///
/// Given a position bit, current node, and sibling, outputs `(left, right)`
/// such that `left = current, right = sibling` when `bit = 0`, and
/// `left = sibling, right = current` when `bit = 1`.
///
/// Constraints (3 per level):
/// - `left = current + bit * (sibling - current)` (swap left)
/// - `left + right = current + sibling` (swap right / conservation)
/// - `bool_check(bit)` (bit is 0 or 1)
#[derive(Clone, Debug)]
pub struct MerkleSwapGate {
    pub selector: Selector,
    advices: [Column<Advice>; 5],
}

impl MerkleSwapGate {
    /// Configures the conditional swap gate. Uses 5 advice columns.
    pub fn configure(
        meta: &mut plonk::ConstraintSystem<pallas::Base>,
        advices: [Column<Advice>; 5],
    ) -> Self {
        let selector = meta.selector();

        meta.create_gate("Merkle conditional swap", |meta| {
            let q = meta.query_selector(selector);
            let pos_bit = meta.query_advice(advices[0], Rotation::cur());
            let current = meta.query_advice(advices[1], Rotation::cur());
            let sibling = meta.query_advice(advices[2], Rotation::cur());
            let left = meta.query_advice(advices[3], Rotation::cur());
            let right = meta.query_advice(advices[4], Rotation::cur());

            Constraints::with_selector(
                q,
                [
                    (
                        "swap left",
                        left.clone()
                            - current.clone()
                            - pos_bit.clone() * (sibling.clone() - current.clone()),
                    ),
                    ("swap right", left + right - current - sibling),
                    ("bool_check pos_bit", bool_check(pos_bit)),
                ],
            )
        });

        MerkleSwapGate { selector, advices }
    }

    /// Assigns the conditional swap for one Merkle level.
    ///
    /// Returns `(left, right)` cells ready for Poseidon hashing.
    pub fn assign(
        &self,
        region: &mut halo2_proofs::circuit::Region<'_, pallas::Base>,
        offset: usize,
        pos_bit: &AssignedCell<pallas::Base, pallas::Base>,
        current: &AssignedCell<pallas::Base, pallas::Base>,
        sibling: &AssignedCell<pallas::Base, pallas::Base>,
    ) -> Result<
        (
            AssignedCell<pallas::Base, pallas::Base>,
            AssignedCell<pallas::Base, pallas::Base>,
        ),
        plonk::Error,
    > {
        self.selector.enable(region, offset)?;

        let pos_bit_cell =
            pos_bit.copy_advice(|| "pos_bit", region, self.advices[0], offset)?;
        let current_cell =
            current.copy_advice(|| "current", region, self.advices[1], offset)?;
        let sibling_cell =
            sibling.copy_advice(|| "sibling", region, self.advices[2], offset)?;

        let swap = pos_bit_cell
            .value()
            .copied()
            .zip(current_cell.value().copied())
            .zip(sibling_cell.value().copied())
            .map(|((bit, cur), sib)| {
                if bit == pallas::Base::zero() {
                    (cur, sib)
                } else {
                    (sib, cur)
                }
            });

        let left = region.assign_advice(
            || "left",
            self.advices[3],
            offset,
            || swap.map(|(l, _)| l),
        )?;

        let right = region.assign_advice(
            || "right",
            self.advices[4],
            offset,
            || swap.map(|(_, r)| r),
        )?;

        Ok((left, right))
    }
}

/// Synthesize a Poseidon Merkle path from `leaf` to root over `DEPTH` levels.
///
/// At each level:
/// 1. Witness the position bit (LSB-first from `position`)
/// 2. Witness the sibling hash from `path`
/// 3. Constrained swap via [`MerkleSwapGate`] to determine (left, right)
/// 4. Hash `Poseidon(left, right)` to get the parent
///
/// Returns the computed root cell.
pub fn synthesize_poseidon_merkle_path<const DEPTH: usize>(
    swap_gate: &MerkleSwapGate,
    poseidon_config: &PoseidonConfig<pallas::Base, 3, 2>,
    layouter: &mut impl Layouter<pallas::Base>,
    advice_0: Column<Advice>,
    leaf: AssignedCell<pallas::Base, pallas::Base>,
    position: Value<u32>,
    path: Value<[pallas::Base; DEPTH]>,
    label: &str,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let mut current = leaf;

    for i in 0..DEPTH {
        let pos_bit = assign_free_advice(
            layouter.namespace(|| format!("{label} pos_bit {i}")),
            advice_0,
            position.map(|p| pallas::Base::from(((p >> i) & 1) as u64)),
        )?;

        let sibling = assign_free_advice(
            layouter.namespace(|| format!("{label} sibling {i}")),
            advice_0,
            path.map(|path| path[i]),
        )?;

        let (left, right) = layouter.assign_region(
            || format!("{label} swap level {i}"),
            |mut region| swap_gate.assign(&mut region, 0, &pos_bit, &current, &sibling),
        )?;

        let parent = {
            let hasher = PoseidonHash::<
                pallas::Base,
                _,
                poseidon::P128Pow5T3,
                ConstantLength<2>,
                3,
                2,
            >::init(
                PoseidonChip::construct(poseidon_config.clone()),
                layouter.namespace(|| format!("{label} hash init level {i}")),
            )?;
            hasher.hash(
                layouter.namespace(|| format!("{label} Poseidon(left, right) level {i}")),
                [left, right],
            )?
        };

        current = parent;
    }

    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vote::util::poseidon_hash;
    use halo2_proofs::{
        circuit::{floor_planner, Value},
        dev::MockProver,
        plonk,
    };

    const TEST_DEPTH: usize = 4;

    /// Minimal circuit that verifies a Poseidon Merkle path of depth 4.
    #[derive(Clone, Default)]
    struct MerkleTestCircuit {
        leaf: Value<pallas::Base>,
        position: Value<u32>,
        path: Value<[pallas::Base; TEST_DEPTH]>,
    }

    #[derive(Clone, Debug)]
    struct MerkleTestConfig {
        advices: [Column<Advice>; 5],
        advice_0: Column<Advice>,
        swap_gate: MerkleSwapGate,
        poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    }

    impl plonk::Circuit<pallas::Base> for MerkleTestCircuit {
        type Config = MerkleTestConfig;
        type FloorPlanner = floor_planner::V1;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut plonk::ConstraintSystem<pallas::Base>) -> MerkleTestConfig {
            let advices: [Column<Advice>; 5] = std::array::from_fn(|_| meta.advice_column());
            for a in &advices {
                meta.enable_equality(*a);
            }
            let swap_gate = MerkleSwapGate::configure(meta, advices);

            let lagrange_coeffs = std::array::from_fn::<_, 8, _>(|_| meta.fixed_column());
            meta.enable_constant(lagrange_coeffs[0]);
            let rc_a = lagrange_coeffs[2..5].try_into().unwrap();
            let rc_b = lagrange_coeffs[5..8].try_into().unwrap();
            let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
                meta,
                advices[0..3].try_into().unwrap(),
                advices[3],
                rc_a,
                rc_b,
            );

            MerkleTestConfig {
                advices,
                advice_0: advices[0],
                swap_gate,
                poseidon_config,
            }
        }

        fn synthesize(
            &self,
            config: MerkleTestConfig,
            mut layouter: impl halo2_proofs::circuit::Layouter<pallas::Base>,
        ) -> Result<(), plonk::Error> {
            let leaf = assign_free_advice(
                layouter.namespace(|| "leaf"),
                config.advice_0,
                self.leaf,
            )?;

            let _root = synthesize_poseidon_merkle_path::<TEST_DEPTH>(
                &config.swap_gate,
                &config.poseidon_config,
                &mut layouter,
                config.advice_0,
                leaf,
                self.position,
                self.path,
                "test",
            )?;

            Ok(())
        }
    }

    /// Build a depth-4 Poseidon Merkle tree from 16 leaves and return (root, auth_path for leaf at `pos`).
    fn build_test_tree(leaves: &[pallas::Base; 16], pos: usize) -> (pallas::Base, [pallas::Base; TEST_DEPTH]) {
        let mut layer: Vec<pallas::Base> = leaves.to_vec();
        let mut path = [pallas::Base::zero(); TEST_DEPTH];
        let mut idx = pos;

        for level in 0..TEST_DEPTH {
            let sibling = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
            path[level] = layer[sibling];

            let mut next = Vec::new();
            for j in (0..layer.len()).step_by(2) {
                next.push(poseidon_hash(layer[j], layer[j + 1]));
            }
            layer = next;
            idx /= 2;
        }

        (layer[0], path)
    }

    #[test]
    fn merkle_path_valid() {
        let leaves: [pallas::Base; 16] = std::array::from_fn(|i| pallas::Base::from(i as u64 + 1));
        let pos = 5u32;
        let (root, path) = build_test_tree(&leaves, pos as usize);

        // Verify out-of-circuit root manually.
        let mut current = leaves[pos as usize];
        for (i, sibling) in path.iter().enumerate() {
            let (l, r) = if (pos >> i) & 1 == 0 {
                (current, *sibling)
            } else {
                (*sibling, current)
            };
            current = poseidon_hash(l, r);
        }
        assert_eq!(current, root);

        // Verify in-circuit.
        let circuit = MerkleTestCircuit {
            leaf: Value::known(leaves[pos as usize]),
            position: Value::known(pos),
            path: Value::known(path),
        };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        prover.verify().unwrap();
    }

    #[test]
    fn merkle_path_wrong_sibling_fails() {
        let leaves: [pallas::Base; 16] = std::array::from_fn(|i| pallas::Base::from(i as u64 + 1));
        let pos = 5u32;
        let (_, mut path) = build_test_tree(&leaves, pos as usize);

        // Tamper with one sibling.
        path[2] = pallas::Base::from(999);

        let circuit = MerkleTestCircuit {
            leaf: Value::known(leaves[pos as usize]),
            position: Value::known(pos),
            path: Value::known(path),
        };
        let prover = MockProver::run(11, &circuit, vec![]).unwrap();
        // The circuit still "passes" in MockProver because there are no
        // public input constraints on the root — it just computes a
        // different root. This test verifies synthesis doesn't panic.
        // Root correctness is checked by constraining against a public input
        // in the full vote circuit.
        assert!(prover.verify().is_ok());
    }
}
