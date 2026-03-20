//! IMT non-membership circuit gates and synthesis.
//!
//! Adapted from `valargroup/voting-circuits`
//! (`voting-circuits/src/delegation/imt_circuit.rs`) for the orchard vote module.
//!
//! Contains:
//! - [`IntervalGate`]: interval inclusion check (`low <= nf <= low + width`)
//! - [`ImtNonMembershipConfig`]: bundles swap gate + interval gate
//! - [`synthesize_imt_non_membership`]: orchestrates leaf hash → Merkle path → interval check
//!
//! The Merkle conditional swap gate and path synthesis are provided by
//! [`super::poseidon_merkle`].
//!
//! Source: <https://github.com/valargroup/voting-circuits>

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{self, Advice, Column, Constraints, Expression, Selector},
    poly::Rotation,
};
use pasta_curves::pallas;

use ff::Field;
use halo2_gadgets::{
    ecc::chip::EccConfig,
    poseidon::{
        primitives::{self as poseidon, ConstantLength},
        Hash as PoseidonHash, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig,
    },
};

use crate::circuit::gadget::assign_free_advice;
use crate::constants::OrchardFixedBases;

use super::poseidon_merkle::{MerkleSwapGate, synthesize_poseidon_merkle_path};

/// Depth of the Poseidon-based nullifier range Merkle tree.
/// Matches the PIR tree depth (`valargroup/vote-nullifier-pir`, `imt-tree` crate).
pub const NF_MERKLE_DEPTH: usize = 29;

// ================================================================
// IntervalGate
// ================================================================

/// Interval check gate proving `low <= real_nf <= low + width`.
///
/// **Layout** (2 rows):
/// - Row 0: `advices[0]`=low, `advices[1]`=width, `advices[2]`=real_nf
/// - Row 1: `advices[1]`=x, `advices[2]`=x_shifted (`advices[0]` unused)
///
/// **Constraints**:
/// - `x = real_nf - low`
/// - `x_shifted = 2^250 - width + x - 1`
///
/// The caller must range-check both `x` and `x_shifted` to `[0, 2^250)`.
///
/// NOTE: The 250-bit range checks are only sound when every IMT bracket
/// has width < 2^250. The IMT MUST be initialized with ~17 sentinel
/// nullifiers at multiples of 2^250 before any real nullifiers are
/// inserted. This invariant holds permanently once established, since
/// inserting a nullifier only splits a bracket into two smaller ones.
#[derive(Clone, Debug)]
pub(crate) struct IntervalGate {
    pub(crate) q_interval: Selector,
    advices: [Column<Advice>; 3],
}

impl IntervalGate {
    /// Configures the gate. Uses `advices[0..3]`.
    pub(crate) fn configure(
        meta: &mut plonk::ConstraintSystem<pallas::Base>,
        advices: [Column<Advice>; 3],
    ) -> Self {
        let q_interval = meta.selector();
        meta.create_gate("Interval check", |meta| {
            let q = meta.query_selector(q_interval);
            let low = meta.query_advice(advices[0], Rotation::cur());
            let width = meta.query_advice(advices[1], Rotation::cur());
            let real_nf = meta.query_advice(advices[2], Rotation::cur());
            let x = meta.query_advice(advices[1], Rotation::next());
            let x_shifted = meta.query_advice(advices[2], Rotation::next());

            let two_250 = Expression::Constant(pallas::Base::from(2u64).pow([250, 0, 0, 0]));
            let one = Expression::Constant(pallas::Base::one());

            Constraints::with_selector(
                q,
                [
                    // Lower bound: x = real_nf - low.
                    // Range-checking x to [0, 2^250) proves real_nf >= low.
                    ("x = real_nf - low", x.clone() - (real_nf - low)),
                    // Upper bound: x_shifted = 2^250 - width + x - 1.
                    // If real_nf <= low + width then x <= width, so x_shifted <= 2^250 - 1.
                    // If real_nf > low + width then x > width, so x_shifted >= 2^250 (fails).
                    (
                        "x_shifted = 2^250 - width + x - 1",
                        x_shifted - (two_250 - width + x - one),
                    ),
                ],
            )
        });

        IntervalGate {
            q_interval,
            advices,
        }
    }

    /// Assigns the interval check region. Returns `(x, x_shifted)` for range checking.
    pub(crate) fn assign(
        &self,
        region: &mut halo2_proofs::circuit::Region<'_, pallas::Base>,
        offset: usize,
        low: &AssignedCell<pallas::Base, pallas::Base>,
        width: &AssignedCell<pallas::Base, pallas::Base>,
        real_nf: &AssignedCell<pallas::Base, pallas::Base>,
    ) -> Result<
        (
            AssignedCell<pallas::Base, pallas::Base>,
            AssignedCell<pallas::Base, pallas::Base>,
        ),
        plonk::Error,
    > {
        self.q_interval.enable(region, offset)?;

        low.copy_advice(|| "low", region, self.advices[0], offset)?;
        width.copy_advice(|| "width", region, self.advices[1], offset)?;
        real_nf.copy_advice(|| "real_nf", region, self.advices[2], offset)?;

        let x = region.assign_advice(
            || "x = real_nf - low",
            self.advices[1],
            offset + 1,
            || {
                real_nf
                    .value()
                    .copied()
                    .zip(low.value().copied())
                    .map(|(nf, s)| nf - s)
            },
        )?;

        let two_250 = pallas::Base::from(2u64).pow([250, 0, 0, 0]);
        let x_shifted = region.assign_advice(
            || "x_shifted = 2^250 - width + x - 1",
            self.advices[2],
            offset + 1,
            || {
                width
                    .value()
                    .copied()
                    .zip(x.value().copied())
                    .map(|(w_val, x_val)| two_250 - w_val + x_val - pallas::Base::one())
            },
        )?;

        Ok((x, x_shifted))
    }
}

// ================================================================
// ImtNonMembershipConfig
// ================================================================

/// Bundles the Merkle swap gate, interval gate, and the columns they need.
#[derive(Clone, Debug)]
pub(crate) struct ImtNonMembershipConfig {
    pub(crate) swap_gate: MerkleSwapGate,
    pub(crate) interval_gate: IntervalGate,
    pub(crate) advice_0: Column<Advice>,
}

impl ImtNonMembershipConfig {
    /// Configures both IMT gates. Uses `advices[0..5]` for the swap gate and
    /// `advices[0..3]` for the interval gate (overlapping is fine — different selectors).
    pub(crate) fn configure(
        meta: &mut plonk::ConstraintSystem<pallas::Base>,
        advices: &[Column<Advice>; 10],
    ) -> Self {
        let swap_gate = MerkleSwapGate::configure(
            meta,
            [advices[0], advices[1], advices[2], advices[3], advices[4]],
        );
        let interval_gate = IntervalGate::configure(meta, [advices[0], advices[1], advices[2]]);
        ImtNonMembershipConfig {
            swap_gate,
            interval_gate,
            advice_0: advices[0],
        }
    }
}

// ================================================================
// synthesize_imt_non_membership
// ================================================================

/// Synthesizes the IMT non-membership proof for a single note.
///
/// Orchestrates:
/// 1. Witness low/width
/// 2. Poseidon leaf hash = `Poseidon(low, width)`
/// 3. `NF_MERKLE_DEPTH`-level Merkle path via [`synthesize_poseidon_merkle_path`]
/// 4. Interval check using [`IntervalGate`]
/// 5. Range checks on `x`, `x_shifted` to `[0, 2^250)` (25 limbs × 10 bits)
///
/// Returns `imt_root` which the caller constrains against the public NF anchor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn synthesize_imt_non_membership(
    imt_config: &ImtNonMembershipConfig,
    poseidon_config: &PoseidonConfig<pallas::Base, 3, 2>,
    ecc_config: &EccConfig<OrchardFixedBases>,
    layouter: &mut impl Layouter<pallas::Base>,
    imt_low: Value<pallas::Base>,
    imt_width: Value<pallas::Base>,
    imt_leaf_pos: Value<u32>,
    imt_path: Value<[pallas::Base; NF_MERKLE_DEPTH]>,
    real_nf: &AssignedCell<pallas::Base, pallas::Base>,
    slot: usize,
) -> Result<AssignedCell<pallas::Base, pallas::Base>, plonk::Error> {
    let s = slot;

    let imt_low_cell = assign_free_advice(
        layouter.namespace(|| format!("note {s} imt_low")),
        imt_config.advice_0,
        imt_low,
    )?;

    let imt_width_cell = assign_free_advice(
        layouter.namespace(|| format!("note {s} imt_width")),
        imt_config.advice_0,
        imt_width,
    )?;

    // Compute leaf hash: Poseidon(low, width).
    let leaf_hash = {
        let poseidon_hasher = PoseidonHash::<
            pallas::Base,
            _,
            poseidon::P128Pow5T3,
            ConstantLength<2>,
            3,
            2,
        >::init(
            PoseidonChip::construct(poseidon_config.clone()),
            layouter.namespace(|| format!("note {s} imt leaf hash init")),
        )?;
        poseidon_hasher.hash(
            layouter.namespace(|| format!("note {s} Poseidon(low, width)")),
            [imt_low_cell.clone(), imt_width_cell.clone()],
        )?
    };

    // NF_MERKLE_DEPTH-level Poseidon Merkle path from leaf_hash to imt_root.
    let imt_root = synthesize_poseidon_merkle_path::<NF_MERKLE_DEPTH>(
        &imt_config.swap_gate,
        poseidon_config,
        layouter,
        imt_config.advice_0,
        leaf_hash,
        imt_leaf_pos,
        imt_path,
        &format!("note {s} imt"),
    )?;

    // Interval check: prove low <= real_nf <= low + width.
    let (x, x_shifted) = layouter.assign_region(
        || format!("note {s} interval check"),
        |mut region| {
            imt_config
                .interval_gate
                .assign(&mut region, 0, &imt_low_cell, &imt_width_cell, real_nf)
        },
    )?;

    // Range checks: 25 limbs × 10 bits = 250-bit range.
    // x in [0, 2^250) proves low <= real_nf.
    // x_shifted in [0, 2^250) proves real_nf <= low + width.
    ecc_config.lookup_config.copy_check(
        layouter.namespace(|| format!("note {s} x < 2^250")),
        x,
        25,
        true,
    )?;

    ecc_config.lookup_config.copy_check(
        layouter.namespace(|| format!("note {s} x_shifted < 2^250")),
        x_shifted,
        25,
        true,
    )?;

    Ok(imt_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use halo2_gadgets::{
        ecc::chip::EccChip,
        sinsemilla::chip::SinsemillaChip,
        utilities::lookup_range_check::LookupRangeCheckConfig,
    };
    use halo2_proofs::{
        circuit::{floor_planner, Value},
        dev::MockProver,
        plonk,
    };
    use crate::constants::{OrchardCommitDomains, OrchardFixedBases, OrchardHashDomains};

    /// Minimal circuit that tests IntervalGate + ECC range checks.
    ///
    /// Proves `low <= real_nf <= low + width` using the interval gate
    /// and 250-bit strict range checks via the ECC lookup table.
    #[derive(Clone, Default)]
    struct IntervalTestCircuit {
        low: Value<pallas::Base>,
        width: Value<pallas::Base>,
        real_nf: Value<pallas::Base>,
    }

    use halo2_gadgets::sinsemilla::chip::SinsemillaConfig;

    #[derive(Clone, Debug)]
    struct IntervalTestConfig {
        interval_gate: IntervalGate,
        advice_0: Column<Advice>,
        ecc_config: EccConfig<OrchardFixedBases>,
        sinsemilla_config:
            SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    }

    impl plonk::Circuit<pallas::Base> for IntervalTestCircuit {
        type Config = IntervalTestConfig;
        type FloorPlanner = floor_planner::V1;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut plonk::ConstraintSystem<pallas::Base>) -> IntervalTestConfig {
            let advices: [Column<Advice>; 10] = std::array::from_fn(|_| meta.advice_column());
            for a in &advices {
                meta.enable_equality(*a);
            }

            let table_idx = meta.lookup_table_column();
            let lookup = (
                table_idx,
                meta.lookup_table_column(),
                meta.lookup_table_column(),
            );

            let lagrange_coeffs: [_; 8] = std::array::from_fn(|_| meta.fixed_column());
            meta.enable_constant(lagrange_coeffs[0]);

            let range_check = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);
            let ecc_config =
                EccChip::<OrchardFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);

            // Sinsemilla table is needed for the ECC lookup range checks.
            let sinsemilla_config = SinsemillaChip::<
                OrchardHashDomains,
                OrchardCommitDomains,
                OrchardFixedBases,
            >::configure(
                meta,
                advices[..5].try_into().unwrap(),
                advices[6],
                lagrange_coeffs[0],
                lookup,
                range_check,
            );

            let interval_gate =
                IntervalGate::configure(meta, [advices[0], advices[1], advices[2]]);

            IntervalTestConfig {
                interval_gate,
                advice_0: advices[0],
                ecc_config,
                sinsemilla_config,
            }
        }

        fn synthesize(
            &self,
            config: IntervalTestConfig,
            mut layouter: impl halo2_proofs::circuit::Layouter<pallas::Base>,
        ) -> Result<(), plonk::Error> {
            // Load Sinsemilla lookup table (required for ECC range checks).
            SinsemillaChip::<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>::load(
                config.sinsemilla_config.clone(),
                &mut layouter,
            )?;

            let low = assign_free_advice(
                layouter.namespace(|| "low"),
                config.advice_0,
                self.low,
            )?;
            let width = assign_free_advice(
                layouter.namespace(|| "width"),
                config.advice_0,
                self.width,
            )?;
            let real_nf = assign_free_advice(
                layouter.namespace(|| "real_nf"),
                config.advice_0,
                self.real_nf,
            )?;

            let (x, x_shifted) = layouter.assign_region(
                || "interval check",
                |mut region| {
                    config
                        .interval_gate
                        .assign(&mut region, 0, &low, &width, &real_nf)
                },
            )?;

            // 250-bit strict range checks.
            config.ecc_config.lookup_config.copy_check(
                layouter.namespace(|| "x < 2^250"),
                x,
                25,
                true,
            )?;
            config.ecc_config.lookup_config.copy_check(
                layouter.namespace(|| "x_shifted < 2^250"),
                x_shifted,
                25,
                true,
            )?;

            Ok(())
        }
    }

    #[test]
    fn interval_check_valid() {
        // nf=12 in range [10, 15] → should pass.
        let circuit = IntervalTestCircuit {
            low: Value::known(pallas::Base::from(10)),
            width: Value::known(pallas::Base::from(5)),
            real_nf: Value::known(pallas::Base::from(12)),
        };
        let prover = MockProver::run(15, &circuit, vec![]).unwrap();
        prover.verify().unwrap();
    }

    #[test]
    fn interval_check_at_lower_boundary() {
        // nf == low → x = 0, should pass.
        let circuit = IntervalTestCircuit {
            low: Value::known(pallas::Base::from(100)),
            width: Value::known(pallas::Base::from(50)),
            real_nf: Value::known(pallas::Base::from(100)),
        };
        let prover = MockProver::run(15, &circuit, vec![]).unwrap();
        prover.verify().unwrap();
    }

    #[test]
    fn interval_check_at_upper_boundary() {
        // nf == low + width → x = width, x_shifted = 2^250 - 1, should pass.
        let circuit = IntervalTestCircuit {
            low: Value::known(pallas::Base::from(100)),
            width: Value::known(pallas::Base::from(50)),
            real_nf: Value::known(pallas::Base::from(150)),
        };
        let prover = MockProver::run(15, &circuit, vec![]).unwrap();
        prover.verify().unwrap();
    }

    #[test]
    fn interval_check_above_range_fails() {
        // nf = 16 > 15 = low + width → should fail.
        let circuit = IntervalTestCircuit {
            low: Value::known(pallas::Base::from(10)),
            width: Value::known(pallas::Base::from(5)),
            real_nf: Value::known(pallas::Base::from(16)),
        };
        let prover = MockProver::run(15, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn interval_check_below_range_fails() {
        // nf = 9 < 10 = low → field wraps, x is huge, should fail.
        let circuit = IntervalTestCircuit {
            low: Value::known(pallas::Base::from(10)),
            width: Value::known(pallas::Base::from(5)),
            real_nf: Value::known(pallas::Base::from(9)),
        };
        let prover = MockProver::run(15, &circuit, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
