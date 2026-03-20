//! Range-check chip for PIR-compatible nullifier non-membership proofs.
//!
//! Proves `value - low <= width` using two soft range checks:
//!   1. `offset = value - low` fits in M bits (proves value >= low)
//!   2. `diff = width - offset` fits in M bits (proves offset <= width)
//!
//! Returns a boolean (1 = in range, 0 = not in range) so the caller can
//! conditionally enforce via `v_old * (1 - success) = 0`.

use halo2_gadgets::utilities::lookup_range_check::LookupRangeCheckConfig;
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Constraints, Error, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use pasta_curves::pallas::Base as Fp;

use super::logical::{IsZeroChip, IsZeroChipConfig};

// K * NUM_WORDS = 252 bits — max range width supported.
// With sentinel nullifiers, all widths < 2^250, well within this limit.
const K: usize = 9;
const NUM_WORDS: usize = 28;

/// Configuration for the range-check chip.
#[derive(Clone, Debug)]
pub struct IntervalChipConfig {
    s_range: Selector,
    s_and: Selector,
    a: Column<Advice>,
    b: Column<Advice>,
    c: Column<Advice>,
    table_idx: TableColumn,
    range_config: LookupRangeCheckConfig<Fp, K>,
    is_zero_config: IsZeroChipConfig,
}

/// Chip that proves `value - low <= width` (PIR-compatible range check).
#[derive(Clone, Debug)]
pub struct IntervalChip {
    config: IntervalChipConfig,
}

impl Chip<Fp> for IntervalChip {
    type Config = IntervalChipConfig;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl IntervalChip {
    /// Configure the range-check chip.
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        a: Column<Advice>,
        b: Column<Advice>,
        c: Column<Advice>,
        table_idx: TableColumn,
    ) -> IntervalChipConfig {
        meta.enable_equality(a);
        meta.enable_equality(b);
        meta.enable_equality(c);
        let s_range = meta.selector();
        let s_and = meta.selector();

        // Gate layout:
        //   Row 0: value  | low    | width
        //   Row 1: offset | diff   | (unused)
        // Constraints:
        //   offset = value - low
        //   diff = width - offset
        meta.create_gate("range_check", |meta| {
            let s_range = meta.query_selector(s_range);
            let value = meta.query_advice(a, Rotation::cur());
            let low = meta.query_advice(b, Rotation::cur());
            let width = meta.query_advice(c, Rotation::cur());
            let offset = meta.query_advice(a, Rotation::next());
            let diff = meta.query_advice(b, Rotation::next());
            Constraints::with_selector(
                s_range,
                [
                    offset.clone() - (value - low),
                    diff - (width - offset),
                ],
            )
        });

        meta.create_gate("and", |meta| {
            let s_and = meta.query_selector(s_and);
            let a = meta.query_advice(a, Rotation::cur());
            let b = meta.query_advice(b, Rotation::cur());
            let c = meta.query_advice(c, Rotation::cur());
            Constraints::with_selector(s_and, [c - a * b])
        });

        let range_config = LookupRangeCheckConfig::configure(meta, b, table_idx);
        let is_zero_config = IsZeroChip::configure(meta, a, b, c);
        IntervalChipConfig {
            s_range,
            s_and,
            a,
            b,
            c,
            table_idx,
            range_config,
            is_zero_config,
        }
    }

    /// Construct the chip from its configuration.
    pub fn construct(config: IntervalChipConfig) -> IntervalChip {
        IntervalChip { config }
    }

    /// Load the K-bit lookup table.
    pub(crate) fn load(
        &self,
        mut layouter: impl Layouter<Fp>,
        table_idx: TableColumn,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "table_idx",
            |mut table| {
                for index in 0..(1 << K) {
                    table.assign_cell(
                        || "table_idx",
                        table_idx,
                        index,
                        || Value::known(Fp::from(index as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Prove that `value - low <= width`.
    ///
    /// Returns a boolean cell: 1 if in range, 0 if not.
    /// Uses soft range checks (strict=false) so that out-of-range values
    /// produce a 0 result instead of hard-failing the proof. This allows
    /// the caller to conditionally enforce via `v_old * (1 - result) = 0`.
    pub fn check_in_range(
        &self,
        mut layouter: impl Layouter<Fp>,
        value: AssignedCell<Fp, Fp>,
        low: AssignedCell<Fp, Fp>,
        width: AssignedCell<Fp, Fp>,
    ) -> Result<AssignedCell<Fp, Fp>, Error> {
        let config = &self.config;
        let is_zero_chip = IsZeroChip::construct(config.is_zero_config.clone());

        let (offset, diff) = layouter.assign_region(
            || "check range",
            |mut region| {
                config.s_range.enable(&mut region, 0)?;
                value.copy_advice(|| "value", &mut region, config.a, 0)?;
                low.copy_advice(|| "low", &mut region, config.b, 0)?;
                width.copy_advice(|| "width", &mut region, config.c, 0)?;

                let offset_val = value.value().zip(low.value()).map(|(v, l)| v - l);
                let diff_val = width.value().zip(offset_val).map(|(w, o)| w - o);

                let offset = region.assign_advice(|| "offset", config.a, 1, || offset_val)?;
                let diff = region.assign_advice(|| "diff", config.b, 1, || diff_val)?;
                Ok((offset, diff))
            },
        )?;

        // Soft range check on offset: does it fit in M = K * NUM_WORDS bits?
        let c1 = config.range_config.copy_check(
            layouter.namespace(|| "offset < 2^M"),
            offset,
            NUM_WORDS,
            false,
        )?;
        let c1 = c1.last().unwrap();
        let c1 = is_zero_chip.is_zero(layouter.namespace(|| "c1 <- offset fits"), c1.clone())?;

        // Soft range check on diff: does it fit in M bits?
        let c2 = config.range_config.copy_check(
            layouter.namespace(|| "diff < 2^M"),
            diff,
            NUM_WORDS,
            false,
        )?;
        let c2 = c2.last().unwrap();
        let c2 = is_zero_chip.is_zero(layouter.namespace(|| "c2 <- diff fits"), c2.clone())?;

        // success = c1 AND c2
        let success = layouter.assign_region(
            || "c1 && c2",
            |mut region| {
                config.s_and.enable(&mut region, 0)?;
                c1.copy_advice(|| "c1", &mut region, config.a, 0)?;
                c2.copy_advice(|| "c2", &mut region, config.b, 0)?;
                let c1_c2 = c1.value().zip(c2.value()).map(|(c1, c2)| c1 * c2);
                let success = region.assign_advice(|| "c1 and c2", config.c, 0, || c1_c2)?;
                Ok(success)
            },
        )?;

        Ok(success)
    }
}

// ---------------------------------------------------------------------------
// Test circuit
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct TestCircuitConfig {
    interval_config: IntervalChipConfig,
}

#[derive(Default)]
struct TestCircuit {
    value: Value<Fp>,
    low: Value<Fp>,
    width: Value<Fp>,
}

impl Circuit<Fp> for TestCircuit {
    type Config = TestCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> TestCircuitConfig {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let table_idx = meta.lookup_table_column();
        let f = meta.fixed_column();
        meta.enable_constant(f);
        let interval_config = IntervalChip::configure(meta, a, b, c, table_idx);
        TestCircuitConfig { interval_config }
    }

    fn synthesize(
        &self,
        config: TestCircuitConfig,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let interval_chip = IntervalChip::construct(config.interval_config.clone());
        interval_chip.load(
            layouter.namespace(|| "load lookup table"),
            config.interval_config.table_idx,
        )?;
        let (value, low, width) = layouter.assign_region(
            || "load witnesses",
            |mut region| {
                let value =
                    region.assign_advice(|| "value", config.interval_config.a, 0, || self.value)?;
                let low =
                    region.assign_advice(|| "low", config.interval_config.b, 0, || self.low)?;
                let width =
                    region.assign_advice(|| "width", config.interval_config.c, 0, || self.width)?;
                Ok((value, low, width))
            },
        )?;
        let success = interval_chip.check_in_range(
            layouter.namespace(|| "check in range"),
            value,
            low,
            width,
        )?;
        // Constrain success == 1 (must be in range for circuit to be satisfied)
        layouter.assign_region(
            || "constrain success",
            |mut region| {
                let one = region.assign_advice_from_constant(
                    || "one",
                    config.interval_config.a,
                    0,
                    Fp::one(),
                )?;
                region.constrain_equal(success.cell(), one.cell())
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_in_range() -> Result<(), Error> {
        // value=12, low=10, width=2 → offset=2, diff=0 → both fit ✓
        let test = TestCircuit {
            value: Value::known(Fp::from(12)),
            low: Value::known(Fp::from(10)),
            width: Value::known(Fp::from(2)),
        };
        let prover = halo2_proofs::dev::MockProver::run(11, &test, vec![])?;
        prover.verify().unwrap();
        Ok(())
    }

    #[test]
    fn test_value_at_low_bound() -> Result<(), Error> {
        // value=10, low=10, width=5 → offset=0, diff=5 → both fit ✓
        let test = TestCircuit {
            value: Value::known(Fp::from(10)),
            low: Value::known(Fp::from(10)),
            width: Value::known(Fp::from(5)),
        };
        let prover = halo2_proofs::dev::MockProver::run(11, &test, vec![])?;
        prover.verify().unwrap();
        Ok(())
    }

    #[test]
    fn test_value_at_high_bound() -> Result<(), Error> {
        // value=15, low=10, width=5 → offset=5, diff=0 → both fit ✓
        let test = TestCircuit {
            value: Value::known(Fp::from(15)),
            low: Value::known(Fp::from(10)),
            width: Value::known(Fp::from(5)),
        };
        let prover = halo2_proofs::dev::MockProver::run(11, &test, vec![])?;
        prover.verify().unwrap();
        Ok(())
    }

    #[test]
    fn test_value_above_range() {
        // value=16, low=10, width=5 → offset=6, diff=5-6 wraps → diff doesn't fit
        let test = TestCircuit {
            value: Value::known(Fp::from(16)),
            low: Value::known(Fp::from(10)),
            width: Value::known(Fp::from(5)),
        };
        let prover = halo2_proofs::dev::MockProver::run(11, &test, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }

    #[test]
    fn test_value_below_range() {
        // value=9, low=10, width=5 → offset=9-10 wraps to huge → doesn't fit
        let test = TestCircuit {
            value: Value::known(Fp::from(9)),
            low: Value::known(Fp::from(10)),
            width: Value::known(Fp::from(5)),
        };
        let prover = halo2_proofs::dev::MockProver::run(11, &test, vec![]).unwrap();
        assert!(prover.verify().is_err());
    }
}
