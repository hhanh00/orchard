use halo2_proofs::{circuit::Chip, plonk::{Advice, Column, ConstraintSystem, Selector}};
use pasta_curves::Fp;

#[derive(Clone, Debug)]
pub struct PaillierChipConfig {
    s: Selector,
    a: Column<Advice>,
}

pub struct PaillierChip {
    config: PaillierChipConfig,
}

impl Chip<Fp> for PaillierChip {
    type Config = PaillierChipConfig;

    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl PaillierChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fp>,
        a: Column<Advice>,
    ) -> PaillierChipConfig {
        meta.enable_equality(a);
        let s = meta.selector();
        // TODO: Create gate
        PaillierChipConfig {
            s,
            a,
        }
    }

    pub fn construct(config: PaillierChipConfig) -> PaillierChip {
        PaillierChip { config }
    }

    // ...
}
