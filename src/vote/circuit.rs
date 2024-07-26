//! The Orchard Action circuit implementation.

use group::{Curve, GroupEncoding};
use halo2_proofs::{
    circuit::{floor_planner, Layouter, Value},
    plonk::{self, Advice, Column, Constraints, Expression, Instance as InstanceColumn, Selector},
    poly::Rotation,
};
use pasta_curves::{arithmetic::CurveAffine, pallas, vesta};

use crate::{
    builder::SpendInfo,
    constants::{
        OrchardCommitDomains, OrchardFixedBases, OrchardFixedBasesFull, OrchardHashDomains,
        MERKLE_DEPTH_ORCHARD,
    },
    keys::{
        CommitIvkRandomness, DiversifiedTransmissionKey, NullifierDerivingKey, SpendValidatingKey,
    },
    note::{
        commitment::{NoteCommitTrapdoor, NoteCommitment},
        nullifier::Nullifier,
        ExtractedNoteCommitment, Note,
    },
    primitives::redpallas::{SpendAuth, VerificationKey},
    spec::NonIdentityPallasPoint,
    tree::{Anchor, MerkleHashOrchard},
    value::{NoteValue, ValueCommitTrapdoor, ValueCommitment},
};
use crate::{
    circuit::{
        commit_ivk::{CommitIvkChip, CommitIvkConfig},
        gadget::{
            add_chip::{AddChip, AddConfig},
            assign_free_advice, commit_ivk, derive_nullifier, note_commit, value_commit_orchard,
        },
        note_commit::{NoteCommitChip, NoteCommitConfig},
    },
    pob::interval::{IntervalChip, IntervalChipConfig},
};
use halo2_gadgets::{
    ecc::{
        chip::{EccChip, EccConfig},
        FixedPoint, NonIdentityPoint, Point, ScalarFixed, ScalarFixedShort, ScalarVar,
    },
    poseidon::{primitives as poseidon, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig},
    sinsemilla::{
        chip::{SinsemillaChip, SinsemillaConfig},
        merkle::{
            chip::{MerkleChip, MerkleConfig},
            MerklePath,
        },
    },
    utilities::lookup_range_check::LookupRangeCheckConfig,
};

use super::proof::Halo2Instance;

/// Size of the Orchard circuit.
const K: u32 = 12;

// Absolute offsets for public inputs.
const ANCHOR: usize = 0;
const CV_NET_X: usize = 1;
const CV_NET_Y: usize = 2;
const DOMAIN_NF: usize = 3;
const RK_X: usize = 4;
const RK_Y: usize = 5;
const CMX: usize = 6;
const NF_ANCHOR: usize = 7;
const DOMAIN: usize = 8;

#[derive(Debug)]
pub struct VotePowerInfo {
    domain_nf: Nullifier,
    nf_start: Nullifier,
    nf_path: crate::tree::MerklePath,
}

impl VotePowerInfo {
    pub fn from_parts(
        domain_nf: Nullifier,
        nf_start: Nullifier,
        nf_path: crate::tree::MerklePath,
    ) -> Self {
        VotePowerInfo {
            domain_nf,
            nf_start,
            nf_path,
        }
    }
}

/// Configuration needed to use the Orchard Action circuit.
#[derive(Clone, Debug)]
pub struct Config {
    primary: Column<InstanceColumn>,
    q_orchard: Selector,
    advices: [Column<Advice>; 10],
    add_config: AddConfig,
    ecc_config: EccConfig<OrchardFixedBases>,
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    merkle_config_1: MerkleConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    merkle_config_2: MerkleConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    sinsemilla_config_1:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    sinsemilla_config_2:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    commit_ivk_config: CommitIvkConfig,
    old_note_commit_config: NoteCommitConfig,
    new_note_commit_config: NoteCommitConfig,
    nf_interval_config: IntervalChipConfig,
}

impl Config {
    pub fn add_chip(&self) -> AddChip {
        AddChip::construct(self.add_config.clone())
    }

    pub fn commit_ivk_chip(&self) -> CommitIvkChip {
        CommitIvkChip::construct(self.commit_ivk_config.clone())
    }

    pub fn ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.ecc_config.clone())
    }

    pub fn sinsemilla_chip_1(
        &self,
    ) -> SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        SinsemillaChip::construct(self.sinsemilla_config_1.clone())
    }

    pub fn sinsemilla_chip_2(
        &self,
    ) -> SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        SinsemillaChip::construct(self.sinsemilla_config_2.clone())
    }

    pub fn merkle_chip_1(
        &self,
    ) -> MerkleChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        MerkleChip::construct(self.merkle_config_1.clone())
    }

    pub fn merkle_chip_2(
        &self,
    ) -> MerkleChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        MerkleChip::construct(self.merkle_config_2.clone())
    }

    pub fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.poseidon_config.clone())
    }

    pub fn note_commit_chip_new(&self) -> NoteCommitChip {
        NoteCommitChip::construct(self.new_note_commit_config.clone())
    }

    pub fn note_commit_chip_old(&self) -> NoteCommitChip {
        NoteCommitChip::construct(self.old_note_commit_config.clone())
    }
}

/// The Orchard Action circuit.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    pub(crate) path: Value<[MerkleHashOrchard; MERKLE_DEPTH_ORCHARD]>,
    pub(crate) pos: Value<u32>,
    pub(crate) g_d_old: Value<NonIdentityPallasPoint>,
    pub(crate) pk_d_old: Value<DiversifiedTransmissionKey>,
    pub(crate) v_old: Value<NoteValue>,
    pub(crate) rho_old: Value<Nullifier>,
    pub(crate) psi_old: Value<pallas::Base>,
    pub(crate) rcm_old: Value<NoteCommitTrapdoor>,
    pub(crate) nf_old: Value<Nullifier>,
    pub(crate) nf_start: Value<Nullifier>,
    pub(crate) nf_path: Value<[MerkleHashOrchard; MERKLE_DEPTH_ORCHARD]>,
    pub(crate) nf_pos: Value<u32>,
    pub(crate) cm_old: Value<NoteCommitment>,
    pub(crate) alpha: Value<pallas::Scalar>,
    pub(crate) ak: Value<SpendValidatingKey>,
    pub(crate) nk: Value<NullifierDerivingKey>,
    pub(crate) rivk: Value<CommitIvkRandomness>,
    pub(crate) g_d_new: Value<NonIdentityPallasPoint>,
    pub(crate) pk_d_new: Value<DiversifiedTransmissionKey>,
    pub(crate) v_new: Value<NoteValue>,
    pub(crate) psi_new: Value<pallas::Base>,
    pub(crate) rcm_new: Value<NoteCommitTrapdoor>,
    pub(crate) rcv: Value<ValueCommitTrapdoor>,
}

impl Circuit {
    /// This constructor is public to enable creation of custom builders.
    /// If you are not creating a custom builder, use [`Builder`] to compose
    /// and authorize a transaction.
    ///
    /// Constructs a `Circuit` from the following components:
    /// - `spend`: [`SpendInfo`] of the note spent in scope of the action
    /// - `output_note`: a note created in scope of the action
    /// - `alpha`: a scalar used for randomization of the action spend validating key
    /// - `rcv`: trapdoor for the action value commitment
    ///
    /// Returns `None` if the `rho` of the `output_note` is not equal
    /// to the nullifier of the spent note.
    ///
    /// [`SpendInfo`]: crate::builder::SpendInfo
    /// [`Builder`]: crate::builder::Builder
    pub fn from_action_context(
        vote_power: VotePowerInfo,
        spend: SpendInfo,
        output_note: Note,
        alpha: pallas::Scalar,
        rcv: ValueCommitTrapdoor,
    ) -> Option<Circuit> {
        (spend.note.nullifier(&spend.fvk) == output_note.rho()).then(|| {
            Self::from_action_context_unchecked(vote_power, spend, output_note, alpha, rcv)
        })
    }

    pub(crate) fn from_action_context_unchecked(
        vote_power: VotePowerInfo,
        spend: SpendInfo,
        output_note: Note,
        alpha: pallas::Scalar,
        rcv: ValueCommitTrapdoor,
    ) -> Circuit {
        let sender_address = spend.note.recipient();
        let rho_old = spend.note.rho();
        let psi_old = spend.note.rseed().psi(&rho_old);
        let rcm_old = spend.note.rseed().rcm(&rho_old);
        let nf_old = spend.note.nullifier(&spend.fvk);

        let rho_new = output_note.rho();
        let psi_new = output_note.rseed().psi(&rho_new);
        let rcm_new = output_note.rseed().rcm(&rho_new);

        Circuit {
            path: Value::known(spend.merkle_path.auth_path()),
            pos: Value::known(spend.merkle_path.position()),
            g_d_old: Value::known(sender_address.g_d()),
            pk_d_old: Value::known(*sender_address.pk_d()),
            v_old: Value::known(spend.note.value()),
            rho_old: Value::known(rho_old),
            psi_old: Value::known(psi_old),
            rcm_old: Value::known(rcm_old),
            nf_old: Value::known(nf_old),
            nf_start: Value::known(vote_power.nf_start),
            nf_path: Value::known(vote_power.nf_path.auth_path()),
            nf_pos: Value::known(vote_power.nf_path.position()),
            cm_old: Value::known(spend.note.commitment()),
            alpha: Value::known(alpha),
            ak: Value::known(spend.fvk.clone().into()),
            nk: Value::known(*spend.fvk.nk()),
            rivk: Value::known(spend.fvk.rivk(spend.scope)),
            g_d_new: Value::known(output_note.recipient().g_d()),
            pk_d_new: Value::known(*output_note.recipient().pk_d()),
            v_new: Value::known(output_note.value()),
            psi_new: Value::known(psi_new),
            rcm_new: Value::known(rcm_new),
            rcv: Value::known(rcv),
        }
    }
}

impl plonk::Circuit<pallas::Base> for Circuit {
    type Config = Config;
    type FloorPlanner = floor_planner::V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut plonk::ConstraintSystem<pallas::Base>) -> Self::Config {
        // Advice columns used in the Orchard circuit.
        let advices = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // Constrain v_old - v_new = magnitude * sign    (https://p.z.cash/ZKS:action-cv-net-integrity?partial).
        // Either v_old = 0, or calculated root = anchor (https://p.z.cash/ZKS:action-merkle-path-validity?partial).
        // Constrain calculated nf_root = nf_anchor
        // Constrain nf_pos even
        let q_orchard = meta.selector();
        meta.create_gate("Orchard circuit checks", |meta| {
            let q_orchard = meta.query_selector(q_orchard);
            let v_old = meta.query_advice(advices[0], Rotation::cur());
            let v_new = meta.query_advice(advices[1], Rotation::cur());
            let magnitude = meta.query_advice(advices[2], Rotation::cur());
            let sign = meta.query_advice(advices[3], Rotation::cur());

            let root = meta.query_advice(advices[4], Rotation::cur());
            let anchor = meta.query_advice(advices[5], Rotation::cur());

            let nf_root = meta.query_advice(advices[6], Rotation::cur());
            let nf_anchor = meta.query_advice(advices[7], Rotation::cur());

            // The constraint "nf_pos is even" checks that nf_start is the beginning
            // of a nf interval (and not the end)
            // However, it is technically not necessary because nf_end is
            // the first item of the Merkle Authorization Path and therefore
            // is the sibling of nf_start
            // If nf_start were the end of the range, nf_end would be the beginning
            // and the range check would fail
            // For clarity, the constraint is still explicitly efforced by the circuit
            let nf_pos = meta.query_advice(advices[8], Rotation::cur());
            let nf_pos_half = meta.query_advice(advices[9], Rotation::cur());

            let nf_in_range = meta.query_advice(advices[0], Rotation::next());
            let one = Expression::Constant(pallas::Base::one());

            Constraints::with_selector(
                q_orchard,
                [
                    (
                        "v_old - v_new = magnitude * sign",
                        v_old.clone() - v_new.clone() - magnitude * sign,
                    ),
                    (
                        "Either v_old = 0, or root = anchor",
                        v_old.clone() * (root - anchor),
                    ),
                    (
                        "Either v_old = 0, or nf root = anchor",
                        v_old.clone() * (nf_root - nf_anchor),
                    ),
                    (
                        "Either v_old = 0, or nf_in_range",
                        v_old.clone() * (one - nf_in_range),
                    ),
                    ("nf_pos is even", nf_pos - nf_pos_half.clone() - nf_pos_half),
                ],
            )
        });

        // Addition of two field elements.
        let add_config = AddChip::configure(meta, advices[7], advices[8], advices[6]);

        // Fixed columns for the Sinsemilla generator lookup table
        let table_idx = meta.lookup_table_column();
        let lookup = (
            table_idx,
            meta.lookup_table_column(),
            meta.lookup_table_column(),
        );

        // Instance column used for public inputs
        let primary = meta.instance_column();
        meta.enable_equality(primary);

        // Permutation over all advice columns.
        for advice in advices.iter() {
            meta.enable_equality(*advice);
        }

        // Poseidon requires four advice columns, while ECC incomplete addition requires
        // six, so we could choose to configure them in parallel. However, we only use a
        // single Poseidon invocation, and we have the rows to accommodate it serially.
        // Instead, we reduce the proof size by sharing fixed columns between the ECC and
        // Poseidon chips.
        let lagrange_coeffs = [
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
            meta.fixed_column(),
        ];
        let rc_a = lagrange_coeffs[2..5].try_into().unwrap();
        let rc_b = lagrange_coeffs[5..8].try_into().unwrap();

        // Also use the first Lagrange coefficient column for loading global constants.
        // It's free real estate :)
        meta.enable_constant(lagrange_coeffs[0]);

        // We have a lot of free space in the right-most advice columns; use one of them
        // for all of our range checks.
        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], table_idx);

        // Configuration for curve point operations.
        // This uses 10 advice columns and spans the whole circuit.
        let ecc_config =
            EccChip::<OrchardFixedBases>::configure(meta, advices, lagrange_coeffs, range_check);

        // Configuration for the Poseidon hash.
        let poseidon_config = PoseidonChip::configure::<poseidon::P128Pow5T3>(
            meta,
            // We place the state columns after the partial_sbox column so that the
            // pad-and-add region can be laid out more efficiently.
            advices[6..9].try_into().unwrap(),
            advices[5],
            rc_a,
            rc_b,
        );

        // Configuration for a Sinsemilla hash instantiation and a
        // Merkle hash instantiation using this Sinsemilla instance.
        // Since the Sinsemilla config uses only 5 advice columns,
        // we can fit two instances side-by-side.
        let (sinsemilla_config_1, merkle_config_1) = {
            let sinsemilla_config_1 = SinsemillaChip::configure(
                meta,
                advices[..5].try_into().unwrap(),
                advices[6],
                lagrange_coeffs[0],
                lookup,
                range_check,
            );
            let merkle_config_1 = MerkleChip::configure(meta, sinsemilla_config_1.clone());

            (sinsemilla_config_1, merkle_config_1)
        };

        // Configuration for a Sinsemilla hash instantiation and a
        // Merkle hash instantiation using this Sinsemilla instance.
        // Since the Sinsemilla config uses only 5 advice columns,
        // we can fit two instances side-by-side.
        let (sinsemilla_config_2, merkle_config_2) = {
            let sinsemilla_config_2 = SinsemillaChip::configure(
                meta,
                advices[5..].try_into().unwrap(),
                advices[7],
                lagrange_coeffs[1],
                lookup,
                range_check,
            );
            let merkle_config_2 = MerkleChip::configure(meta, sinsemilla_config_2.clone());

            (sinsemilla_config_2, merkle_config_2)
        };

        // Configuration to handle decomposition and canonicity checking
        // for CommitIvk.
        let commit_ivk_config = CommitIvkChip::configure(meta, advices);

        // Configuration to handle decomposition and canonicity checking
        // for NoteCommit_old.
        let old_note_commit_config =
            NoteCommitChip::configure(meta, advices, sinsemilla_config_1.clone());

        // Configuration to handle decomposition and canonicity checking
        // for NoteCommit_new.
        let new_note_commit_config =
            NoteCommitChip::configure(meta, advices, sinsemilla_config_2.clone());

        let nf_interval_config =
            IntervalChip::configure(meta, advices[0], advices[1], advices[2], lookup.0);

        Config {
            primary,
            q_orchard,
            advices,
            add_config,
            ecc_config,
            poseidon_config,
            merkle_config_1,
            merkle_config_2,
            sinsemilla_config_1,
            sinsemilla_config_2,
            commit_ivk_config,
            old_note_commit_config,
            new_note_commit_config,
            nf_interval_config,
        }
    }

    #[allow(non_snake_case)]
    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        // Load the Sinsemilla generator lookup table used by the whole circuit.
        SinsemillaChip::load(config.sinsemilla_config_1.clone(), &mut layouter)?;

        // Construct the ECC chip.
        let ecc_chip = config.ecc_chip();

        let nf_interval = IntervalChip::construct(config.nf_interval_config.clone());

        // Witness private inputs that are used across multiple checks.
        let (
            domain,
            psi_old,
            rho_old,
            cm_old,
            g_d_old,
            ak_P,
            nk,
            v_old,
            v_new,
            nf_pos,
            nf_start,
            nf_end,
        ) = {
            // Witness election domain
            let domain = layouter.assign_region(
                || "copy domain",
                |mut region| {
                    region.assign_advice_from_instance(
                        || "instance domain",
                        config.primary,
                        DOMAIN,
                        config.advices[0],
                        0,
                    )
                },
            )?;

            // Witness psi_old
            let psi_old = assign_free_advice(
                layouter.namespace(|| "witness psi_old"),
                config.advices[0],
                self.psi_old,
            )?;

            // Witness rho_old
            let rho_old = assign_free_advice(
                layouter.namespace(|| "witness rho_old"),
                config.advices[0],
                self.rho_old.map(|rho| rho.0),
            )?;

            // Witness cm_old
            let cm_old = Point::new(
                ecc_chip.clone(),
                layouter.namespace(|| "cm_old"),
                self.cm_old.as_ref().map(|cm| cm.inner().to_affine()),
            )?;

            // Witness g_d_old
            let g_d_old = NonIdentityPoint::new(
                ecc_chip.clone(),
                layouter.namespace(|| "gd_old"),
                self.g_d_old.as_ref().map(|gd| gd.to_affine()),
            )?;

            // Witness ak_P.
            let ak_P: Value<pallas::Point> = self.ak.as_ref().map(|ak| ak.into());
            let ak_P = NonIdentityPoint::new(
                ecc_chip.clone(),
                layouter.namespace(|| "witness ak_P"),
                ak_P.map(|ak_P| ak_P.to_affine()),
            )?;

            // Witness nk.
            let nk = assign_free_advice(
                layouter.namespace(|| "witness nk"),
                config.advices[0],
                self.nk.map(|nk| nk.inner()),
            )?;

            // Witness v_old.
            let v_old = assign_free_advice(
                layouter.namespace(|| "witness v_old"),
                config.advices[0],
                self.v_old,
            )?;

            // Witness v_new.
            let v_new = assign_free_advice(
                layouter.namespace(|| "witness v_new"),
                config.advices[0],
                self.v_new,
            )?;

            // Witness nf_pos.
            let nf_pos = assign_free_advice(
                layouter.namespace(|| "witness nf_pos"),
                config.advices[0],
                self.nf_pos.map(|pos| pallas::Base::from(pos as u64)),
            )?;

            // Witness nf_start.
            let nf_start = assign_free_advice(
                layouter.namespace(|| "witness nf_start"),
                config.advices[0],
                self.nf_start.map(|nf| nf.0),
            )?;

            // Witness nf_end as the first level of the Merkle path
            // By construction of the exclusion nullifier MT,
            // Leaves of the tree are pairs of nf_start, nf_end,
            // therefore nf_start is always the left node and
            // nf_end the sibling
            let nf_end = assign_free_advice(
                layouter.namespace(|| "witness nf_end"),
                config.advices[0],
                self.nf_path.map(|p| p[0].0),
            )?;

            (
                domain, psi_old, rho_old, cm_old, g_d_old, ak_P, nk, v_old, v_new, nf_pos,
                nf_start, nf_end,
            )
        };

        // Merkle path validity check (https://p.z.cash/ZKS:action-merkle-path-validity?partial).
        let root = {
            let path = self
                .path
                .map(|typed_path| typed_path.map(|node| node.inner()));
            let merkle_inputs = MerklePath::construct(
                [config.merkle_chip_1(), config.merkle_chip_2()],
                OrchardHashDomains::MerkleCrh,
                self.pos,
                path,
            );
            let leaf = cm_old.extract_p().inner().clone();
            merkle_inputs.calculate_root(layouter.namespace(|| "Merkle path"), leaf)?
        };

        // nullifier Merkle path validity check
        let nf_root = {
            let nf_path = self
                .nf_path
                .map(|typed_path| typed_path.map(|node| node.inner()));
            let merkle_inputs = MerklePath::construct(
                [config.merkle_chip_1(), config.merkle_chip_2()],
                OrchardHashDomains::MerkleCrh,
                self.nf_pos,
                nf_path,
            );
            let leaf = nf_start.clone();
            merkle_inputs.calculate_root(layouter.namespace(|| "Merkle path"), leaf)?
        };

        // Value commitment integrity (https://p.z.cash/ZKS:action-cv-net-integrity?partial).
        let v_net_magnitude_sign = {
            // Witness the magnitude and sign of v_net = v_old - v_new
            let v_net_magnitude_sign = {
                let v_net = self.v_old - self.v_new;
                let magnitude_sign = v_net.map(|v_net| {
                    let (magnitude, sign) = v_net.magnitude_sign();

                    (
                        // magnitude is guaranteed to be an unsigned 64-bit value.
                        // Therefore, we can move it into the base field.
                        pallas::Base::from(magnitude),
                        match sign {
                            crate::value::Sign::Positive => pallas::Base::one(),
                            crate::value::Sign::Negative => -pallas::Base::one(),
                        },
                    )
                });

                let magnitude = assign_free_advice(
                    layouter.namespace(|| "v_net magnitude"),
                    config.advices[9],
                    magnitude_sign.map(|m_s| m_s.0),
                )?;
                let sign = assign_free_advice(
                    layouter.namespace(|| "v_net sign"),
                    config.advices[9],
                    magnitude_sign.map(|m_s| m_s.1),
                )?;
                (magnitude, sign)
            };

            let v_net = ScalarFixedShort::new(
                ecc_chip.clone(),
                layouter.namespace(|| "v_net"),
                v_net_magnitude_sign.clone(),
            )?;
            let rcv = ScalarFixed::new(
                ecc_chip.clone(),
                layouter.namespace(|| "rcv"),
                self.rcv.as_ref().map(|rcv| rcv.inner()),
            )?;

            let cv_net = value_commit_orchard(
                layouter.namespace(|| "cv_net = ValueCommit^Orchard_rcv(v_net)"),
                ecc_chip.clone(),
                v_net,
                rcv,
            )?;

            // Constrain cv_net to equal public input
            layouter.constrain_instance(cv_net.inner().x().cell(), config.primary, CV_NET_X)?;
            layouter.constrain_instance(cv_net.inner().y().cell(), config.primary, CV_NET_Y)?;

            // Return the magnitude and sign so we can use them in the Orchard gate.
            v_net_magnitude_sign
        };

        // Nullifier integrity (https://p.z.cash/ZKS:action-nullifier-integrity).
        let nf_old = {
            let nf_old = derive_nullifier(
                layouter.namespace(|| "nf_old = DeriveNullifier_nk(rho_old, psi_old, cm_old)"),
                config.poseidon_chip(),
                config.add_chip(),
                ecc_chip.clone(),
                rho_old.clone(),
                &psi_old,
                &cm_old,
                nk.clone(),
            )?;

            nf_old
        };

        // Domain Nullifier integrity
        let domain_nf = {
            let domain_nf = crate::circuit::gadget::derive_domain_nullifier(
                layouter.namespace(|| {
                    "domain_nf = DeriveNullifier_domain_nk(rho_old, psi_old, cm_old)"
                }),
                config.poseidon_chip(),
                config.poseidon_chip(),
                config.add_chip(),
                ecc_chip.clone(),
                domain.clone(),
                rho_old.clone(),
                &psi_old,
                &cm_old,
                nk.clone(),
            )?;

            // Constrain nf_old to equal public input
            layouter.constrain_instance(domain_nf.inner().cell(), config.primary, DOMAIN_NF)?;
            domain_nf
        };

        // Spend authority (https://p.z.cash/ZKS:action-spend-authority)
        {
            let alpha =
                ScalarFixed::new(ecc_chip.clone(), layouter.namespace(|| "alpha"), self.alpha)?;

            // alpha_commitment = [alpha] SpendAuthG
            let (alpha_commitment, _) = {
                let spend_auth_g = OrchardFixedBasesFull::SpendAuthG;
                let spend_auth_g = FixedPoint::from_inner(ecc_chip.clone(), spend_auth_g);
                spend_auth_g.mul(layouter.namespace(|| "[alpha] SpendAuthG"), alpha)?
            };

            // [alpha] SpendAuthG + ak_P
            let rk = alpha_commitment.add(layouter.namespace(|| "rk"), &ak_P)?;

            // Constrain rk to equal public input
            layouter.constrain_instance(rk.inner().x().cell(), config.primary, RK_X)?;
            layouter.constrain_instance(rk.inner().y().cell(), config.primary, RK_Y)?;
        }

        // Diversified address integrity (https://p.z.cash/ZKS:action-addr-integrity?partial).
        let pk_d_old = {
            let ivk = {
                let ak = ak_P.extract_p().inner().clone();
                let rivk = ScalarFixed::new(
                    ecc_chip.clone(),
                    layouter.namespace(|| "rivk"),
                    self.rivk.map(|rivk| rivk.inner()),
                )?;

                commit_ivk(
                    config.sinsemilla_chip_1(),
                    ecc_chip.clone(),
                    config.commit_ivk_chip(),
                    layouter.namespace(|| "CommitIvk"),
                    ak,
                    nk,
                    rivk,
                )?
            };
            let ivk =
                ScalarVar::from_base(ecc_chip.clone(), layouter.namespace(|| "ivk"), ivk.inner())?;

            // [ivk] g_d_old
            // The scalar value is passed through and discarded.
            let (derived_pk_d_old, _ivk) =
                g_d_old.mul(layouter.namespace(|| "[ivk] g_d_old"), ivk)?;

            // Constrain derived pk_d_old to equal witnessed pk_d_old
            //
            // This equality constraint is technically superfluous, because the assigned
            // value of `derived_pk_d_old` is an equivalent witness. But it's nice to see
            // an explicit connection between circuit-synthesized values, and explicit
            // prover witnesses. We could get the best of both worlds with a write-on-copy
            // abstraction (https://github.com/zcash/halo2/issues/334).
            let pk_d_old = NonIdentityPoint::new(
                ecc_chip.clone(),
                layouter.namespace(|| "witness pk_d_old"),
                self.pk_d_old.map(|pk_d_old| pk_d_old.inner().to_affine()),
            )?;
            derived_pk_d_old
                .constrain_equal(layouter.namespace(|| "pk_d_old equality"), &pk_d_old)?;

            pk_d_old
        };

        // Old note commitment integrity (https://p.z.cash/ZKS:action-cm-old-integrity?partial).
        {
            let rcm_old = ScalarFixed::new(
                ecc_chip.clone(),
                layouter.namespace(|| "rcm_old"),
                self.rcm_old.as_ref().map(|rcm_old| rcm_old.inner()),
            )?;

            // g★_d || pk★_d || i2lebsp_{64}(v) || i2lebsp_{255}(rho) || i2lebsp_{255}(psi)
            let derived_cm_old = note_commit(
                layouter.namespace(|| {
                    "g★_d || pk★_d || i2lebsp_{64}(v) || i2lebsp_{255}(rho) || i2lebsp_{255}(psi)"
                }),
                config.sinsemilla_chip_1(),
                config.ecc_chip(),
                config.note_commit_chip_old(),
                g_d_old.inner(),
                pk_d_old.inner(),
                v_old.clone(),
                rho_old,
                psi_old,
                rcm_old,
            )?;

            // Constrain derived cm_old to equal witnessed cm_old
            derived_cm_old.constrain_equal(layouter.namespace(|| "cm_old equality"), &cm_old)?;
        }

        // New note commitment integrity (https://p.z.cash/ZKS:action-cmx-new-integrity?partial).
        {
            // Witness g_d_new
            let g_d_new = {
                let g_d_new = self.g_d_new.map(|g_d_new| g_d_new.to_affine());
                NonIdentityPoint::new(
                    ecc_chip.clone(),
                    layouter.namespace(|| "witness g_d_new_star"),
                    g_d_new,
                )?
            };

            // Witness pk_d_new
            let pk_d_new = {
                let pk_d_new = self.pk_d_new.map(|pk_d_new| pk_d_new.inner().to_affine());
                NonIdentityPoint::new(
                    ecc_chip.clone(),
                    layouter.namespace(|| "witness pk_d_new"),
                    pk_d_new,
                )?
            };

            // ρ^new = dnf^old
            let rho_new = domain_nf.inner().clone();

            // Witness psi_new
            let psi_new = assign_free_advice(
                layouter.namespace(|| "witness psi_new"),
                config.advices[0],
                self.psi_new,
            )?;

            let rcm_new = ScalarFixed::new(
                ecc_chip,
                layouter.namespace(|| "rcm_new"),
                self.rcm_new.as_ref().map(|rcm_new| rcm_new.inner()),
            )?;

            // g★_d || pk★_d || i2lebsp_{64}(v) || i2lebsp_{255}(rho) || i2lebsp_{255}(psi)
            let cm_new = note_commit(
                layouter.namespace(|| {
                    "g★_d || pk★_d || i2lebsp_{64}(v) || i2lebsp_{255}(rho) || i2lebsp_{255}(psi)"
                }),
                config.sinsemilla_chip_2(),
                config.ecc_chip(),
                config.note_commit_chip_new(),
                g_d_new.inner(),
                pk_d_new.inner(),
                v_new.clone(),
                rho_new,
                psi_new,
                rcm_new,
            )?;

            let cmx = cm_new.extract_p();

            // Constrain cmx to equal public input
            layouter.constrain_instance(cmx.inner().cell(), config.primary, CMX)?;
        }

        // Range constraint on nf_old
        let nf_in_range = nf_interval.check_in_interval(
            layouter.namespace(|| "nf in [nf_start, nf_end]"),
            nf_old.inner().clone(),
            nf_start,
            nf_end,
        )?;

        // Constrain the remaining Orchard circuit checks.
        layouter.assign_region(
            || "Orchard circuit checks",
            |mut region| {
                v_old.copy_advice(|| "v_old", &mut region, config.advices[0], 0)?;
                v_new.copy_advice(|| "v_new", &mut region, config.advices[1], 0)?;
                v_net_magnitude_sign.0.copy_advice(
                    || "v_net magnitude",
                    &mut region,
                    config.advices[2],
                    0,
                )?;
                v_net_magnitude_sign.1.copy_advice(
                    || "v_net sign",
                    &mut region,
                    config.advices[3],
                    0,
                )?;

                root.copy_advice(|| "calculated root", &mut region, config.advices[4], 0)?;
                region.assign_advice_from_instance(
                    || "pub input anchor",
                    config.primary,
                    ANCHOR,
                    config.advices[5],
                    0,
                )?;

                nf_root.copy_advice(|| "calculated nf_root", &mut region, config.advices[6], 0)?;
                region.assign_advice_from_instance(
                    || "pub input nullifier anchor",
                    config.primary,
                    NF_ANCHOR,
                    config.advices[7],
                    0,
                )?;
                nf_pos.copy_advice(|| "nf_pos", &mut region, config.advices[8], 0)?;
                let nf_pos_half = self.nf_pos.map(|v| pallas::Base::from((v / 2) as u64));
                region.assign_advice(|| "half nf_pos", config.advices[9], 0, || nf_pos_half)?;

                nf_in_range.copy_advice(|| "nf_in_range", &mut region, config.advices[0], 1)?;

                config.q_orchard.enable(&mut region, 0)?;
                Ok(())
            },
        )?;

        Ok(())
    }
}

///
#[derive(Clone, Debug)]
pub struct ElectionDomain(pub pallas::Base);

/// Public inputs to the Orchard Action circuit.
#[derive(Clone, Debug)]
pub struct Instance {
    pub(crate) anchor: Anchor,
    pub(crate) cv_net: ValueCommitment,
    pub(crate) domain_nf: Nullifier,
    pub(crate) rk: VerificationKey<SpendAuth>,
    pub(crate) cmx: ExtractedNoteCommitment,
    pub(crate) domain: ElectionDomain,
    pub(crate) nf_anchor: Anchor,
}

impl Instance {
    /// Constructs an [`Instance`] from its constituent parts.
    ///
    /// This API can be used in combination with [`Proof::verify`] to build verification
    /// pipelines for many proofs, where you don't want to pass around the full bundle.
    /// Use [`Bundle::verify_proof`] instead if you have the full bundle.
    ///
    /// [`Bundle::verify_proof`]: crate::Bundle::verify_proof
    pub fn from_parts(
        anchor: Anchor,
        cv_net: ValueCommitment,
        domain_nf: Nullifier,
        rk: VerificationKey<SpendAuth>,
        cmx: ExtractedNoteCommitment,
        domain: ElectionDomain,
        nf_anchor: Anchor,
    ) -> Self {
        Instance {
            anchor,
            cv_net,
            domain_nf,
            rk,
            cmx,
            domain,
            nf_anchor,
        }
    }
}

impl Halo2Instance for Instance {
    fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        let mut instance = vec![vesta::Scalar::zero(); 9];

        instance[ANCHOR] = self.anchor.inner();
        instance[CV_NET_X] = self.cv_net.x();
        instance[CV_NET_Y] = self.cv_net.y();
        instance[DOMAIN_NF] = self.domain_nf.0;

        let rk = pallas::Point::from_bytes(&self.rk.clone().into())
            .unwrap()
            .to_affine()
            .coordinates()
            .unwrap();

        instance[RK_X] = *rk.x();
        instance[RK_Y] = *rk.y();
        instance[CMX] = self.cmx.inner();

        instance[DOMAIN] = self.domain.0;
        instance[NF_ANCHOR] = self.nf_anchor.inner();

        instance
    }
}

impl super::proof::Statement for Circuit {
    type Circuit = Circuit;
    type Instance = Instance;
}

#[cfg(test)]
mod tests {
    use ff::Field;
    use halo2_proofs::dev::MockProver;
    use pallas::{Base as Fp, Scalar as Fq};
    use rand::{RngCore, SeedableRng};
    use zcash_note_encryption::OUT_CIPHERTEXT_SIZE;

    use crate::{
        keys::{FullViewingKey, Scope, SpendAuthorizingKey, SpendingKey},
        note::{RandomSeed, TransmittedNoteCiphertext},
        note_encryption::OrchardNoteEncryption,
        tree::MerklePath,
        vote::{
            path::{calculate_merkle_paths, make_nf_leaves},
            proof::{ProvingKey, VerifyingKey},
        },
        Action,
    };

    use super::*;

    enum NoteType {
        Ours,
        Spent,
        Others,
    }

    struct VoteNote {
        idx: usize,
        nf: Nullifier,
        nf_start: Nullifier,
        nf_path: MerklePath,
        cmx_path: MerklePath,
    }
    
    fn filter_notes<F>(
        notes: &[Note],
        fvk: &FullViewingKey,
        filter: F,
    ) -> (Vec<Nullifier>, Vec<VoteNote>)
    where
        F: Fn(usize, &Note) -> NoteType,
    {
        let mut ours = vec![];
        let mut nfs = vec![];
        for (idx, n) in notes.iter().enumerate() {
            match filter(idx, n) {
                NoteType::Ours => {
                    ours.push(VoteNote {
                        idx,
                        nf: n.nullifier(fvk),
                        nf_start: Nullifier(Fp::ZERO),
                        nf_path: MerklePath::new(0, [Fp::ZERO; MERKLE_DEPTH_ORCHARD]),
                        cmx_path: MerklePath::new(0, [Fp::ZERO; MERKLE_DEPTH_ORCHARD]),
                    });
                }
                NoteType::Spent => nfs.push(n.nullifier(fvk)),
                NoteType::Others => (),
            }
        }
        (nfs, ours)
    }

    #[test]
    fn f() -> Result<(), Box<dyn std::error::Error>> {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let voter_sk = SpendingKey::random(&mut rng);
        let voter_fvk = FullViewingKey::from(&voter_sk);
        let voter_address = voter_fvk.address_at(0u64, Scope::External);

        let domain = ElectionDomain(Fp::random(&mut rng));
        let mut notes = vec![];
        let mut cmxs = vec![];
        for _ in 0..100 {
            let rho = Nullifier::dummy(&mut rng);
            let rseed = RandomSeed::random(&mut rng, &rho);
            let v = rng.next_u32() % 100;
            let note = Note::from_parts(
                voter_address.clone(),
                NoteValue::from_raw((v as u64) * 100_000_000),
                rho,
                rseed,
            )
            .unwrap();
            notes.push(note);
            let cmx = note.commitment();
            let cmx = ExtractedNoteCommitment::from(cmx);
            let cmx = cmx.to_bytes();
            cmxs.push(cmx);
        }

        const N_CANDIDATES: usize = 2;

        let (mut nfs, mut my_notes) = filter_notes(&notes, &voter_fvk, |idx, _| {
            if idx % 20 == 0 {
                NoteType::Ours
            } else if idx % 3 == 0 {
                NoteType::Spent
            } else {
                NoteType::Others
            }
        });

        let my_pos: Vec<_> = my_notes.iter().map(|n| n.idx as u32).collect();
        let (anchor, cmx_paths) = calculate_merkle_paths(0, &my_pos, &cmxs);
        let anchor = Anchor::from_bytes(anchor).unwrap();
        for (n, cmx_path) in my_notes.iter_mut().zip(cmx_paths.iter()) {
            n.cmx_path = MerklePath::from_parts(
                cmx_path.position,
                cmx_path
                    .path
                    .iter()
                    .map(|h| MerkleHashOrchard::from_bytes(h).unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            );
        }

        let total_value = my_notes
            .iter()
            .map(|n| notes[n.idx].value().inner())
            .sum::<u64>();

        let my_nfs: Vec<_> = my_notes
            .iter()
            .map(|n| notes[n.idx].nullifier(&voter_fvk))
            .collect();

        nfs.sort();
        let (nfs, nfs_pos) = make_nf_leaves(&nfs, &my_nfs);

        let (nf_anchor, nf_paths) = calculate_merkle_paths(0, &nfs_pos, &nfs);
        let nf_anchor = Anchor::from_bytes(nf_anchor).unwrap();
        for (n, nf_path) in my_notes.iter_mut().zip(nf_paths.iter()) {
            n.nf_start = Nullifier::from_bytes(&nf_path.value).unwrap();
            n.nf_path = MerklePath::from_parts(
                nf_path.position,
                nf_path
                    .path
                    .iter()
                    .map(|h| MerkleHashOrchard::from_bytes(h).unwrap())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            );
        }

        let value = total_value / N_CANDIDATES as u64;
        let value = NoteValue::from_raw(value);
        let entropy = [0u8; 32];
        let n_actions = my_notes.len().max(N_CANDIDATES as usize);
        println!("{} {} {}", my_notes.len(), N_CANDIDATES, n_actions);
        let _pk = ProvingKey::<Circuit>::build();
        let _vk = VerifyingKey::<Circuit>::build();
        for c in 0..n_actions {
            println!("{}", c);
            let (dummy_sk, dummy_fvk, dummy_spend) = Note::dummy(&mut rng, None);
            let (spend_sk, spend_fvk, spend) = if c < my_notes.len() {
                (&voter_sk, &voter_fvk, &notes[my_notes[c].idx])
            } else {
                (&dummy_sk, &dummy_fvk, &dummy_spend)
            };
            let domain_nf = spend.nullifier_domain(spend_fvk, domain.0);

            let sk = SpendingKey::from_zip32_seed(&entropy, 133, c as u32).unwrap();
            let fvk = FullViewingKey::from(&sk);
            let candidate = fvk.address_at(0u64, Scope::External);
            let rho = domain_nf;
            let rseed = RandomSeed::random(&mut rng, &rho);
            let output = Note::from_parts(candidate, value, rho, rseed).unwrap();

            let value_net = spend.value() - output.value();
            let rcv = ValueCommitTrapdoor::random(&mut rng);
            let cv_net = ValueCommitment::derive(value_net, rcv.clone());

            let alpha = Fq::random(&mut rng);
            let spk = SpendAuthorizingKey::from(spend_sk);
            let rk = spk.randomize(&alpha);
            let rk = VerificationKey::<SpendAuth>::from(&rk);
            let cmx = output.commitment();
            let cmx = ExtractedNoteCommitment::from(cmx);

            let instance = Instance::from_parts(
                anchor,
                cv_net.clone(),
                domain_nf,
                rk.clone(),
                cmx,
                domain.clone(),
                nf_anchor,
            );

            let encryptor =
                OrchardNoteEncryption::new(None, output.clone(), voter_address, [0u8; 512]);
            let encrypted_note = TransmittedNoteCiphertext {
                epk_bytes: encryptor.epk().to_bytes().0,
                enc_ciphertext: encryptor.encrypt_note_plaintext(),
                out_ciphertext: [0u8; OUT_CIPHERTEXT_SIZE],
            };
            let _action = Action::from_parts(domain_nf, rk, cmx, encrypted_note, cv_net, ());

            let vote_power = if c < my_notes.len() {
                VotePowerInfo {
                    domain_nf,
                    nf_start: my_notes[c].nf_start,
                    nf_path: my_notes[c].nf_path.clone(),
                }
            } else {
                VotePowerInfo {
                    domain_nf,
                    nf_start: Nullifier::dummy(&mut rng),
                    nf_path: MerklePath::dummy(&mut rng),
                }
            };
            let cmx_path = if c < my_notes.len() {
                my_notes[c].cmx_path.clone()
            } else {
                MerklePath::dummy(&mut rng)
            };
            let spend_info = SpendInfo::new(spend_fvk.clone(), spend.clone(), cmx_path).unwrap();
            let output_note = output.clone();

            assert!(spend.nullifier_domain(spend_fvk, domain.0) == output_note.rho());
            let circuit =
                Circuit::from_action_context_unchecked(vote_power, spend_info, output_note, alpha, rcv);
            println!("Create proof");
            let instance = instance.to_halo2_instance();
            let prover = MockProver::run(K, &circuit, vec![instance]).unwrap();
            prover.verify().unwrap();

            // let instances = &[instance];
            // let proof = Proof::<Circuit>::create(&pk, &[circuit], instances, &mut rng).unwrap();
            // proof.verify(&vk, instances).unwrap();
        }
        Ok(())
    }
}
