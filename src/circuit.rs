//! The Orchard Action circuit implementation.
//!
//! This module defines the common structures, traits and implementations for the
//! Orchard Action circuit, supporting both the standard ("Vanilla") and ZSA variations.

use alloc::vec::Vec;

use group::{Curve, GroupEncoding};
use halo2_proofs::{
    circuit::{floor_planner, Layouter, Value},
    plonk::{
        self, Advice, BatchVerifier, Column, Instance as InstanceColumn, Selector, SingleVerifier,
    },
    transcript::{Blake2bRead, Blake2bWrite},
};
use pasta_curves::{arithmetic::CurveAffine, pallas, vesta};
use rand::RngCore;

use crate::{
    builder::SpendInfo,
    bundle::Flags,
    circuit::{
        commit_ivk::CommitIvkConfig, gadget::add_chip::AddConfig, note_commit::NoteCommitConfig,
    },
    constants::{
        OrchardCommitDomains, OrchardFixedBases, OrchardHashDomains, MERKLE_DEPTH_ORCHARD,
    },
    keys::{
        CommitIvkRandomness, DiversifiedTransmissionKey, NullifierDerivingKey, SpendValidatingKey,
    },
    note::{
        commitment::{NoteCommitTrapdoor, NoteCommitment},
        nullifier::Nullifier,
        AssetBase, ExtractedNoteCommitment, Note, Rho,
    },
    primitives::redpallas::{SpendAuth, VerificationKey},
    spec::NonIdentityPallasPoint,
    tree::{Anchor, MerkleHashOrchard},
    value::{NoteValue, ValueCommitTrapdoor, ValueCommitment},
};
use halo2_gadgets::{
    ecc::chip::EccConfig,
    poseidon::Pow5Config as PoseidonConfig,
    sinsemilla::{chip::SinsemillaConfig, merkle::chip::MerkleConfig},
    utilities::lookup_range_check::PallasLookupRangeCheck,
};

mod circuit_vanilla;
mod circuit_zsa;

pub(in crate::circuit) mod commit_ivk;
pub(in crate::circuit) mod derive_nullifier;
pub(in crate::circuit) mod gadget;
pub(in crate::circuit) mod note_commit;
pub(in crate::circuit) mod orchard_sinsemilla_chip;
pub(in crate::circuit) mod value_commit_orchard;

pub use crate::Proof;

/// Size of the Orchard circuit.
const K: u32 = 11;

// Absolute offsets for public inputs.
const ANCHOR: usize = 0;
const CV_NET_X: usize = 1;
const CV_NET_Y: usize = 2;
const NF_OLD: usize = 3;
const RK_X: usize = 4;
const RK_Y: usize = 5;
const CMX: usize = 6;
const ENABLE_SPEND: usize = 7;
const ENABLE_OUTPUT: usize = 8;
const ENABLE_ZSA: usize = 9;

/// Configuration needed to use the Orchard Action circuit.
#[derive(Clone, Debug)]
pub struct Config<Lookup: PallasLookupRangeCheck> {
    primary: Column<InstanceColumn>,
    q_orchard: Selector,
    advices: [Column<Advice>; 10],
    add_config: AddConfig,
    ecc_config: EccConfig<OrchardFixedBases, Lookup>,
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    merkle_config_1:
        MerkleConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases, Lookup>,
    merkle_config_2:
        MerkleConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases, Lookup>,
    sinsemilla_config_1:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases, Lookup>,
    sinsemilla_config_2:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases, Lookup>,
    commit_ivk_config: CommitIvkConfig,
    old_note_commit_config: NoteCommitConfig<Lookup>,
    new_note_commit_config: NoteCommitConfig<Lookup>,
}

/// The `OrchardCircuit` trait defines an interface for different implementations of the PLONK circuit
/// for the different Orchard protocol flavors (Vanilla and ZSA). It serves as a bridge between
/// plonk::Circuit interfaces and specific requirements of the Orchard protocol's variations.
pub trait OrchardCircuit: Sized + Default {
    /// Substitution for Config type of plonk::Circuit trait
    type Config: Clone;

    /// Wrapper for configure function of plonk::Circuit trait
    fn configure(meta: &mut plonk::ConstraintSystem<pallas::Base>) -> Self::Config;

    /// Wrapper for configure function of plonk::Circuit trait
    fn synthesize(
        circuit: &Witnesses,
        config: Self::Config,
        layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error>;

    /// Builds the ZSA-specific witnesses for the circuit.
    /// For OrchardVanilla circuits, it should return `Value::unknown()`.
    fn build_additional_zsa_witnesses(
        psi_nf: pallas::Base,
        asset: AssetBase,
        split_flag: bool,
    ) -> Value<AdditionalZsaWitnesses>;
}

impl<C: OrchardCircuit> plonk::Circuit<pallas::Base> for Circuit<C> {
    type Config = C::Config;
    type FloorPlanner = floor_planner::V1;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut plonk::ConstraintSystem<pallas::Base>) -> Self::Config {
        C::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), plonk::Error> {
        C::synthesize(&self.witnesses, config, layouter)
    }
}

/// The Orchard Action circuit.
#[derive(Clone, Debug, Default)]
pub struct Circuit<C: OrchardCircuit> {
    pub(crate) witnesses: Witnesses,
    pub(crate) phantom: core::marker::PhantomData<C>,
}

/// The ZSA-specific witnesses.
#[derive(Clone, Debug)]
pub struct AdditionalZsaWitnesses {
    pub(crate) psi_nf: pallas::Base,
    pub(crate) asset: AssetBase,
    pub(crate) split_flag: bool,
}

fn unpack(
    zsa_values: Value<AdditionalZsaWitnesses>,
) -> (Value<pallas::Base>, Value<AssetBase>, Value<bool>) {
    (
        zsa_values.clone().map(|values| values.psi_nf),
        zsa_values.clone().map(|values| values.asset),
        zsa_values.map(|values| values.split_flag),
    )
}

/// The Orchard Action witnesses
#[derive(Clone, Debug, Default)]
pub struct Witnesses {
    pub(crate) path: Value<[MerkleHashOrchard; MERKLE_DEPTH_ORCHARD]>,
    pub(crate) pos: Value<u32>,
    pub(crate) g_d_old: Value<NonIdentityPallasPoint>,
    pub(crate) pk_d_old: Value<DiversifiedTransmissionKey>,
    pub(crate) v_old: Value<NoteValue>,
    pub(crate) rho_old: Value<Rho>,
    pub(crate) psi_old: Value<pallas::Base>,
    pub(crate) rcm_old: Value<NoteCommitTrapdoor>,
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

    // The ZSA-specific witnesses.
    // For OrchardVanilla circuits, this field should be initialized to `Value::unknown()`.
    pub(crate) additional_zsa_witnesses: Value<AdditionalZsaWitnesses>,
}

impl Witnesses {
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
    pub fn from_action_context<C: OrchardCircuit>(
        spend: SpendInfo,
        output_note: Note,
        alpha: pallas::Scalar,
        rcv: ValueCommitTrapdoor,
    ) -> Option<Self> {
        (Rho::from_nf_old(spend.note.nullifier(&spend.fvk)) == output_note.rho())
            .then(|| Self::from_action_context_unchecked::<C>(spend, output_note, alpha, rcv))
    }

    pub(crate) fn from_action_context_unchecked<C: OrchardCircuit>(
        spend: SpendInfo,
        output_note: Note,
        alpha: pallas::Scalar,
        rcv: ValueCommitTrapdoor,
    ) -> Self {
        let sender_address = spend.note.recipient();
        let rho_old = spend.note.rho();
        let psi_old = spend.note.rseed().psi(&rho_old);
        let rcm_old = spend.note.rseed().rcm(&rho_old);

        let rho_new = output_note.rho();
        let psi_new = output_note.rseed().psi(&rho_new);
        let rcm_new = output_note.rseed().rcm(&rho_new);

        let nf_rseed = spend.note.rseed_split_note().unwrap_or(*spend.note.rseed());
        let psi_nf = nf_rseed.psi(&rho_old);
        let additional_zsa_witnesses =
            C::build_additional_zsa_witnesses(psi_nf, spend.note.asset(), spend.split_flag);

        Witnesses {
            path: Value::known(spend.merkle_path.auth_path()),
            pos: Value::known(spend.merkle_path.position()),
            g_d_old: Value::known(sender_address.g_d()),
            pk_d_old: Value::known(*sender_address.pk_d()),
            v_old: Value::known(spend.note.value()),
            rho_old: Value::known(rho_old),
            psi_old: Value::known(psi_old),
            rcm_old: Value::known(rcm_old),
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

            additional_zsa_witnesses,
        }
    }
}

/// The verifying key for the Orchard Action circuit.
///
/// In the current type system, this could be a verifying key for either
/// the original Orchard Action circuit, or the OrchardZSA circuit.
#[derive(Debug)]
pub struct VerifyingKey {
    pub(crate) params: halo2_proofs::poly::commitment::Params<vesta::Affine>,
    pub(crate) vk: plonk::VerifyingKey<vesta::Affine>,
}

impl VerifyingKey {
    /// Builds the verifying key.
    pub fn build<C: OrchardCircuit>() -> Self {
        let params = halo2_proofs::poly::commitment::Params::new(K);
        let circuit: Circuit<C> = Default::default();

        let vk = plonk::keygen_vk(&params, &circuit).unwrap();

        VerifyingKey { params, vk }
    }
}

/// The proving key for the Orchard Action circuit.
///
/// In the current type system, this could be a proving key for either
/// the original Orchard Action circuit, or the OrchardZSA circuit.
#[derive(Debug)]
pub struct ProvingKey {
    params: halo2_proofs::poly::commitment::Params<vesta::Affine>,
    pk: plonk::ProvingKey<vesta::Affine>,
}

impl ProvingKey {
    /// Builds the proving key.
    pub fn build<C: OrchardCircuit>() -> Self {
        let params = halo2_proofs::poly::commitment::Params::new(K);
        let circuit: Circuit<C> = Default::default();

        let vk = plonk::keygen_vk(&params, &circuit).unwrap();
        let pk = plonk::keygen_pk(&params, vk, &circuit).unwrap();

        ProvingKey { params, pk }
    }
}

/// Public inputs to the Orchard Action circuit.
///
/// The `enable_zsa` field was introduced with the ZSA feature; it did not exist before.
/// In vanilla Orchard, `enable_zsa` is always false, so this method always appends a zero to the
/// instance vector. Since halo2_proofs pads instance values with zero, old proofs (without this
/// extra entry) and new proofs behave identically.
#[derive(Clone, Debug)]
pub struct Instance {
    pub(crate) anchor: Anchor,
    pub(crate) cv_net: ValueCommitment,
    pub(crate) nf_old: Nullifier,
    pub(crate) rk: VerificationKey<SpendAuth>,
    pub(crate) cmx: ExtractedNoteCommitment,
    pub(crate) enable_spend: bool,
    pub(crate) enable_output: bool,
    pub(crate) enable_zsa: bool,
}

impl Instance {
    /// Constructs an [`Instance`] from its constituent parts.
    ///
    /// This API can be used in combination with [`Proof::verify`] to build verification
    /// pipelines for many proofs, where you don't want to pass around the full bundle.
    /// Use [`Bundle::verify_proof`] instead if you have the full bundle.
    ///
    /// Returns `None` if `rk` is the identity [`pasta_curves::pallas::Point`].
    /// zcashd v6.12.1 and Zebra 4.3.1 both added a consensus rule rejecting
    /// transactions whose Orchard actions have an identity `rk`; the Zcash
    /// protocol specification will be updated to match, and this crate
    /// aligns with that rule.
    ///
    /// See:
    /// - <https://zodl.com/zcashd-zebra-april-2026-disclosure/>
    /// - <https://zfnd.org/zebra-4-3-1-critical-security-fixes-dockerized-mining-and-ci-hardening/>
    ///
    /// [`Bundle::verify_proof`]: crate::Bundle::verify_proof
    pub fn from_parts(
        anchor: Anchor,
        cv_net: ValueCommitment,
        nf_old: Nullifier,
        rk: VerificationKey<SpendAuth>,
        cmx: ExtractedNoteCommitment,
        flags: Flags,
    ) -> Option<Self> {
        (!rk.is_identity()).then_some(Instance {
            anchor,
            cv_net,
            nf_old,
            rk,
            cmx,
            enable_spend: flags.spends_enabled(),
            enable_output: flags.outputs_enabled(),
            enable_zsa: flags.zsa_enabled(),
        })
    }

    /// Returns the Merkle tree anchor of this instance.

    /// Returns the Merkle tree anchor of this instance.
    pub(crate) fn anchor(&self) -> &Anchor {
        &self.anchor
    }

    /// Returns the commitment to the net value of this instance.
    pub(crate) fn cv_net(&self) -> &ValueCommitment {
        &self.cv_net
    }

    /// Returns the nullifier of the note being spent by this instance.
    pub(crate) fn nf_old(&self) -> &Nullifier {
        &self.nf_old
    }

    /// Returns the randomized verification key of this instance.
    pub(crate) fn rk(&self) -> &VerificationKey<SpendAuth> {
        &self.rk
    }

    /// Returns the commitment to the new note being created by this instance.
    pub(crate) fn cmx(&self) -> &ExtractedNoteCommitment {
        &self.cmx
    }

    /// Returns whether spends are enabled for this instance.
    pub(crate) fn enable_spend(&self) -> bool {
        self.enable_spend
    }

    /// Returns whether outputs are enabled for this instance.
    pub(crate) fn enable_output(&self) -> bool {
        self.enable_output
    }

    /// Returns whether ZSA functionality is enabled for this instance.
    pub(crate) fn enable_zsa(&self) -> bool {
        self.enable_zsa
    }

    /// Note: Before the ZSA feature was introduced, this method returned a 9-element instance slice.
    /// With ZSA, it now returns 10 elements, the last one corresponding to `enable_zsa`.
    /// In vanilla Orchard, `enable_zsa` is always false, so this extra element is always zero.
    /// Since halo2_proofs pads instance values with zero, old proofs (without this element)
    /// and new proofs behave identically.
    fn to_halo2_instance(&self) -> [[vesta::Scalar; 10]; 1] {
        let mut instance = [vesta::Scalar::zero(); 10];

        instance[ANCHOR] = self.anchor.inner();
        instance[CV_NET_X] = self.cv_net.x();
        instance[CV_NET_Y] = self.cv_net.y();
        instance[NF_OLD] = self.nf_old.inner();

        let rk = pallas::Point::from_bytes(&self.rk.clone().into())
            .expect("the cached byte encoding of a VerificationKey<_> is canonical")
            .to_affine()
            .coordinates()
            .expect("rk is non-identity by construction");

        instance[RK_X] = *rk.x();
        instance[RK_Y] = *rk.y();
        instance[CMX] = self.cmx.inner();
        instance[ENABLE_SPEND] = vesta::Scalar::from(u64::from(self.enable_spend));
        instance[ENABLE_OUTPUT] = vesta::Scalar::from(u64::from(self.enable_output));
        instance[ENABLE_ZSA] = vesta::Scalar::from(u64::from(self.enable_zsa));

        [instance]
    }
}

impl Proof {
    /// Creates a proof for the given circuits and instances.
    pub fn create<C: OrchardCircuit>(
        pk: &ProvingKey,
        circuits: &[Circuit<C>],
        instances: &[Instance],
        mut rng: impl RngCore,
    ) -> Result<Self, plonk::Error> {
        let instances: Vec<_> = instances.iter().map(|i| i.to_halo2_instance()).collect();
        let instances: Vec<Vec<_>> = instances
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instances: Vec<_> = instances.iter().map(|i| &i[..]).collect();

        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &pk.params,
            &pk.pk,
            circuits,
            &instances,
            &mut rng,
            &mut transcript,
        )?;
        Ok(Proof(transcript.finalize()))
    }

    /// Verifies this proof with the given instances.
    pub fn verify(&self, vk: &VerifyingKey, instances: &[Instance]) -> Result<(), plonk::Error> {
        let instances: Vec<_> = instances.iter().map(|i| i.to_halo2_instance()).collect();
        let instances: Vec<Vec<_>> = instances
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instances: Vec<_> = instances.iter().map(|i| &i[..]).collect();

        let strategy = SingleVerifier::new(&vk.params);
        let mut transcript = Blake2bRead::init(&self.0[..]);
        plonk::verify_proof(&vk.params, &vk.vk, strategy, &instances, &mut transcript)
    }

    /// Adds this proof to the given batch for verification with the given instances.
    ///
    /// Use this API if you want more control over how proof batches are processed. If you
    /// just want to batch-validate Orchard bundles, use [`bundle::BatchValidator`].
    ///
    /// [`bundle::BatchValidator`]: crate::bundle::BatchValidator
    pub fn add_to_batch(&self, batch: &mut BatchVerifier<vesta::Affine>, instances: Vec<Instance>) {
        let instances = instances
            .iter()
            .map(|i| {
                i.to_halo2_instance()
                    .into_iter()
                    .map(|c| c.into_iter().collect())
                    .collect()
            })
            .collect();

        batch.add_proof(instances, self.0.clone());
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;
    use core::iter;

    use ff::Field;
    use halo2_proofs::{circuit::Value, dev::MockProver};
    use pasta_curves::pallas;
    use rand::{rngs::OsRng, RngCore};

    use super::{Circuit, Instance, Proof, ProvingKey, VerifyingKey, Witnesses, K};
    use crate::{
        flavor::OrchardVanilla,
        keys::SpendValidatingKey,
        note::{AssetBase, Note, Rho},
        tree::MerklePath,
        value::{ValueCommitTrapdoor, ValueCommitment},
    };

    fn generate_circuit_instance<R: RngCore>(mut rng: R) -> (Circuit<OrchardVanilla>, Instance) {
        let (_, fvk, spent_note) = Note::dummy(&mut rng, None);

        let sender_address = spent_note.recipient();
        let nk = *fvk.nk();
        let rivk = fvk.rivk(fvk.scope_for_address(&spent_note.recipient()).unwrap());
        let nf_old = spent_note.nullifier(&fvk);
        let rho = Rho::from_nf_old(nf_old);
        let ak: SpendValidatingKey = fvk.into();
        let alpha = pallas::Scalar::random(&mut rng);
        let rk = ak.randomize(&alpha);

        let (_, _, output_note) = Note::dummy(&mut rng, Some(rho));
        let cmx = output_note.commitment().into();

        let value = spent_note.value() - output_note.value();
        let rcv = ValueCommitTrapdoor::random(&mut rng);
        let cv_net = ValueCommitment::derive(value, rcv.clone(), AssetBase::zatoshi());

        let path = MerklePath::dummy(&mut rng);
        let anchor = path.root(spent_note.commitment().into());

        (
            Circuit {
                witnesses: Witnesses {
                    path: Value::known(path.auth_path()),
                    pos: Value::known(path.position()),
                    g_d_old: Value::known(sender_address.g_d()),
                    pk_d_old: Value::known(*sender_address.pk_d()),
                    v_old: Value::known(spent_note.value()),
                    rho_old: Value::known(spent_note.rho()),
                    psi_old: Value::known(spent_note.rseed().psi(&spent_note.rho())),
                    rcm_old: Value::known(spent_note.rseed().rcm(&spent_note.rho())),
                    cm_old: Value::known(spent_note.commitment()),
                    alpha: Value::known(alpha),
                    ak: Value::known(ak),
                    nk: Value::known(nk),
                    rivk: Value::known(rivk),
                    g_d_new: Value::known(output_note.recipient().g_d()),
                    pk_d_new: Value::known(*output_note.recipient().pk_d()),
                    v_new: Value::known(output_note.value()),
                    psi_new: Value::known(output_note.rseed().psi(&output_note.rho())),
                    rcm_new: Value::known(output_note.rseed().rcm(&output_note.rho())),
                    rcv: Value::known(rcv),
                    additional_zsa_witnesses: Value::unknown(),
                },
                phantom: core::marker::PhantomData,
            },
            Instance {
                anchor,
                cv_net,
                nf_old,
                rk,
                cmx,
                enable_spend: true,
                enable_output: true,
                enable_zsa: false,
            },
        )
    }

    // TODO: recast as a proptest
    #[test]
    fn round_trip() {
        let mut rng = OsRng;

        let (circuits, instances): (Vec<_>, Vec<_>) = iter::once(())
            .map(|()| generate_circuit_instance(&mut rng))
            .unzip();

        let vk = VerifyingKey::build::<OrchardVanilla>();

        // Test that the pinned verification key (representing the circuit)
        // is as expected.
        {
            assert_eq!(
                format!("{:#?}\n", vk.vk.pinned()),
                include_str!("circuit/circuit_description_vanilla").replace("\r\n", "\n")
            );
        }

        // Test that the proof size is as expected.
        let expected_proof_size = {
            let circuit_cost =
                halo2_proofs::dev::CircuitCost::<pasta_curves::vesta::Point, _>::measure(
                    K,
                    &circuits[0],
                );
            assert_eq!(usize::from(circuit_cost.proof_size(1)), 4992);
            assert_eq!(usize::from(circuit_cost.proof_size(2)), 7264);
            usize::from(circuit_cost.proof_size(instances.len()))
        };

        for (circuit, instance) in circuits.iter().zip(instances.iter()) {
            assert_eq!(
                MockProver::run(
                    K,
                    circuit,
                    instance
                        .to_halo2_instance()
                        .iter()
                        .map(|p| p.to_vec())
                        .collect()
                )
                .unwrap()
                .verify(),
                Ok(())
            );
        }

        let pk = ProvingKey::build::<OrchardVanilla>();
        let proof = Proof::create(&pk, &circuits, &instances, &mut rng).unwrap();
        assert!(proof.verify(&vk, &instances).is_ok());
        assert_eq!(proof.0.len(), expected_proof_size);
    }

    #[test]
    fn serialized_proof_test_case() {
        use std::io::{Read, Write};

        let vk = VerifyingKey::build::<OrchardVanilla>();

        fn write_test_case<W: Write>(
            mut w: W,
            instance: &Instance,
            proof: &Proof,
        ) -> std::io::Result<()> {
            w.write_all(&instance.anchor().to_bytes())?;
            w.write_all(&instance.cv_net().to_bytes())?;
            w.write_all(&instance.nf_old().to_bytes())?;
            w.write_all(&<[u8; 32]>::from(instance.rk()))?;
            w.write_all(&instance.cmx().to_bytes())?;
            w.write_all(&[
                u8::from(instance.enable_spend()),
                u8::from(instance.enable_output()),
                u8::from(instance.enable_zsa()),
            ])?;
            w.write_all(proof.as_ref())?;
            Ok(())
        }

        fn read_test_case<R: Read>(mut r: R) -> std::io::Result<(Instance, Proof)> {
            let read_32_bytes = |r: &mut R| {
                let mut ret = [0u8; 32];
                r.read_exact(&mut ret).unwrap();
                ret
            };
            let read_bool = |r: &mut R| {
                let mut byte = [0u8; 1];
                r.read_exact(&mut byte).unwrap();
                match byte {
                    [0] => false,
                    [1] => true,
                    _ => panic!("Unexpected non-boolean byte"),
                }
            };

            let anchor = crate::Anchor::from_bytes(read_32_bytes(&mut r)).unwrap();
            let cv_net = ValueCommitment::from_bytes(&read_32_bytes(&mut r)).unwrap();
            let nf_old = crate::note::Nullifier::from_bytes(&read_32_bytes(&mut r)).unwrap();
            let rk = read_32_bytes(&mut r).try_into().unwrap();
            let cmx =
                crate::note::ExtractedNoteCommitment::from_bytes(&read_32_bytes(&mut r)).unwrap();
            let enable_spend = read_bool(&mut r);
            let enable_output = read_bool(&mut r);
            let enable_zsa = read_bool(&mut r);
            let instance =
                Instance::from_parts(
                    anchor,
                    cv_net,
                    nf_old,
                    rk,
                    cmx,
                    crate::bundle::Flags::from_parts(enable_spend, enable_output, enable_zsa),
                )
                    .expect("test vectors were generated with non-identity rk");

            let mut proof_bytes = vec![];
            r.read_to_end(&mut proof_bytes)?;
            let proof = Proof::new(proof_bytes);

            Ok((instance, proof))
        }

        if std::env::var_os("ORCHARD_CIRCUIT_TEST_GENERATE_NEW_PROOF").is_some() {
            let create_proof = || -> std::io::Result<()> {
                let mut rng = OsRng;

                let (circuit, instance) = generate_circuit_instance(OsRng);
                let instances = core::slice::from_ref(&instance);

                let pk = ProvingKey::build::<OrchardVanilla>();
                let proof = Proof::create(&pk, &[circuit], instances, &mut rng).unwrap();
                assert!(proof.verify(&vk, instances).is_ok());

                let file = std::fs::File::create("circuit_proof_test_case.bin")?;
                write_test_case(file, &instance, &proof)
            };
            create_proof().expect("should be able to write new proof");
        }

        // Parse the hardcoded proof test case.
        let (instance, proof) = {
            let test_case_bytes = std::fs::read("circuit_proof_test_case.bin")
                .expect("circuit_proof_test_case.bin not found. Run with ORCHARD_CIRCUIT_TEST_GENERATE_NEW_PROOF=1 to generate it.");
            read_test_case(&test_case_bytes[..]).expect("proof must be valid")
        };
        assert_eq!(proof.0.len(), 4992);

        assert!(proof.verify(&vk, &[instance]).is_ok());
    }

    #[cfg(feature = "dev-graph")]
    #[test]
    fn print_action_circuit() {
        use plotters::prelude::*;

        let root = BitMapBackend::new("action-circuit-layout.png", (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("Orchard Action Circuit", ("sans-serif", 60))
            .unwrap();

        let circuit = Circuit::<OrchardVanilla> {
            witnesses: Witnesses {
                path: Value::unknown(),
                pos: Value::unknown(),
                g_d_old: Value::unknown(),
                pk_d_old: Value::unknown(),
                v_old: Value::unknown(),
                rho_old: Value::unknown(),
                psi_old: Value::unknown(),
                rcm_old: Value::unknown(),
                cm_old: Value::unknown(),
                alpha: Value::unknown(),
                ak: Value::unknown(),
                nk: Value::unknown(),
                rivk: Value::unknown(),
                g_d_new: Value::unknown(),
                pk_d_new: Value::unknown(),
                v_new: Value::unknown(),
                psi_new: Value::unknown(),
                rcm_new: Value::unknown(),
                rcv: Value::unknown(),
                additional_zsa_witnesses: Value::unknown(),
            },
            phantom: core::marker::PhantomData,
        };
        halo2_proofs::dev::CircuitLayout::default()
            .show_labels(false)
            .view_height(0..(1 << 11))
            .render(K, &circuit, &root)
            .unwrap();
    }

    mod from_parts_rk_identity {
        use ff::{Field as _, PrimeField as _};
        use pasta_curves::pallas;

        use super::super::Instance;
        use crate::{
            note::{ExtractedNoteCommitment, Nullifier},
            primitives::redpallas::{self, SpendAuth},
            tree::Anchor,
            value::{ValueCommitTrapdoor, ValueCommitment, ValueSum},
        };

        /// Non-rk fields for `Instance`. Distinct non-zero patterns avoid
        /// accidental overlap with sentinel values. See the analogous helper
        /// in `src/action.rs` for notes on which of these fields have
        /// consensus-level validity checks elsewhere in the pipeline.
        fn dummy_other_fields() -> (Anchor, ValueCommitment, Nullifier, ExtractedNoteCommitment) {
            let anchor = Anchor::from_bytes([6u8; 32]).unwrap();
            let cv_net =
                ValueCommitment::derive(ValueSum::from_raw(42), ValueCommitTrapdoor::ZERO, crate::note::AssetBase::zatoshi());
            let nf_old = Nullifier::from_bytes(&[1u8; 32]).unwrap();
            let cmx = ExtractedNoteCommitment::from_bytes(&[2u8; 32]).unwrap();
            (anchor, cv_net, nf_old, cmx)
        }

        fn identity_rk() -> redpallas::VerificationKey<SpendAuth> {
            redpallas::VerificationKey::<SpendAuth>::try_from([0u8; 32])
                .expect("plain redpallas accepts the identity encoding")
        }

        fn non_identity_rk() -> redpallas::VerificationKey<SpendAuth> {
            let ask_bytes: [u8; 32] = pallas::Scalar::ONE.to_repr().into();
            let ask = redpallas::SigningKey::<SpendAuth>::try_from(ask_bytes)
                .expect("1 is a valid scalar");
            (&ask).into()
        }

        #[test]
        fn rejects_identity_rk() {
            let (anchor, cv_net, nf_old, cmx) = dummy_other_fields();
            let result =
                Instance::from_parts(
                    anchor,
                    cv_net,
                    nf_old,
                    identity_rk(),
                    cmx,
                    crate::bundle::Flags::from_parts(true, true, false),
                );
            assert!(result.is_none());
        }

        #[test]
        fn accepts_non_identity_rk() {
            let (anchor, cv_net, nf_old, cmx) = dummy_other_fields();
            let rk = non_identity_rk();
            let instance =
                Instance::from_parts(
                    anchor,
                    cv_net,
                    nf_old,
                    rk.clone(),
                    cmx,
                    crate::bundle::Flags::from_parts(true, true, false),
                )
                    .expect("non-identity rk must be accepted");
            assert_eq!(instance.rk(), &rk);
        }
    }
}
