//! The Orchard Action circuit implementation.

use super::{
    errors::VoteError,
    proof::{ProvingKey, VerifyingKey},
    Hash,
};
use blake2b_simd::Params;
use group::Curve;
use halo2_proofs::{
    circuit::{floor_planner, Layouter, Value},
    plonk::{self, Advice, Column, Error, Instance as InstanceColumn},
};
use pasta_curves::{arithmetic::CurveAffine, pallas, vesta};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{
    circuit::{
        gadget::{
            add_chip::{AddChip, AddConfig},
            assign_free_advice, note_commit, value_commit_orchard,
        },
        note_commit::{NoteCommitChip, NoteCommitConfig},
    },
    primitives::redpallas::{Binding, Signature, SigningKey, VerificationKey},
    value::ValueSum,
    vote::proof::Proof,
};
use crate::{
    constants::{OrchardCommitDomains, OrchardFixedBases, OrchardHashDomains},
    keys::DiversifiedTransmissionKey,
    note::{
        commitment::NoteCommitTrapdoor, nullifier::Nullifier, ExtractedNoteCommitment, RandomSeed,
    },
    spec::NonIdentityPallasPoint,
    value::{NoteValue, ValueCommitTrapdoor, ValueCommitment},
    Address,
};
use halo2_gadgets::{
    ecc::{
        chip::{EccChip, EccConfig},
        NonIdentityPoint, ScalarFixed, ScalarFixedShort,
    },
    poseidon::{primitives as poseidon, Pow5Chip as PoseidonChip, Pow5Config as PoseidonConfig},
    sinsemilla::chip::{SinsemillaChip, SinsemillaConfig},
    utilities::lookup_range_check::LookupRangeCheckConfig,
};

use super::{proof::Halo2Instance, DecryptedVote};

/// Size of the Orchard circuit.
const K: u32 = 12;

// Absolute offsets for public inputs.
const CV_NET_X: usize = 0;
const CV_NET_Y: usize = 1;
const CMX: usize = 2;
const GD_X: usize = 3;
const GD_Y: usize = 4;
const PKD_X: usize = 5;
const PKD_Y: usize = 6;

pub struct VotePowerInfo {
    dnf: Nullifier,
    nf_start: Nullifier,
    nf_path: crate::tree::MerklePath,
}

impl VotePowerInfo {
    fn from_parts(dnf: Nullifier, nf_start: Nullifier, nf_path: crate::tree::MerklePath) -> Self {
        VotePowerInfo {
            dnf,
            nf_start,
            nf_path,
        }
    }
}

/// Configuration needed to use the Orchard Action circuit.
#[derive(Clone, Debug)]
pub struct Config {
    primary: Column<InstanceColumn>,
    advices: [Column<Advice>; 10],
    add_config: AddConfig,
    ecc_config: EccConfig<OrchardFixedBases>,
    poseidon_config: PoseidonConfig<pallas::Base, 3, 2>,
    sinsemilla_config_1:
        SinsemillaConfig<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases>,
    old_note_commit_config: NoteCommitConfig,
}

impl Config {
    pub(crate) fn add_chip(&self) -> AddChip {
        AddChip::construct(self.add_config.clone())
    }

    pub(crate) fn ecc_chip(&self) -> EccChip<OrchardFixedBases> {
        EccChip::construct(self.ecc_config.clone())
    }

    pub(crate) fn sinsemilla_chip_1(
        &self,
    ) -> SinsemillaChip<OrchardHashDomains, OrchardCommitDomains, OrchardFixedBases> {
        SinsemillaChip::construct(self.sinsemilla_config_1.clone())
    }

    pub(crate) fn poseidon_chip(&self) -> PoseidonChip<pallas::Base, 3, 2> {
        PoseidonChip::construct(self.poseidon_config.clone())
    }

    pub(crate) fn note_commit_chip_old(&self) -> NoteCommitChip {
        NoteCommitChip::construct(self.old_note_commit_config.clone())
    }
}

/// The Orchard Action circuit.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    g_d_old: Value<NonIdentityPallasPoint>,
    pk_d_old: Value<DiversifiedTransmissionKey>,
    v_old: Value<NoteValue>,
    rho_old: Value<Nullifier>,
    psi_old: Value<pallas::Base>,
    rcm_old: Value<NoteCommitTrapdoor>,
    rcv: Value<ValueCommitTrapdoor>,
}

impl Circuit {
    fn from_parts(
        g_d_old: NonIdentityPallasPoint,
        pk_d_old: DiversifiedTransmissionKey,
        v_old: NoteValue,
        rho_old: Nullifier,
        rseed_old: RandomSeed,
        rcv: ValueCommitTrapdoor,
    ) -> Circuit {
        let psi_old = rseed_old.psi(&rho_old);
        let rcm_old = rseed_old.rcm(&rho_old);
        Circuit {
            g_d_old: Value::known(g_d_old),
            pk_d_old: Value::known(pk_d_old),
            v_old: Value::known(v_old),
            rho_old: Value::known(rho_old),
            psi_old: Value::known(psi_old),
            rcm_old: Value::known(rcm_old),
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
        let sinsemilla_config_1 = SinsemillaChip::configure(
            meta,
            advices[..5].try_into().unwrap(),
            advices[6],
            lagrange_coeffs[0],
            lookup,
            range_check,
        );

        // Configuration to handle decomposition and canonicity checking
        // for NoteCommit_old.
        let old_note_commit_config =
            NoteCommitChip::configure(meta, advices, sinsemilla_config_1.clone());

        Config {
            primary,
            advices,
            add_config,
            ecc_config,
            poseidon_config,
            sinsemilla_config_1,
            old_note_commit_config,
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

        // Witness private inputs that are used across multiple checks.
        let (psi_old, rho_old, g_d_old, pk_d_old, nv_old, v_old) = {
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

            // Witness g_d_old
            let g_d_old = NonIdentityPoint::new(
                ecc_chip.clone(),
                layouter.namespace(|| "gd_old"),
                self.g_d_old.as_ref().map(|gd| gd.to_affine()),
            )?;

            // Witness pk_d_old
            let pk_d_old = NonIdentityPoint::new(
                ecc_chip.clone(),
                layouter.namespace(|| "pk_d_old"),
                self.pk_d_old.as_ref().map(|pkd| pkd.inner().to_affine()),
            )?;

            // Witness v_old.
            let nv_old = assign_free_advice(
                layouter.namespace(|| "witness v_old"),
                config.advices[0],
                self.v_old,
            )?;

            let v_old = assign_free_advice(
                layouter.namespace(|| "witness v_old"),
                config.advices[0],
                self.v_old.map(|v| pallas::Base::from(v.inner())),
            )?;

            // Constrain cv_net to equal public input
            layouter.constrain_instance(g_d_old.inner().x().cell(), config.primary, GD_X)?;
            layouter.constrain_instance(g_d_old.inner().y().cell(), config.primary, GD_Y)?;
            layouter.constrain_instance(pk_d_old.inner().x().cell(), config.primary, PKD_X)?;
            layouter.constrain_instance(pk_d_old.inner().y().cell(), config.primary, PKD_Y)?;

            (psi_old, rho_old, g_d_old, pk_d_old, nv_old, v_old)
        };

        // Value commitment integrity (https://p.z.cash/ZKS:action-cv-net-integrity?partial).
        {
            let sign = layouter.assign_region(
                || "sign",
                |mut region| {
                    region.assign_advice_from_constant(
                        || "1",
                        config.advices[0],
                        0,
                        pallas::Base::one(),
                    )
                },
            )?;

            let v_net = ScalarFixedShort::new(
                ecc_chip.clone(),
                layouter.namespace(|| "v_net"),
                (v_old, sign),
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
        };

        // Old note commitment integrity (https://p.z.cash/ZKS:action-cm-old-integrity?partial).
        {
            let rcm_old = ScalarFixed::new(
                ecc_chip.clone(),
                layouter.namespace(|| "rcm_old"),
                self.rcm_old.as_ref().map(|rcm_old| rcm_old.inner()),
            )?;

            // g★_d || pk★_d || i2lebsp_{64}(v) || i2lebsp_{255}(rho) || i2lebsp_{255}(psi)
            let cm_old = note_commit(
                layouter.namespace(|| {
                    "g★_d || pk★_d || i2lebsp_{64}(v) || i2lebsp_{255}(rho) || i2lebsp_{255}(psi)"
                }),
                config.sinsemilla_chip_1(),
                config.ecc_chip(),
                config.note_commit_chip_old(),
                g_d_old.inner(),
                pk_d_old.inner(),
                nv_old,
                rho_old,
                psi_old,
                rcm_old,
            )?;

            let cmx = cm_old.extract_p();

            // Constrain cmx to equal public input
            layouter.constrain_instance(cmx.inner().cell(), config.primary, CMX)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ElectionDomain(pub pallas::Base);

/// Public inputs to the Orchard Action circuit.
#[derive(Clone, Debug)]
pub struct Instance {
    cv_net: ValueCommitment,
    cmx: ExtractedNoteCommitment,
    address: Address,
}

impl Instance {
    /// Constructs an [`Instance`] from its constituent parts.
    ///
    /// This API can be used in combination with [`Proof::verify`] to build verification
    /// pipelines for many proofs, where you don't want to pass around the full bundle.
    /// Use [`Bundle::verify_proof`] instead if you have the full bundle.
    ///
    /// [`Bundle::verify_proof`]: crate::Bundle::verify_proof
    fn from_parts(cv_net: ValueCommitment, cmx: ExtractedNoteCommitment, address: Address) -> Self {
        Instance {
            cv_net,
            cmx,
            address,
        }
    }
}

impl Halo2Instance for Instance {
    fn to_halo2_instance(&self) -> Vec<vesta::Scalar> {
        let mut instance = vec![vesta::Scalar::zero(); 7];

        instance[CV_NET_X] = self.cv_net.x();
        instance[CV_NET_Y] = self.cv_net.y();

        let gd = self.address.g_d().to_affine().coordinates().unwrap();
        let pkd = self
            .address
            .pk_d()
            .inner()
            .to_affine()
            .coordinates()
            .unwrap();

        instance[GD_X] = *gd.x();
        instance[GD_Y] = *gd.y();
        instance[PKD_X] = *pkd.x();
        instance[PKD_Y] = *pkd.y();
        instance[CMX] = self.cmx.inner();

        instance
    }
}

impl super::proof::Statement for Circuit {
    type Circuit = Circuit;
    type Instance = Instance;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoteCount {
    pub cv: Hash,
    pub cmx: Hash,
}

///
#[derive(Serialize, Deserialize, Debug)]
pub struct CandidateCount {
    ///
    pub candidate: Vec<u8>,
    ///
    pub total_vote: u64,
    votes: Vec<VoteCount>,
}

impl CandidateCount {
    fn sig_hash(&self) -> Hash {
        let bin_data = serde_cbor::to_vec(&self).unwrap();
        let p = Params::new()
            .hash_length(32)
            .personal(b"Ballot_____Count")
            .hash(&bin_data);
        p.as_bytes().try_into().unwrap()
    }

    fn binding_sign<R: RngCore + CryptoRng>(
        &self,
        rcv: ValueCommitTrapdoor,
        mut rng: R,
    ) -> Vec<u8> {
        let bsk: SigningKey<Binding> = rcv.to_bytes().try_into().unwrap();
        let sig_hash = self.sig_hash();

        let binding_signature = bsk.sign(&mut rng, sig_hash.as_ref());
        let binding_signature: [u8; 64] = (&binding_signature).into();
        binding_signature.to_vec()
    }
}

///
#[derive(Serialize, Deserialize, Debug)]
pub struct CandidateCountEnvelope {
    data: CandidateCount,
    binding_signature: Vec<u8>,
    proofs: Vec<Vec<u8>>,
}

impl CandidateCountEnvelope {
    ///
    pub fn verify(self, vk: &VerifyingKey<Circuit>) -> Result<CandidateCount, VoteError> {
        let mut address = [0u8; 43];
        address.copy_from_slice(&self.data.candidate);
        let address = Address::from_raw_address_bytes(&address).unwrap();
        for (vc, p) in self.data.votes.iter().zip(self.proofs.iter()) {
            let proof = Proof::<Circuit>::new(p.clone());
            let cv = ValueCommitment::from_bytes(&vc.cv).unwrap();
            let cmx = ExtractedNoteCommitment::from_bytes(&vc.cmx).unwrap();
            let instance = Instance::from_parts(cv, cmx, address);
            proof.verify(vk, &[instance])?;
        }

        let sig_hash = self.data.sig_hash();
        let signature: [u8; 64] = self.binding_signature.clone().try_into().unwrap();
        let signature = Signature::<Binding>::from(signature);
        let cv = self
            .data
            .votes
            .iter()
            .map(|vc| ValueCommitment::from_bytes(&vc.cv).unwrap())
            .sum::<ValueCommitment>()
            - ValueCommitment::derive_from_value(self.data.total_vote as i64);
        let pk: VerificationKey<Binding> = cv.to_bytes().try_into().unwrap();
        pk.verify(&sig_hash, &signature)
            .map_err(|_| VoteError::InvalidBindingSignature)?;
        Ok(self.data)
    }
}

///
#[derive(Debug)]
pub struct CountBuilder<'a> {
    pk: &'a ProvingKey<Circuit>,
    vk: &'a VerifyingKey<Circuit>,
    candidate: Address,
    rcv: ValueCommitTrapdoor,
    total_value: u64,
    votes: Vec<VoteCount>,
    proofs: Vec<Vec<u8>>,
}

impl<'a> CountBuilder<'a> {
    ///
    pub fn new(
        candidate: Address,
        pk: &'a ProvingKey<Circuit>,
        vk: &'a VerifyingKey<Circuit>,
    ) -> Self {
        CountBuilder {
            pk,
            vk,
            candidate,
            rcv: ValueCommitTrapdoor::zero(),
            total_value: 0,
            votes: vec![],
            proofs: vec![],
        }
    }

    ///
    pub fn add_vote<R: RngCore + CryptoRng>(
        &mut self,
        vote: &DecryptedVote,
        mut rng: R,
    ) -> Result<(), Error> {
        let rcv = ValueCommitTrapdoor::random(&mut rng);
        let cv = ValueCommitment::derive(ValueSum::from_raw(vote.value as i64), rcv.clone());
        self.rcv = self.rcv.clone() + &rcv;
        self.total_value += vote.value;
        let cmx = vote.cmx;
        self.votes.push(VoteCount {
            cv: cv.to_bytes(),
            cmx: cmx.to_bytes(),
        });
        let address = vote.address;
        assert_eq!(address, self.candidate);
        let instance = Instance::from_parts(cv, cmx, address);
        let circuit = Circuit::from_parts(
            address.g_d(),
            address.pk_d().clone(),
            NoteValue::from_raw(vote.value),
            vote.rho,
            vote.rseed.clone(),
            rcv,
        );

        // let fp_instance = instance.to_halo2_instance();
        // let prover = MockProver::run(K, &circuit, vec![fp_instance]).unwrap();
        // prover.verify().unwrap();

        let proof = Proof::<Circuit>::create(
            self.pk,
            &[circuit],
            std::slice::from_ref(&instance),
            &mut rng,
        )?;

        // proof.verify(self.vk, &[instance])?;
        self.proofs.push(proof.as_ref().to_vec());
        Ok(())
    }

    ///
    pub fn build<R: RngCore + CryptoRng>(self, mut rng: R) -> CandidateCountEnvelope {
        let data: CandidateCount = CandidateCount {
            candidate: self.candidate.to_raw_address_bytes().to_vec(),
            total_vote: self.total_value,
            votes: self.votes,
        };
        let binding_signature: [u8; 64] = data.binding_sign(self.rcv, &mut rng).try_into().unwrap();
        CandidateCountEnvelope {
            data,
            binding_signature: binding_signature.to_vec(),
            proofs: self.proofs,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    use crate::{
        primitives::redpallas::{Signature, VerificationKey},
        vote::encryption::DecryptedVote,
    };

    use super::*;

    #[test]
    fn f() {
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        let vote = DecryptedVote::random(&mut rng);
        let pk = ProvingKey::<Circuit>::build();
        let vk = VerifyingKey::<Circuit>::build();
        let mut builder = CountBuilder::new(vote.address, &pk, &vk);
        builder.add_vote(&vote, &mut rng).unwrap();
        let count = builder.build(&mut rng);

        let c = count.verify(&vk).unwrap();
        println!("{} {}", hex::encode(&c.candidate), c.total_vote);
    }
}
