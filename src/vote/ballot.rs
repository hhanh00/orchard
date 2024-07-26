use blake2b_simd::Params;
use ff::{Field, PrimeField};
use halo2_proofs::{dev::MockProver, plonk::Error};
use pasta_curves::pallas::{Base as Fp, Scalar as Fq};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use zcash_note_encryption::{try_compact_note_decryption, ShieldedOutput, OUT_CIPHERTEXT_SIZE};

use super::{count::CountBuilder, path::calculate_merkle_paths, proof::Halo2Instance, DecryptedVote, EncryptedVote, Hash};
use crate::{
    builder::SpendInfo,
    constants::MERKLE_DEPTH_ORCHARD,
    keys::{FullViewingKey, PreparedIncomingViewingKey, Scope, SpendAuthorizingKey, SpendingKey},
    note::{ExtractedNoteCommitment, Nullifier, RandomSeed, TransmittedNoteCiphertext},
    note_encryption::{OrchardDomain, OrchardNoteEncryption},
    primitives::redpallas::{Binding, Signature, SigningKey, SpendAuth, VerificationKey},
    tree::{MerkleHashOrchard, MerklePath},
    value::{NoteValue, ValueCommitTrapdoor, ValueCommitment},
    vote::{
        circuit::{Circuit, ElectionDomain, Instance, VotePowerInfo},
        proof::{Proof, ProvingKey, VerifyingKey},
    },
    Address, Anchor, Note,
};

#[derive(Debug)]
struct VoteInput {
    sk: SpendingKey,
    fvk: FullViewingKey,
    pos: u32,
    note: Note,
    nf_start: Nullifier,
    nf_pos: u32,
    nf_path: MerklePath,
    cmx_path: MerklePath,
}

impl VoteInput {
    pub fn from_parts(pos: u32, sk: SpendingKey, note: Note) -> Self {
        let fvk = FullViewingKey::from(&sk);
        VoteInput {
            sk,
            fvk,
            pos,
            note,
            nf_pos: 0,
            nf_start: Nullifier(Fp::zero()),
            nf_path: MerklePath::new(0, [Fp::zero(); MERKLE_DEPTH_ORCHARD]),
            cmx_path: MerklePath::new(0, [Fp::zero(); MERKLE_DEPTH_ORCHARD]),
        }
    }
}

#[derive(Debug)]
struct VoteOutput(Address, u64);

#[derive(Clone, Serialize, Deserialize, Debug)]
pub(crate) struct BallotAction {
    pub(crate) cv: [u8; 32],
    pub(crate) rk: [u8; 32],
    pub(crate) nf: [u8; 32],
    pub(crate) cmx: [u8; 32],
    pub(crate) epk: [u8; 32],
    #[serde(with = "hex")]
    pub(crate) enc: [u8; 52],
}

///
#[derive(Serialize, Deserialize, Debug)]
pub struct BallotData {
    pub(crate) actions: Vec<BallotAction>,
}

impl BallotData {
    pub fn count_candidate(&self, idx_candidate: usize, fvk: &FullViewingKey) -> DecryptedVote {
        let ba = &self.actions[idx_candidate];
        let ev = EncryptedVote(ba.clone());
        let dv = ev.decrypt(fvk);
        dv
    }

    pub fn sig_hash(&self) -> Hash {
        let bin_data = serde_cbor::to_vec(&self).unwrap();
        let p = Params::new()
            .hash_length(32)
            .personal(b"Ballot______Data")
            .hash(&bin_data);
        p.as_bytes().try_into().unwrap()
    }

    pub fn binding_sign<R: RngCore + CryptoRng>(
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
pub struct BallotEnvelope {
    ///
    pub data: BallotData,
    pub(crate) binding_signature: Vec<u8>,
    pub(crate) proofs: Vec<Vec<u8>>,
}

impl BallotEnvelope {
    ///
    pub fn build<R: RngCore + CryptoRng>(
        data: BallotData,
        rcv: ValueCommitTrapdoor,
        proofs: Vec<Vec<u8>>,
        mut rng: R,
    ) -> Self {
        let binding_signature: [u8; 64] = data.binding_sign(rcv, &mut rng).try_into().unwrap();
        BallotEnvelope {
            data,
            binding_signature: binding_signature.to_vec(),
            proofs,
        }
    }

    ///
    pub fn verify(
        self,
        vk: &VerifyingKey<Circuit>,
        domain: ElectionDomain,
        anchor: Anchor,
        nf_anchor: Anchor,
    ) -> Result<BallotData, Error> {
        let data = self.data;
        let proofs = &self.proofs;
        for (p, action) in proofs.iter().zip(data.actions.iter()) {
            let proof = Proof::<Circuit>::new(p.clone());
            let cv_net = ValueCommitment::from_bytes(&action.cv).unwrap();
            let domain_nf = Nullifier::from_bytes(&action.nf).unwrap();
            let rk: VerificationKey<SpendAuth> = action.rk.try_into().unwrap();
            let cmx = ExtractedNoteCommitment::from_bytes(&action.cmx).unwrap();
            let instance = Instance::from_parts(
                anchor,
                cv_net.clone(),
                domain_nf,
                rk.clone(),
                cmx,
                domain.clone(),
                nf_anchor,
            );

            proof.verify(vk, &[instance])?;
        }

        let signature: [u8; 64] = self.binding_signature.try_into().unwrap();
        let signature = Signature::<Binding>::from(signature);

        let cv = data
            .actions
            .iter()
            .map(|ba| ValueCommitment::from_bytes(&ba.cv).unwrap())
            .sum::<ValueCommitment>();
        let pk: VerificationKey<Binding> = cv.to_bytes().try_into().unwrap();
        pk.verify(&data.sig_hash(), &signature)
            .map_err(|_| Error::ConstraintSystemFailure)?;
        Ok(data)
    }
}

///
#[derive(Debug)]
pub struct BallotBuilder<'a> {
    pk: &'a ProvingKey<Circuit>,
    vk: &'a VerifyingKey<Circuit>,
    domain: Fp,

    cmxs: Vec<Hash>,
    nfs: Vec<Nullifier>,

    inputs: Vec<VoteInput>,
    outputs: Vec<VoteOutput>,
}

const COIN_TYPE: u32 = 133;

impl<'a> BallotBuilder<'a> {
    ///
    pub fn new(
        name: &str,
        cmxs: Vec<Hash>,
        nfs: Vec<Nullifier>,
        pk: &'a ProvingKey<Circuit>,
        vk: &'a VerifyingKey<Circuit>,
    ) -> Self {
        let domain = crate::pob::domain(name.as_bytes());
        BallotBuilder {
            pk,
            vk,
            domain,
            cmxs,
            nfs,
            inputs: vec![],
            outputs: vec![],
        }
    }

    ///
    pub fn add_note(&mut self, pos: u32, sk: SpendingKey, note: Note) -> Result<(), Error> {
        self.inputs.push(VoteInput::from_parts(pos, sk, note));
        Ok(())
    }

    ///
    pub fn add_candidate(&mut self, address: Address, weight: u64) -> Result<(), Error> {
        self.outputs.push(VoteOutput(address, weight));
        Ok(())
    }

    ///
    pub fn build<R: RngCore + CryptoRng>(
        mut self,
        mut rng: R,
    ) -> Result<BallotEnvelope, Error> {
        println!("domain {}", hex::encode(self.domain.to_repr()));
        let mut proofs = vec![];
        let positions = self.inputs.iter().map(|i| i.pos).collect::<Vec<_>>();
        let (root, cmx_paths) = calculate_merkle_paths(0, &positions, &self.cmxs);
        let anchor = Anchor::from_bytes(root).unwrap();
        println!("anchor: {}", hex::encode(anchor.to_bytes()));

        for (cmx_path, i) in cmx_paths.into_iter().zip(self.inputs.iter_mut()) {
            i.cmx_path = MerklePath::from_parts(
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

        let n_actions = self.inputs.len().max(self.outputs.len());

        if self.inputs.len() <= n_actions {
            let n_dummy_inputs = n_actions - self.inputs.len();
            for _ in 0..n_dummy_inputs {
                let (sk, _, note) = Note::dummy(&mut rng, None);
                self.inputs.push(VoteInput::from_parts(0, sk, note));
            }
        }

        for input in self.inputs.iter_mut() {
            let nf = input.note.nullifier(&input.fvk);
            let find = self.nfs.binary_search(&nf);
            let nf_pos = match find {
                Ok(pos) => pos,
                Err(pos) => pos - 1,
            };
            let nf_start = self.nfs[nf_pos];
            let nf_end = self.nfs[nf_pos + 1];
            assert!(
                nf_pos % 2 == 0,
                "{} {:?} {:?} {:?} {:?}",
                nf_pos,
                find,
                nf,
                nf_start,
                nf_end
            );
            input.nf_pos = nf_pos as u32;
            input.nf_start = nf_start;
        }
        let positions = self.inputs.iter().map(|v| v.nf_pos).collect::<Vec<_>>();
        let nfs = self.nfs.iter().map(|nf| nf.0.to_repr()).collect::<Vec<_>>();
        let (nf_root, nf_paths) = calculate_merkle_paths(0, &positions, &nfs);
        let nf_anchor = Anchor::from_bytes(nf_root).unwrap();
        println!("nf_anchor: {}", hex::encode(nf_anchor.to_bytes()));

        for (nf_path, i) in nf_paths.into_iter().zip(self.inputs.iter_mut()) {
            i.nf_path = MerklePath::from_parts(
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

        let mut ballot_actions = vec![];
        let mut total_rcv = ValueCommitTrapdoor::zero();
        for i in 0..n_actions {
            let spend = &self.inputs[i];
            let rho = spend.note.nullifier_domain(&spend.fvk, self.domain);
            let (_, _, dummy_vote) = Note::dummy(&mut rng, Some(rho));
            let (candidate, output) = if i < self.outputs.len() {
                let VoteOutput(recipient, weight) = self.outputs[i];
                let rseed = RandomSeed::random(&mut rng, &rho);
                let note =
                    Note::from_parts(recipient, NoteValue::from_raw(weight), rho, rseed).unwrap();
                (recipient, note)
            } else {
                let recipient = dummy_vote.recipient();
                (recipient, dummy_vote)
            };
            let domain_nf = spend.note.nullifier_domain(&spend.fvk, self.domain);

            let value_net = spend.note.value() - output.value();
            let rcv = ValueCommitTrapdoor::random(&mut rng);
            total_rcv = total_rcv + &rcv;
            let cv_net = ValueCommitment::derive(value_net, rcv.clone());

            let alpha = Fq::random(&mut rng);
            let spk = SpendAuthorizingKey::from(&spend.sk);
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
                ElectionDomain(self.domain.clone()),
                nf_anchor,
            );

            let encryptor = OrchardNoteEncryption::new(None, output.clone(), candidate, [0u8; 512]);
            let encrypted_note = TransmittedNoteCiphertext {
                epk_bytes: encryptor.epk().to_bytes().0,
                enc_ciphertext: encryptor.encrypt_note_plaintext(),
                out_ciphertext: [0u8; OUT_CIPHERTEXT_SIZE],
            };
            // let _action = Action::from_parts(domain_nf, rk, cmx, encrypted_note, cv_net, ());

            let vote_power =
                VotePowerInfo::from_parts(domain_nf, spend.nf_start, spend.nf_path.clone());
            let spend_info = SpendInfo::new(
                spend.fvk.clone(),
                spend.note.clone(),
                spend.cmx_path.clone(),
            )
            .unwrap();
            let output_note = output.clone();

            assert!(spend.note.nullifier_domain(&spend.fvk, self.domain) == output_note.rho());
            let circuit = Circuit::from_action_context_unchecked(
                vote_power,
                spend_info,
                output_note,
                alpha,
                rcv,
            );
            println!("Create proof");
            let fp_instance = instance.to_halo2_instance();
            let prover = MockProver::run(12, &circuit, vec![fp_instance]).unwrap();
            prover.verify().unwrap();

            let proof = Proof::<Circuit>::create(
                self.pk,
                &[circuit],
                std::slice::from_ref(&instance),
                &mut rng,
            )?;

            proof.verify(self.vk, &[instance])?;
            proofs.push(proof.as_ref().to_vec());

            let rk: Hash = rk.into();
            let action = BallotAction {
                cv: cv_net.to_bytes(),
                rk,
                nf: domain_nf.to_bytes(),
                cmx: cmx.to_bytes(),
                epk: encrypted_note.epk_bytes,
                enc: encrypted_note.enc_ciphertext[0..52].try_into().unwrap(),
            };
            let bin_action = bincode::serialize(&action).unwrap();
            println!("{}", hex::encode(&bin_action));
            ballot_actions.push(action);
        }
        let data = BallotData {
            actions: ballot_actions,
        };
        let envelope = BallotEnvelope::build(data, total_rcv, proofs, &mut rng);
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::BallotBuilder;
    use crate::{
        keys::{FullViewingKey, Scope, SpendingKey},
        note::{ExtractedNoteCommitment, Nullifier, RandomSeed},
        primitives::redpallas::{Binding, Signature, VerificationKey},
        value::{NoteValue, ValueCommitment},
        vote::{
            ballot::{BallotEnvelope, COIN_TYPE},
            circuit::{Circuit, ElectionDomain},
            encryption::EncryptedVote,
            path::build_nf_ranges,
            proof::{ProvingKey, VerifyingKey},
        },
        Anchor, Note,
    };
    use ff::PrimeField;
    use pasta_curves::pallas;
    use rand::{RngCore, SeedableRng};

    #[test]
    fn f() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let sk = SpendingKey::random(&mut rng);
        let fvk = FullViewingKey::from(&sk);
        let my_address = fvk.address_at(0u64, Scope::External);
        let pk = ProvingKey::<Circuit>::build();
        let vk = VerifyingKey::build();

        let mut notes = vec![];
        let mut cmxs = vec![];
        let mut nfs = vec![];
        for i in 0..100 {
            let rho = Nullifier::dummy(&mut rng);
            let rseed = RandomSeed::random(&mut rng, &rho);
            let v = rng.next_u32() % 100;
            let note = Note::from_parts(
                my_address.clone(),
                NoteValue::from_raw((v as u64) * 100_000_000),
                rho,
                rseed,
            )
            .unwrap();
            let cmx = note.commitment();
            let cmx = ExtractedNoteCommitment::from(cmx);
            let cmx = cmx.to_bytes();
            cmxs.push(cmx);

            if i % 10 == 0 {
                notes.push((i, note));
            } else if i % 3 == 0 {
                let nf = note.nullifier(&fvk);
                nfs.push(nf);
            }
        }
        nfs.sort();
        let nfs = build_nf_ranges(nfs.into_iter());

        let mut builder = BallotBuilder::new("test-election", cmxs, nfs, &pk, &vk);
        let total_value = notes.iter().map(|(_, n)| n.value().inner()).sum::<u64>();
        for (i, n) in notes {
            builder.add_note(i, sk, n.clone()).unwrap();
        }
        const N_CANDIDATES: u64 = 4;
        assert!(total_value % N_CANDIDATES == 0);
        for i in 0..N_CANDIDATES {
            let sk = SpendingKey::from_zip32_seed(&[0u8; 32], COIN_TYPE, i as u32).unwrap();
            let fvk = FullViewingKey::from(&sk);
            let address = fvk.address_at(0u64, Scope::External);
            builder
                .add_candidate(address, total_value / N_CANDIDATES)
                .unwrap();
        }
        let (b, rcv, proofs) = builder.build(&mut rng).unwrap();
        let bas = &b.actions;
        for i in 0..N_CANDIDATES {
            let sk = SpendingKey::from_zip32_seed(&[0u8; 32], COIN_TYPE, i as u32).unwrap();
            let fvk = FullViewingKey::from(&sk);
            let ba = &bas[i as usize];
            let enc = EncryptedVote(ba.clone());
            let dec = enc.decrypt(&fvk);
            println!(
                "{} : {}",
                hex::encode(&dec.address.to_raw_address_bytes()),
                dec.value
            );
        }
        let signature: [u8; 64] = b.binding_sign(rcv, &mut rng).try_into().unwrap();
        println!("signature {}", hex::encode(&signature));
        let signature = Signature::<Binding>::from(signature);
        let cv = bas
            .iter()
            .map(|ba| ValueCommitment::from_bytes(&ba.cv).unwrap())
            .sum::<ValueCommitment>();
        let pk: VerificationKey<Binding> = cv.to_bytes().try_into().unwrap();
        pk.verify(&b.sig_hash(), &signature).unwrap();
    }

    #[test]
    fn g() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let sk = SpendingKey::random(&mut rng);
        let fvk = FullViewingKey::from(&sk);
        let my_address = fvk.address_at(0u64, Scope::External);
        let pk = ProvingKey::<Circuit>::build();
        let vk = VerifyingKey::build();

        let mut notes = vec![];
        let mut cmxs = vec![];
        let mut nfs = vec![];
        for i in 0..100 {
            let rho = Nullifier::dummy(&mut rng);
            let rseed = RandomSeed::random(&mut rng, &rho);
            let v = rng.next_u32() % 100;
            let note = Note::from_parts(
                my_address.clone(),
                NoteValue::from_raw((v as u64) * 100_000_000),
                rho,
                rseed,
            )
            .unwrap();
            let cmx = note.commitment();
            let cmx = ExtractedNoteCommitment::from(cmx);
            let cmx = cmx.to_bytes();
            cmxs.push(cmx);

            if i % 10 == 0 {
                notes.push((i, note));
            } else if i % 3 == 0 {
                let nf = note.nullifier(&fvk);
                nfs.push(nf);
            }
        }
        nfs.sort();
        let nfs = build_nf_ranges(nfs.into_iter());

        let mut builder = BallotBuilder::new("test-election", cmxs, nfs, &pk, &vk);
        let total_value = notes.iter().map(|(_, n)| n.value().inner()).sum::<u64>();
        for (i, n) in notes {
            builder.add_note(i, sk, n.clone()).unwrap();
        }
        const N_CANDIDATES: u64 = 4;
        assert!(total_value % N_CANDIDATES == 0);
        for i in 0..N_CANDIDATES {
            let sk = SpendingKey::from_zip32_seed(&[0u8; 32], COIN_TYPE, i as u32).unwrap();
            let fvk = FullViewingKey::from(&sk);
            let address = fvk.address_at(0u64, Scope::External);
            builder
                .add_candidate(address, total_value / N_CANDIDATES)
                .unwrap();
        }
        let (data, rcv, proofs) = builder.build(&mut rng).unwrap();

        let ballot = BallotEnvelope::build(data, rcv, proofs, &mut rng);

        let domain = pallas::Base::from_repr(
            hex::decode("81262ef6776ba0266f34b94373c1e684488b00de6c0bcc4f9225f00d31c16502")
                .unwrap()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        let anchor = Anchor::from_bytes(
            hex::decode("0339a0698071fc106e36801ac4c5717c6688219a4857bec5b63f77ff38d4ef28")
                .unwrap()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        let nf_anchor = Anchor::from_bytes(
            hex::decode("94ea7451f2895856adedf21273e5f28456c098efb441e1ecc0f961d00e700626")
                .unwrap()
                .try_into()
                .unwrap(),
        )
        .unwrap();

        let _data = ballot.verify(&vk, ElectionDomain(domain), anchor, nf_anchor).unwrap();
    }
}
