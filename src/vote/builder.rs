use std::error::Error;

use ff::{Field, PrimeField};
use halo2_proofs::dev::MockProver;
use pasta_curves::pallas::{Base as Fp, Scalar as Fq};
use rand::{CryptoRng, RngCore};
use zcash_note_encryption::OUT_CIPHERTEXT_SIZE;

use super::{path::calculate_merkle_paths, proof::Halo2Instance, Election, Hash};
use crate::{
    builder::SpendInfo, constants::MERKLE_DEPTH_ORCHARD, keys::{FullViewingKey, Scope, SpendAuthorizingKey, SpendingKey}, note::{ExtractedNoteCommitment, Nullifier, RandomSeed, TransmittedNoteCiphertext}, note_encryption::OrchardNoteEncryption, primitives::redpallas::{SpendAuth, VerificationKey}, tree::{MerkleHashOrchard, MerklePath}, value::{NoteValue, ValueCommitTrapdoor, ValueCommitment}, vote::circuit::{Circuit, ElectionDomain, Instance, VotePowerInfo}, Action, Address, Anchor, Note
};

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

struct VoteOutput(Address, u64);

pub struct BallotBuilder {
    election_seed: Hash,
    domain: Fp,

    cmxs: Vec<Hash>,
    nfs: Vec<Nullifier>,

    inputs: Vec<VoteInput>,
    outputs: Vec<VoteOutput>,
}

const COIN_TYPE: u32 = 133;

impl BallotBuilder {
    pub fn new(name: &str, seed: Hash, cmxs: Vec<Hash>, nfs: Vec<Nullifier>) -> Self {
        let domain = crate::pob::domain(name.as_bytes());
        BallotBuilder {
            election_seed: seed,
            domain,
            cmxs,
            nfs,
            inputs: vec![],
            outputs: vec![],
        }
    }

    pub fn add_note(
        &mut self,
        pos: u32,
        sk: SpendingKey,
        note: Note,
    ) -> Result<(), Box<dyn Error>> {
        self.inputs
            .push(VoteInput::from_parts(pos, sk, note));
        Ok(())
    }

    pub fn add_candidate(&mut self, address: Address, weight: u64) -> Result<(), Box<dyn Error>> {
        self.outputs.push(VoteOutput(address, weight));
        Ok(())
    }

    pub fn build<R: RngCore + CryptoRng>(mut self, mut rng: R) -> Result<(), Box<dyn Error>> {
        let positions = self.inputs.iter().map(|i| i.pos).collect::<Vec<_>>();
        let (root, cmx_paths) = calculate_merkle_paths(0, &positions, &self.cmxs);
        let anchor = Anchor::from_bytes(root).unwrap();

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
                self.inputs
                    .push(VoteInput::from_parts(0, sk, note));
            }
        }

        for input in self.inputs.iter_mut() {
            let nf = input.note.nullifier(&input.fvk);
            let nf_pos = match self.nfs.binary_search(&nf) {
                Ok(pos) => pos,
                Err(pos) => pos - 1,
            };
            assert!(nf_pos % 2 == 0);
            let nf_start = self.nfs[nf_pos];
            input.nf_pos = nf_pos as u32;
            input.nf_start = nf_start;
        }
        let positions = self.inputs.iter().map(|v| v.nf_pos).collect::<Vec<_>>();
        let nfs = self.nfs.iter().map(|nf| nf.0.to_repr()).collect::<Vec<_>>();
        let (nf_root, nf_paths) = calculate_merkle_paths(0, &positions, &nfs);
        let nf_anchor = Anchor::from_bytes(nf_root).unwrap();

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

        for i in 0..n_actions {
            let spend = &self.inputs[i];
            let rho = spend.note.nullifier(&spend.fvk);
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
            let _action = Action::from_parts(domain_nf, rk, cmx, encrypted_note, cv_net, ());

            let vote_power =
                VotePowerInfo::from_parts(domain_nf, spend.nf_start, spend.nf_path.clone());
            let spend_info = SpendInfo::new(
                spend.fvk.clone(),
                spend.note.clone(),
                spend.cmx_path.clone(),
            )
            .unwrap();
            let output_note = output.clone();

            assert!(spend.note.nullifier(&spend.fvk) == output_note.rho());
            let circuit =
                Circuit::from_action_context(vote_power, spend_info, output_note, alpha, rcv)
                    .unwrap();
            println!("Create proof");
            let instance = instance.to_halo2_instance();
            let prover = MockProver::run(12, &circuit, vec![instance]).unwrap();
            prover.verify().unwrap();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::{RngCore, SeedableRng};
    use crate::{
        keys::{FullViewingKey, Scope, SpendingKey}, note::{ExtractedNoteCommitment, Nullifier, RandomSeed}, value::NoteValue, vote::{builder::COIN_TYPE, path::{make_nf_leaves, nf_leaves}}, Note
    };
    use super::BallotBuilder;

    #[test]
    fn f() {
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let sk = SpendingKey::random(&mut rng);
        let fvk = FullViewingKey::from(&sk);
        let my_address = fvk.address_at(0u64, Scope::External);

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
            }
            else if i % 3 == 0 {
                let nf = note.nullifier(&fvk);
                nfs.push(nf);
            }
        }
        nfs.sort();
        let nfs = nf_leaves(&nfs);

        let mut builder = BallotBuilder::new("test-election", [42u8; 32], cmxs, nfs);
        let total_value = notes
            .iter()
            .map(|(_, n)| n.value().inner())
            .sum::<u64>();
        for (i, n) in notes {
            builder.add_note(i, sk, n.clone()).unwrap();
        }
        const N_CANDIDATES: u64 = 4;
        assert!(total_value % N_CANDIDATES == 0);
        for i in 0..N_CANDIDATES {
                let sk =
                    SpendingKey::from_zip32_seed(&[0u8; 32], COIN_TYPE, i as u32).unwrap();
                let fvk = FullViewingKey::from(&sk);
                let address = fvk.address_at(0u64, Scope::External);
                builder.add_candidate(address, total_value / N_CANDIDATES).unwrap();
        }
        builder.build(&mut rng).unwrap();
    }
}
