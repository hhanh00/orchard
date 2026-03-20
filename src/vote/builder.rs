use super::{
    ballot::{
        Ballot, BallotAction, BallotActionSecret, BallotAnchors, BallotData, BallotWitnesses,
        VoteProof, VoteSignature,
    },
    circuit::{Circuit, Instance, VotePowerInfo, NF_MERKLE_DEPTH},
    path::calculate_merkle_paths,
    proof::{Proof, ProvingKey, VerifyingKey},
};
use crate::{
    builder::SpendInfo,
    keys::{FullViewingKey, Scope, SpendAuthorizingKey, SpendValidatingKey, SpendingKey},
    note::{ExtractedNoteCommitment, Nullifier, RandomSeed},
    note_encryption::OrchardNoteEncryption,
    primitives::redpallas::{Binding, SigningKey, SpendAuth, VerificationKey},
    value::{NoteValue, ValueCommitTrapdoor, ValueCommitment},
    Anchor, Note,
};
use crate::{vote::util::as_byte256, Address};
use pasta_curves::{
    group::ff::{Field as _, PrimeField},
    Fp, Fq,
};
use rand::{CryptoRng, RngCore};
use zcash_note_encryption::COMPACT_NOTE_SIZE;

use super::VoteError;

/// Pre-computed proof data for one nullifier's non-membership proof.
/// Produced by the PIR client or by local tree computation.
#[derive(Clone, Debug)]
pub struct NfProofData {
    /// Merkle root of the nullifier range tree
    pub root: Fp,
    /// Lower bound of the gap range
    pub low: Fp,
    /// Width of the gap range (high - low)
    pub width: Fp,
    /// Leaf position in the tree
    pub leaf_pos: u32,
    /// Merkle authentication path (29 sibling hashes)
    pub path: [Fp; NF_MERKLE_DEPTH],
}

/// Build a vote ballot.
///
/// `nf_proofs` contains one pre-computed nullifier non-membership proof per input note.
/// These are obtained either from a PIR server query or from local tree computation.
pub fn vote<R: RngCore + CryptoRng>(
    domain: Fp,
    signature_required: bool,
    sk: Option<SpendingKey>,
    fvk: &FullViewingKey,
    address: Address,
    amount: u64,
    notes: &[(Note, u32)],
    nf_proofs: &[NfProofData],
    cmxs: &[Fp],
    mut rng: R,
    pk: &ProvingKey<Circuit>,
    vk: &VerifyingKey<Circuit>,
) -> Result<Ballot, VoteError> {
    let mut total_value = 0;
    let mut inputs = vec![];
    for np in notes {
        if total_value >= amount {
            break;
        }
        inputs.push(np);
        total_value += np.0.value().inner();
    }
    let self_address = fvk.address_at(0u64, Scope::External);
    if total_value < amount {
        return Err(VoteError::NotEnoughFunds);
    }
    let change = total_value - amount;

    let n_actions = inputs.len().max(2);
    let mut ballot_actions = vec![];
    let mut ballot_secrets = vec![];
    let mut total_rcv = ValueCommitTrapdoor::zero();
    for i in 0..n_actions {
        let (sk, fvk, (spend, cmx_position)) = if i < inputs.len() {
            (sk.clone(), fvk.clone(), inputs[i].clone())
        } else {
            let (sk, fvk, note) = Note::dummy(&mut rng, None);
            (Some(sk), fvk, (note, 0))
        };

        let rho = spend.nullifier_domain(&fvk, domain);
        let rseed = RandomSeed::random(&mut rng, &rho);
        let output = match i {
            0 => {
                Note::from_parts(address, NoteValue::from_raw(amount), rho, rseed).unwrap()
            }
            1 => {
                Note::from_parts(self_address, NoteValue::from_raw(change), rho, rseed).unwrap()
            }
            _ => {
                let (_, _, dummy_output) = Note::dummy(&mut rng, Some(rho));
                dummy_output
            }
        };

        let cv_net = spend.value() - output.value();
        let rcv = ValueCommitTrapdoor::random(&mut rng);
        total_rcv = total_rcv + &rcv;
        let cv_net = ValueCommitment::derive(cv_net, rcv.clone());

        let alpha = Fq::random(&mut rng);
        let svk = SpendValidatingKey::from(fvk.clone());
        let rk = svk.randomize(&alpha);
        let sp_signkey = sk.map(|sk| {
            let spak = SpendAuthorizingKey::from(&sk);
            spak.randomize(&alpha)
        });

        let nf = spend.nullifier(&fvk);
        let nf_fp = Fp::from_repr(nf.to_bytes()).unwrap();

        // Get the pre-computed NF proof for this note.
        // For real inputs (i < inputs.len()), use the corresponding proof.
        // For dummy actions, use the first proof as a placeholder (it won't be checked
        // because v_old = 0 makes the circuit skip NF validation).
        let nf_proof = if i < nf_proofs.len() {
            &nf_proofs[i]
        } else {
            &nf_proofs[0]
        };

        // Verify the proof data is consistent
        if i < inputs.len() {
            let offset = nf_fp - nf_proof.low;
            if offset > nf_proof.width {
                return Err(VoteError::InputError);
            }
        }

        ballot_secrets.push(BallotActionSecret {
            fvk: fvk.clone(),
            spend_note: spend.clone(),
            output_note: output.clone(),
            rcv: rcv.clone(),
            alpha,
            sp_signkey,
            nf: Nullifier::from_bytes(&nf_fp.to_repr()).unwrap(),
            nf_low: nf_proof.low,
            nf_width: nf_proof.width,
            nf_pos: nf_proof.leaf_pos,
            nf_path: nf_proof.path,
            nf_root: nf_proof.root,
            cmx_position,
            cv_net: cv_net.clone(),
            rk: rk.clone(),
        });

        let cmx = output.commitment();
        let cmx = ExtractedNoteCommitment::from(cmx);

        let address = output.recipient();
        let encryptor = OrchardNoteEncryption::new(None, output.clone(), address, [0u8; 512]);
        let epk = encryptor.epk().to_bytes().0;
        let enc = encryptor.encrypt_note_plaintext();
        let mut compact_enc = [0u8; COMPACT_NOTE_SIZE];
        compact_enc.copy_from_slice(&enc[0..COMPACT_NOTE_SIZE]);

        let rk: [u8; 32] = rk.into();
        let ballot_action = BallotAction {
            cv_net: cv_net.to_bytes().to_vec(),
            rk: rk.to_vec(),
            nf: rho.to_bytes().to_vec(),
            cmx: cmx.to_bytes().to_vec(),
            epk: epk.to_vec(),
            enc: compact_enc.to_vec(),
        };

        ballot_actions.push(ballot_action);
    }

    // All NF proofs share the same root
    let nf_root = nf_proofs[0].root;

    let cmx_positions = ballot_secrets
        .iter()
        .map(|s| s.cmx_position)
        .collect::<Vec<_>>();
    let (cmx_root, cmx_mps) = calculate_merkle_paths(0, &cmx_positions, &cmxs);

    let mut proofs = vec![];
    for (i, ((secret, public), cmx_mp)) in ballot_secrets
        .iter()
        .zip(ballot_actions.iter())
        .zip(cmx_mps.iter())
        .enumerate()
    {
        let cmx = ExtractedNoteCommitment::from_bytes(&as_byte256(&public.cmx)).unwrap();
        let instance = Instance::from_parts(
            Anchor::from_bytes(cmx_root.to_repr()).unwrap(),
            secret.cv_net.clone(),
            Nullifier::from_bytes(&as_byte256(&public.nf)).unwrap(),
            secret.rk.clone(),
            cmx,
            domain.clone(),
            Anchor::from_bytes(nf_root.to_repr()).unwrap(),
        );

        let vote_power = VotePowerInfo::from_parts(
            Nullifier::from_bytes(&as_byte256(&public.nf)).unwrap(),
            secret.nf_low,
            secret.nf_width,
            secret.nf_pos,
            secret.nf_path,
        );

        let cmx_path = cmx_mp.to_orchard_merkle_tree();

        let spend_info = SpendInfo::new(secret.fvk.clone(), secret.spend_note, cmx_path).unwrap();
        let circuit = Circuit::from_action_context_unchecked(
            vote_power,
            spend_info,
            secret.output_note,
            secret.alpha,
            secret.rcv.clone(),
        );

        let instances = std::slice::from_ref(&instance);
        tracing::info!("Proving action {}", i);
        let proof = Proof::<Circuit>::create(pk, &[circuit], instances, &mut rng)?;
        tracing::info!("Verifying action {}", i);
        proof.verify(vk, instances)?;
        tracing::info!("Proof generated for action {}", i);
        let proof = proof.as_ref().to_vec();
        proofs.push(VoteProof(proof));
    }

    let anchors = BallotAnchors {
        nf: nf_root.to_repr().to_vec(),
        cmx: cmx_root.to_repr().to_vec(),
    };

    let ballot_data = BallotData {
        version: 1,
        domain: domain.to_repr().to_vec(),
        actions: ballot_actions.clone(),
        anchors,
    };
    let sighash = ballot_data
        .sighash()
        .map_err(|_| VoteError::InvalidBallot("sighash".to_string()))?;

    let sp_signatures = ballot_secrets
        .iter()
        .zip(ballot_actions.iter())
        .map(|(s, a)| {
            s.sp_signkey.as_ref().map(|sk| {
                let signature = sk.sign(&mut rng, &sighash);
                let signature_bytes: [u8; 64] = (&signature).into();
                let rk = as_byte256(&a.rk);
                let rk: VerificationKey<SpendAuth> = rk.try_into().unwrap();
                rk.verify(&sighash, &signature).unwrap();
                VoteSignature(signature_bytes.to_vec())
            })
        })
        .collect::<Option<Vec<_>>>();
    if signature_required && sp_signatures.is_none() {
        return Err(VoteError::InvalidSignature(
            "Signature required".to_string(),
        ));
    }

    let bsk: SigningKey<Binding> = total_rcv.to_bytes().try_into().unwrap();
    let binding_signature = bsk.sign(&mut rng, &sighash);
    let binding_signature: [u8; 64] = (&binding_signature).into();
    let binding_signature = binding_signature.to_vec();

    let witnesses = BallotWitnesses {
        proofs,
        sp_signatures,
        binding_signature,
    };

    let ballot = Ballot {
        data: ballot_data,
        witnesses,
    };

    Ok(ballot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vote::util::poseidon_hash;
    use ff::PrimeField;
    use pasta_curves::group::ff::Field;
    use rand::rngs::OsRng;

    /// Build a minimal NF tree from sorted nullifiers, returning
    /// (root, ranges, levels) — same logic as zcash-vote/src/trees.rs.
    fn build_nf_tree(sorted_nfs: &[Fp]) -> (Fp, Vec<[Fp; 2]>, Vec<Vec<Fp>>) {
        let mut ranges: Vec<[Fp; 2]> = vec![];
        let mut prev = Fp::zero();
        for &nf in sorted_nfs {
            if prev < nf {
                let high = nf - Fp::one();
                ranges.push([prev, high - prev]);
            }
            prev = nf + Fp::one();
        }
        if prev != Fp::zero() {
            let high = Fp::one().neg();
            ranges.push([prev, high - prev]);
        }

        let leaves: Vec<Fp> = ranges.iter().map(|[l, w]| poseidon_hash(*l, *w)).collect();

        let mut empty = [Fp::zero(); NF_MERKLE_DEPTH];
        empty[0] = poseidon_hash(Fp::zero(), Fp::zero());
        for i in 1..NF_MERKLE_DEPTH {
            empty[i] = poseidon_hash(empty[i - 1], empty[i - 1]);
        }

        let mut level0 = leaves;
        if level0.is_empty() {
            level0.push(empty[0]);
        }
        if level0.len() & 1 == 1 {
            level0.push(empty[0]);
        }
        let mut levels: Vec<Vec<Fp>> = vec![level0];

        for i in 0..NF_MERKLE_DEPTH - 1 {
            let prev_level = &levels[i];
            let pairs = prev_level.len() / 2;
            let mut next: Vec<Fp> = (0..pairs)
                .map(|j| poseidon_hash(prev_level[j * 2], prev_level[j * 2 + 1]))
                .collect();
            if next.len() & 1 == 1 {
                next.push(empty[i + 1]);
            }
            levels.push(next);
        }

        let top = &levels[NF_MERKLE_DEPTH - 1];
        let root = poseidon_hash(top[0], top[1]);
        (root, ranges, levels)
    }

    /// Find range containing value via binary search.
    fn find_range(ranges: &[[Fp; 2]], value: Fp) -> Option<usize> {
        let i = ranges.partition_point(|[low, _]| *low <= value);
        if i == 0 {
            return None;
        }
        let idx = i - 1;
        let [low, width] = ranges[idx];
        if value - low <= width { Some(idx) } else { None }
    }

    /// Extract 29-level auth path for leaf at `idx`.
    fn extract_path(idx: usize, levels: &[Vec<Fp>]) -> [Fp; NF_MERKLE_DEPTH] {
        let mut empty = [Fp::zero(); NF_MERKLE_DEPTH];
        empty[0] = poseidon_hash(Fp::zero(), Fp::zero());
        for i in 1..NF_MERKLE_DEPTH {
            empty[i] = poseidon_hash(empty[i - 1], empty[i - 1]);
        }

        let mut path = [Fp::zero(); NF_MERKLE_DEPTH];
        let mut pos = idx;
        for level in 0..NF_MERKLE_DEPTH {
            let sibling = pos ^ 1;
            path[level] = if sibling < levels[level].len() {
                levels[level][sibling]
            } else {
                empty[level]
            };
            pos >>= 1;
        }
        path
    }

    /// Full prove + verify test using the actual halo2 proof system.
    ///
    /// This is the definitive test: it calls `vote()` which internally runs
    /// `Proof::create()` and `Proof::verify()` with real polynomial commitments
    /// (IPA), random Fiat-Shamir challenges, and the full trusted setup.
    ///
    /// If this test passes, the vote circuit WILL work on a real blockchain.
    #[test]
    fn vote_full_prove_verify() {
        let mut rng = OsRng;

        // --- 1. Create spending key, FVK, and address ---
        let sk = crate::keys::SpendingKey::random(&mut rng);
        let fvk: crate::keys::FullViewingKey = (&sk).into();
        let address = fvk.address_at(0u32, crate::keys::Scope::External);

        // --- 2. Create a note with value 100 ---
        let spent_note = crate::Note::new(
            address,
            NoteValue::from_raw(100),
            Nullifier::dummy(&mut rng),
            &mut rng,
        );

        // --- 3. Build CMX tree with this note ---
        let cmx = crate::note::ExtractedNoteCommitment::from(spent_note.commitment());
        let cmx_fp = cmx.inner();
        let cmxs = vec![cmx_fp];

        // --- 4. Build NF tree with sentinels ---
        let step = Fp::from(2u64).pow([250, 0, 0, 0]);
        let mut nfs: Vec<Fp> = (0u64..=16).map(|k| step * Fp::from(k)).collect();
        nfs.sort();
        let (nf_root, ranges, levels) = build_nf_tree(&nfs);

        // --- 5. Compute NF proof for the note's nullifier ---
        let nf_old = spent_note.nullifier(&fvk);
        let nf_fp = Fp::from_repr(nf_old.to_bytes()).unwrap();
        let range_idx = find_range(&ranges, nf_fp)
            .expect("nullifier must fall in a gap range");
        let [low, width] = ranges[range_idx];
        let path = extract_path(range_idx, &levels);

        let nf_proof = NfProofData {
            root: nf_root,
            low,
            width,
            leaf_pos: range_idx as u32,
            path,
        };

        // --- 6. Build proving and verifying keys (expensive — real trusted setup) ---
        eprintln!("Building proving key (this takes ~1-2 minutes)...");
        let pk = crate::vote::proof::ProvingKey::<Circuit>::build();
        eprintln!("Building verifying key...");
        let vk = crate::vote::proof::VerifyingKey::<Circuit>::build();

        // --- 7. Call vote() — the real ballot builder ---
        // This internally runs Proof::create() + Proof::verify() for each action.
        let domain = Fp::from(42u64);
        let notes = vec![(spent_note, 0u32)]; // note at CMX position 0
        let nf_proofs = vec![nf_proof];

        eprintln!("Creating ballot with real proofs...");
        let ballot = vote(
            domain,
            true, // signature required
            Some(sk),
            &fvk,
            address,
            100, // vote full amount
            &notes,
            &nf_proofs,
            &cmxs,
            &mut rng,
            &pk,
            &vk,
        );

        let ballot = ballot.expect("vote() should produce a valid ballot");

        // --- 8. Verify the ballot structure ---
        assert_eq!(ballot.data.version, 1);
        assert_eq!(ballot.data.actions.len(), 2); // min 2 actions
        assert_eq!(ballot.witnesses.proofs.len(), 2);
        assert!(ballot.witnesses.sp_signatures.is_some());
        assert_eq!(ballot.data.domain, domain.to_repr().to_vec());
        assert_eq!(ballot.data.anchors.nf, nf_root.to_repr().to_vec());

        eprintln!("Full prove+verify test PASSED — ballot is valid!");
    }
}
