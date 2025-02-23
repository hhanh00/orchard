use super::{
    ballot::{
        Ballot, BallotAction, BallotActionSecret, BallotAnchors, BallotData, BallotWitnesses,
        VoteProof, VoteSignature,
    },
    circuit::{Circuit, Instance, VotePowerInfo},
    path::calculate_merkle_paths,
    proof::Proof,
    proof::ProvingKey,
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

///
pub fn vote<R: RngCore + CryptoRng>(
    domain: Fp,
    signature_required: bool,
    sk: Option<SpendingKey>,
    fvk: &FullViewingKey,
    address: Address,
    amount: u64,
    notes: &[(Note, u32)],
    nfs: &[Fp],
    cmxs: &[Fp],
    mut rng: R,
    pk: &ProvingKey<Circuit>,
) -> Result<Ballot, VoteError> {
    // let nfs = list_nf_ranges(connection)?;
    // let cmxs = list_cmxs(connection)?;
    // let address = VoteAddress::decode(address)?.0;
    // let notes = list_notes(connection, 0, fvk)?;

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
                let vote_output =
                    Note::from_parts(address, NoteValue::from_raw(amount), rho, rseed).unwrap();
                vote_output
            }
            1 => {
                let change_output =
                    Note::from_parts(self_address, NoteValue::from_raw(change), rho, rseed)
                        .unwrap();

                change_output
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

        // Derive from seed if available
        let alpha = Fq::random(&mut rng);
        let svk = SpendValidatingKey::from(fvk.clone());
        let rk = svk.randomize(&alpha);
        let sp_signkey = sk.map(|sk| {
            let spak = SpendAuthorizingKey::from(&sk);
            let sp_signkey = spak.randomize(&alpha);
            sp_signkey
        });

        let nf = spend.nullifier(&fvk);
        let nf = Fp::from_repr(nf.to_bytes()).unwrap();
        let position = nfs.binary_search(&nf);
        let nf_position = (match position {
            Ok(position) => position,
            Err(position) => position - 1,
        } & !1); // snap to even position, ie start of range
        let nf_start = nfs[nf_position];
        ballot_secrets.push(BallotActionSecret {
            fvk: fvk.clone(),
            spend_note: spend.clone(),
            output_note: output.clone(),
            rcv: rcv.clone(),
            alpha,
            sp_signkey,
            nf: Nullifier::from_bytes(&nf.to_repr()).unwrap(),
            nf_start: Nullifier::from_bytes(&nf_start.to_repr()).unwrap(),
            nf_position: nf_position as u32,
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

    let nf_positions = ballot_secrets
        .iter()
        .map(|s| s.nf_position)
        .collect::<Vec<_>>();
    let (nf_root, nf_mps) = calculate_merkle_paths(0, &nf_positions, &nfs);

    let cmx_positions = ballot_secrets
        .iter()
        .map(|s| s.cmx_position)
        .collect::<Vec<_>>();
    let (cmx_root, cmx_mps) = calculate_merkle_paths(0, &cmx_positions, &cmxs);

    let mut proofs = vec![];
    for (((secret, public), cmx_mp), nf_mp) in ballot_secrets
        .iter()
        .zip(ballot_actions.iter())
        .zip(cmx_mps.iter())
        .zip(nf_mps.iter())
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
        assert_eq!(
            secret.nf_start,
            Nullifier::from_bytes(&nfs[secret.nf_position as usize].to_repr()).unwrap()
        );
        assert_eq!(nf_mp.position, secret.nf_position);
        let nf_path = nf_mp.to_orchard_merkle_tree();
        let vote_power = VotePowerInfo {
            dnf: Nullifier::from_bytes(&as_byte256(&public.nf)).unwrap(),
            nf_start: secret.nf_start,
            nf_path,
        };
        let cmx_path = cmx_mp.to_orchard_merkle_tree();
        let spend_info = SpendInfo::new(secret.fvk.clone(), secret.spend_note, cmx_path).unwrap();
        let circuit = Circuit::from_action_context_unchecked(
            vote_power,
            spend_info,
            secret.output_note,
            secret.alpha,
            secret.rcv.clone(),
        );

        tracing::info!("Proving");
        let proof =
            Proof::<Circuit>::create(pk, &[circuit], std::slice::from_ref(&instance), &mut rng)?;
        tracing::info!("Verifying");
        tracing::info!("Proof generated");
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
