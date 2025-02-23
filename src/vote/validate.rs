use alloc::string::ToString;
use crate::note::Rho;
use crate::vote::util::CtOpt;
use crate::{
    keys::PreparedIncomingViewingKey,
    note::{ExtractedNoteCommitment, Nullifier},
    note_encryption::{CompactAction, OrchardDomain},
    primitives::redpallas::{Binding, Signature, SpendAuth, VerificationKey},
    value::ValueCommitment,
    vote::{
        ballot::{Ballot, BallotAction, BallotData, BallotWitnesses},
        circuit::{Circuit, Instance},
        proof::Proof,
        util::as_byte256,
    },
    Anchor, Note,
};
use ff::PrimeField;
use pasta_curves::Fp;
use zcash_note_encryption::{try_compact_note_decryption, EphemeralKeyBytes};

use super::{proof::VerifyingKey, VoteError};

///
pub fn validate_ballot(
    ballot: Ballot,
    signature_check: bool,
    vk: &VerifyingKey<Circuit>,
) -> Result<BallotData, VoteError> {
    let Ballot { data, witnesses } = ballot;
    let sighash = data
        .sighash()
        .map_err(|_| VoteError::InvalidBallot("Invalid format".to_string()))?;
    let domain = Fp::from_repr(as_byte256(&data.domain)).unwrap();

    tracing::info!("Verify spending signatures if needed");
    if let Some(sp_signatures) = witnesses.sp_signatures {
        for (signature, action) in sp_signatures.into_iter().zip(data.actions.iter()) {
            let signature: [u8; 64] = signature.0.try_into().map_err(|_| {
                VoteError::InvalidSignature("Signature must be 64 byte long".to_string())
            })?;
            let signature: Signature<SpendAuth> = signature.into();
            let rk = as_byte256(&action.rk);
            let rk: VerificationKey<SpendAuth> = rk
                .try_into()
                .map_err(|_| VoteError::InvalidKey("Invalid public key".to_string()))?;
            rk.verify(&sighash, &signature).map_err(|_| {
                VoteError::InvalidSignature("Signature verification failed".to_string())
            })?;
        }
    } else if signature_check {
        return Err(VoteError::InvalidBallot("Signatures missing".to_string()));
    }

    tracing::info!("Verify binding signature");
    let mut total_cv = ValueCommitment::derive_from_value(0);
    for action in data.actions.iter() {
        let cv_net = as_byte256(&action.cv_net);
        let cv_net = CtOpt(ValueCommitment::from_bytes(&cv_net)).to_result()?;
        total_cv = total_cv + &cv_net;
    }
    let cv: VerificationKey<Binding> = total_cv
        .to_bytes()
        .try_into()
        .map_err(|_| VoteError::InputError)?;
    let binding_signature: [u8; 64] = witnesses
        .binding_signature
        .try_into()
        .map_err(|_| VoteError::InvalidSignature("Invalid binding signature".to_string()))?;
    let binding_signature: Signature<Binding> = binding_signature.into();
    cv.verify(&sighash, &binding_signature)
        .map_err(|_| VoteError::InvalidSignature("Binding Signature".to_string()))?;

    let BallotWitnesses { proofs, .. } = witnesses;

    tracing::info!("Verify ZKP");
    for (proof, action) in proofs.into_iter().zip(data.actions.iter()) {
        let proof: Proof<Circuit> = Proof::new(proof.0);
        let cmx_root = as_byte256(&data.anchors.cmx);
        let nf_root = as_byte256(&data.anchors.nf);
        let cv_net = as_byte256(&action.cv_net);
        let dnf = as_byte256(&action.nf);
        let rk = as_byte256(&action.rk);
        let cmx = as_byte256(&action.cmx);
        tracing::info!("cmx_root {}", hex::encode(&cmx_root));
        tracing::info!("nf_root {}", hex::encode(&nf_root));
        tracing::info!("cv_net {}", hex::encode(&cv_net));
        tracing::info!("dnf {}", hex::encode(&dnf));
        tracing::info!("rk {}", hex::encode(&rk));
        tracing::info!("cmx {}", hex::encode(&cmx));

        let instance = Instance::from_parts(
            CtOpt(Anchor::from_bytes(cmx_root)).to_result()?,
            CtOpt(ValueCommitment::from_bytes(&cv_net)).to_result()?,
            CtOpt(Nullifier::from_bytes(&dnf)).to_result()?,
            rk.try_into().map_err(|_| VoteError::InputError)?,
            CtOpt(ExtractedNoteCommitment::from_bytes(&cmx)).to_result()?,
            domain,
            CtOpt(Anchor::from_bytes(nf_root)).to_result()?,
        );

        proof.verify(vk, &[instance])?;
    }

    // TODO: Verify anchors

    Ok(data)
}

///
pub fn try_decrypt_ballot(
    ivk: &PreparedIncomingViewingKey,
    action: &BallotAction,
) -> Result<Option<Note>, VoteError> {
    let BallotAction {
        nf, cmx, epk, enc, ..
    } = action;

    let nf = Nullifier::from_bytes(&as_byte256(&nf)).unwrap();
    let rho = Rho::from_nf_old(nf);
    let domain = OrchardDomain::for_rho(rho.clone());
    let action = CompactAction::from_parts(
        nf,
        ExtractedNoteCommitment::from_bytes(&as_byte256(&cmx)).unwrap(),
        EphemeralKeyBytes(as_byte256(&epk)),
        enc.clone().try_into().unwrap(),
    );
    let note = try_compact_note_decryption(&domain, ivk, &action).map(|na| na.0);
    Ok(note)
}
