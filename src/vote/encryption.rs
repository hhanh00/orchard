use rand::{CryptoRng, RngCore};
use zcash_note_encryption::{try_compact_note_decryption, ShieldedOutput};

use crate::{keys::{FullViewingKey, PreparedIncomingViewingKey, Scope}, note::{ExtractedNoteCommitment, Nullifier, RandomSeed}, note_encryption::OrchardDomain, value::NoteValue, Address, Note};

use super::BallotAction;

///
#[derive(Debug)]
pub struct EncryptedVote(pub BallotAction);

///
#[derive(Debug)]
pub struct DecryptedVote {
    ///
    pub address: Address,
    ///
    pub value: u64,
    ///
    pub rho: Nullifier,
    ///
    pub rseed: RandomSeed,
    ///
    pub cmx: ExtractedNoteCommitment,
}

impl DecryptedVote {
    ///
    pub fn random<R: RngCore + CryptoRng>(mut rng: R) -> Self {
        let (_, _, note) = Note::dummy(&mut rng, None);
        let v = rng.next_u32() as u64;
        let note = Note::from_parts(note.recipient(), NoteValue::from_raw(v), 
        note.rho(), note.rseed().clone()).unwrap();
        let cmx = ExtractedNoteCommitment::from(note.commitment());
        DecryptedVote {
            address: note.recipient(),
            value: v,
            rho: note.rho(),
            rseed: note.rseed().clone(),
            cmx,
        }
    }
}

impl ShieldedOutput<OrchardDomain, 52> for BallotAction {
    fn ephemeral_key(&self) -> zcash_note_encryption::EphemeralKeyBytes {
        zcash_note_encryption::EphemeralKeyBytes(self.epk)
    }

    fn cmstar_bytes(&self) -> [u8; 32] {
        self.cmx
    }

    fn enc_ciphertext(&self) -> &[u8; 52] {
        &self.enc
    }
}

impl EncryptedVote {
    ///
    pub fn decrypt(&self, fvk: &FullViewingKey) -> DecryptedVote {
        let ba = &self.0;
        let nf = Nullifier::from_bytes(&ba.nf).unwrap();
        let ivk = fvk.to_ivk(Scope::External);
        let orchard_domain = OrchardDomain::for_nullifier(nf);
        let ivk = PreparedIncomingViewingKey::new(&ivk);
        let (note, address) = try_compact_note_decryption(&orchard_domain, &ivk, ba).unwrap();
        let value = note.value().inner();
        DecryptedVote {
            address,
            value,
            rho: note.rho(),
            rseed: note.rseed().clone(),
            cmx: ExtractedNoteCommitment::from_bytes(&self.0.cmx).unwrap(),
        }
    }
}

