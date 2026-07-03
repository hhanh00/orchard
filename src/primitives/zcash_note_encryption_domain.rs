//! This module implements `Domain` and `BatchDomain` traits from the `zcash_note_encryption`
//! crate and contains the common logic for both Normal and ZSA note flavors.

use alloc::vec::Vec;
use blake2b_simd::{Hash, Params};
use group::ff::PrimeField;

use zcash_note_encryption::{
    note_bytes::NoteBytes, BatchDomain, Domain, EphemeralKeyBytes, OutPlaintextBytes,
    OutgoingCipherKey, OUT_PLAINTEXT_SIZE,
};

use crate::{
    address::Address,
    flavor::{note_version_byte, NoteFlavor, COMPACT_NOTE_SIZE_VANILLA, MEMO_SIZE},
    keys::{
        DiversifiedTransmissionKey, Diversifier, EphemeralPublicKey, EphemeralSecretKey,
        OutgoingViewingKey, PreparedEphemeralPublicKey, PreparedIncomingViewingKey, SharedSecret,
    },
    note::{ExtractedNoteCommitment, Note, RandomSeed, Rho},
    note::AssetBase,
    primitives::orchard_domain::OrchardDomain,
    value::{NoteValue, ValueCommitment},
};

const PRF_OCK_ORCHARD_PERSONALIZATION: &[u8; 16] = b"Zcash_Orchardock";

pub(super) type Memo = [u8; MEMO_SIZE];

/// Defined in [Zcash Protocol Spec § 5.4.2: Pseudo Random Functions][concreteprfs].
///
/// [concreteprfs]: https://zips.z.cash/protocol/nu5.pdf#concreteprfs
pub(super) fn prf_ock_orchard(
    ovk: &OutgoingViewingKey,
    cv: &ValueCommitment,
    cmx_bytes: &[u8; 32],
    ephemeral_key: &EphemeralKeyBytes,
) -> OutgoingCipherKey {
    OutgoingCipherKey(
        Params::new()
            .hash_length(32)
            .personal(PRF_OCK_ORCHARD_PERSONALIZATION)
            .to_state()
            .update(ovk.as_ref())
            .update(&cv.to_bytes())
            .update(cmx_bytes)
            .update(ephemeral_key.as_ref())
            .finalize()
            .as_bytes()
            .try_into()
            .unwrap(),
    )
}

/// Returns true if the note plaintext leadByte matches the version byte for the given flavor.
pub(super) fn is_valid_note_plaintext_lead_byte<F: NoteFlavor>(plaintext: &[u8]) -> bool {
    let expected = note_version_byte::<F>();
    plaintext.first() == Some(&expected)
}

/// Parses the note plaintext (excluding the memo) and extracts the note and address if valid.
pub(super) fn parse_note_plaintext_without_memo<F: NoteFlavor>(
    rho: Rho,
    plaintext: &F::CompactNotePlaintextBytes,
    get_validated_pk_d: impl FnOnce(&Diversifier) -> Option<DiversifiedTransmissionKey>,
) -> Option<(Note, Address)> {
    if !is_valid_note_plaintext_lead_byte::<F>(plaintext.as_ref()) {
        return None;
    }

    let diversifier = Diversifier::from_bytes(
        plaintext.as_ref()[1..12]
            .try_into()
            .unwrap(),
    );

    let value = NoteValue::from_bytes(
        plaintext.as_ref()[12..20]
            .try_into()
            .unwrap(),
    );

    let rseed = Option::from(RandomSeed::from_bytes(
        plaintext.as_ref()[20..COMPACT_NOTE_SIZE_VANILLA]
            .try_into()
            .unwrap(),
        &rho,
    ))?;

    let pk_d = get_validated_pk_d(&diversifier)?;
    let recipient = Address::from_parts(diversifier, pk_d);

    // Extract asset (zatoshi for Normal, custom for ZSA)
    let asset = if F::COMPACT_NOTE_SIZE > COMPACT_NOTE_SIZE_VANILLA {
        // ZSA: asset is at offset 52..84
        let asset_bytes: [u8; 32] = plaintext.as_ref()[COMPACT_NOTE_SIZE_VANILLA..COMPACT_NOTE_SIZE_VANILLA + 32]
            .try_into()
            .ok()?;
        Option::from(AssetBase::from_bytes(&asset_bytes))?
    } else {
        AssetBase::zatoshi()
    };

    let note = Option::from(Note::from_parts(recipient, value, asset, rho, rseed))?;

    Some((note, recipient))
}

/// Builds base note plaintext bytes (without memo) for the given flavor.
/// Builds base note plaintext bytes (without memo) for the given flavor.
///
/// `F::NOTE_PLAINTEXT_SIZE` determines whether this is a Normal (564) or ZSA (596) note.
/// Asset bytes are included at offset 52..84 only for ZSA notes.
pub(super) fn build_base_note_plaintext_bytes<F: NoteFlavor>(
    version: u8,
    note: &Note,
) -> F::NotePlaintextBytes {
    let size = F::NOTE_PLAINTEXT_SIZE;
    // Max-size stack buffer avoids heap allocation and const generic issues.
    let mut buf = [0u8; 596]; // max NOTE_PLAINTEXT_SIZE (ZSA: 84 + 512)
    buf[0] = version;
    buf[1..12].copy_from_slice(note.recipient().diversifier().as_array());
    buf[12..20].copy_from_slice(&note.value().to_bytes());
    buf[20..COMPACT_NOTE_SIZE_VANILLA].copy_from_slice(note.rseed().as_bytes());

    if size > COMPACT_NOTE_SIZE_VANILLA + MEMO_SIZE {
        // ZSA: include asset bytes at offset 52..84
        buf[COMPACT_NOTE_SIZE_VANILLA..COMPACT_NOTE_SIZE_VANILLA + 32]
            .copy_from_slice(&note.asset().to_bytes());
    }

    F::NotePlaintextBytes::from_slice(&buf[..size])
        .expect("NotePlaintextBytes size matches F::NOTE_PLAINTEXT_SIZE")
}

impl<F: NoteFlavor> Domain for OrchardDomain<F> {
    type EphemeralSecretKey = EphemeralSecretKey;
    type EphemeralPublicKey = EphemeralPublicKey;
    type PreparedEphemeralPublicKey = PreparedEphemeralPublicKey;
    type SharedSecret = SharedSecret;
    type SymmetricKey = Hash;
    type Note = Note;
    type Recipient = Address;
    type DiversifiedTransmissionKey = DiversifiedTransmissionKey;
    type IncomingViewingKey = PreparedIncomingViewingKey;
    type OutgoingViewingKey = OutgoingViewingKey;
    type ValueCommitment = ValueCommitment;
    type ExtractedCommitment = ExtractedNoteCommitment;
    type ExtractedCommitmentBytes = [u8; 32];
    type Memo = Memo;

    type NotePlaintextBytes = F::NotePlaintextBytes;
    type NoteCiphertextBytes = F::NoteCiphertextBytes;
    type CompactNotePlaintextBytes = F::CompactNotePlaintextBytes;
    type CompactNoteCiphertextBytes = F::CompactNoteCiphertextBytes;

    fn derive_esk(note: &Self::Note) -> Option<Self::EphemeralSecretKey> {
        Some(note.esk())
    }

    fn get_pk_d(note: &Self::Note) -> Self::DiversifiedTransmissionKey {
        *note.recipient().pk_d()
    }

    fn prepare_epk(epk: Self::EphemeralPublicKey) -> Self::PreparedEphemeralPublicKey {
        PreparedEphemeralPublicKey::new(epk)
    }

    fn ka_derive_public(
        note: &Self::Note,
        esk: &Self::EphemeralSecretKey,
    ) -> Self::EphemeralPublicKey {
        esk.derive_public(note.recipient().g_d())
    }

    fn ka_agree_enc(
        esk: &Self::EphemeralSecretKey,
        pk_d: &Self::DiversifiedTransmissionKey,
    ) -> Self::SharedSecret {
        esk.agree(pk_d)
    }

    fn ka_agree_dec(
        ivk: &Self::IncomingViewingKey,
        epk: &Self::PreparedEphemeralPublicKey,
    ) -> Self::SharedSecret {
        epk.agree(ivk)
    }

    fn kdf(secret: Self::SharedSecret, ephemeral_key: &EphemeralKeyBytes) -> Self::SymmetricKey {
        secret.kdf_orchard(ephemeral_key)
    }

    fn note_plaintext_bytes(note: &Self::Note, memo: &Self::Memo) -> F::NotePlaintextBytes {
        let version = note_version_byte::<F>();
        let mut np = build_base_note_plaintext_bytes::<F>(version, note);
        np.as_mut()[F::COMPACT_NOTE_SIZE..].copy_from_slice(memo);
        np
    }

    fn derive_ock(
        ovk: &Self::OutgoingViewingKey,
        cv: &Self::ValueCommitment,
        cmstar_bytes: &Self::ExtractedCommitmentBytes,
        ephemeral_key: &EphemeralKeyBytes,
    ) -> OutgoingCipherKey {
        prf_ock_orchard(ovk, cv, cmstar_bytes, ephemeral_key)
    }

    fn outgoing_plaintext_bytes(
        note: &Self::Note,
        esk: &Self::EphemeralSecretKey,
    ) -> OutPlaintextBytes {
        let mut op = [0; OUT_PLAINTEXT_SIZE];
        op[..32].copy_from_slice(&note.recipient().pk_d().to_bytes());
        op[32..].copy_from_slice(&esk.0.to_repr());
        OutPlaintextBytes(op)
    }

    fn epk_bytes(epk: &Self::EphemeralPublicKey) -> EphemeralKeyBytes {
        epk.to_bytes()
    }

    fn epk(ephemeral_key: &EphemeralKeyBytes) -> Option<Self::EphemeralPublicKey> {
        EphemeralPublicKey::from_bytes(&ephemeral_key.0).into()
    }

    fn cmstar(note: &Self::Note) -> Self::ExtractedCommitment {
        note.commitment().into()
    }

    fn parse_note_plaintext_without_memo_ivk(
        &self,
        ivk: &Self::IncomingViewingKey,
        plaintext: &F::CompactNotePlaintextBytes,
    ) -> Option<(Self::Note, Self::Recipient)> {
        parse_note_plaintext_without_memo::<F>(self.rho, plaintext, |diversifier| {
            Some(DiversifiedTransmissionKey::derive(ivk, diversifier))
        })
    }

    fn parse_note_plaintext_without_memo_ovk(
        &self,
        pk_d: &Self::DiversifiedTransmissionKey,
        plaintext: &F::CompactNotePlaintextBytes,
    ) -> Option<(Self::Note, Self::Recipient)> {
        parse_note_plaintext_without_memo::<F>(self.rho, plaintext, |_| Some(*pk_d))
    }

    fn split_plaintext_at_memo(
        &self,
        plaintext: &F::NotePlaintextBytes,
    ) -> Option<(F::CompactNotePlaintextBytes, Self::Memo)> {
        let (compact, memo) = plaintext.as_ref().split_at(F::COMPACT_NOTE_SIZE);
        Some((
            F::CompactNotePlaintextBytes::from_slice(compact)?,
            memo.try_into().ok()?,
        ))
    }

    fn extract_pk_d(out_plaintext: &OutPlaintextBytes) -> Option<Self::DiversifiedTransmissionKey> {
        DiversifiedTransmissionKey::from_bytes(out_plaintext.0[0..32].try_into().unwrap()).into()
    }

    fn extract_esk(out_plaintext: &OutPlaintextBytes) -> Option<Self::EphemeralSecretKey> {
        EphemeralSecretKey::from_bytes(out_plaintext.0[32..OUT_PLAINTEXT_SIZE].try_into().unwrap())
            .into()
    }
}

impl<F: NoteFlavor> BatchDomain for OrchardDomain<F> {
    fn batch_kdf<'a>(
        items: impl Iterator<Item = (Option<Self::SharedSecret>, &'a EphemeralKeyBytes)>,
    ) -> Vec<Option<Self::SymmetricKey>> {
        let (shared_secrets, ephemeral_keys): (Vec<_>, Vec<_>) = items.unzip();

        SharedSecret::batch_to_affine(shared_secrets)
            .zip(ephemeral_keys)
            .map(|(secret, ephemeral_key)| {
                secret.map(|dhsecret| SharedSecret::kdf_orchard_inner(dhsecret, ephemeral_key))
            })
            .collect()
    }
}
