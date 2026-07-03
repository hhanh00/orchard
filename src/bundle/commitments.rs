//! Utility functions for computing bundle commitments

use alloc::vec::Vec;
use blake2b_simd::{Hash as Blake2bHash, Params, State};

use crate::{
    bundle::{Authorization, Authorized, Bundle},
    flavor::{NoteFlavor, COMPACT_NOTE_SIZE_VANILLA, MEMO_SIZE},
    sighash_kind::OrchardSighashKind,
};

#[cfg(feature = "zsa-issuance")]
mod issuance;

#[cfg(feature = "zsa-issuance")]
pub(crate) use issuance::{hash_issue_bundle_auth_data, hash_issue_bundle_txid_data};

#[cfg(feature = "zsa-issuance")]
pub use issuance::{hash_issue_bundle_auth_empty, hash_issue_bundle_txid_empty};

pub(crate) const ZCASH_ORCHARD_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrchardHash";
pub(crate) const ZCASH_ORCHARD_ACTION_GROUPS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrcActGHash";
pub(crate) const ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION: &[u8; 16] =
    b"ZTxIdOrcActCHash";
pub(crate) const ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION_V6: &[u8; 16] =
    b"ZTxId6OActC_Hash";
pub(crate) const ZCASH_ORCHARD_ACTIONS_MEMOS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrcActMHash";
pub(crate) const ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION: &[u8; 16] =
    b"ZTxIdOrcActNHash";
pub(crate) const ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION_V6: &[u8; 16] =
    b"ZTxId6OActN_Hash";
pub(crate) const ZCASH_ORCHARD_ZSA_BURN_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxIdOrcBurnHash";
pub(crate) const ZCASH_ORCHARD_SIGS_HASH_PERSONALIZATION: &[u8; 16] = b"ZTxAuthOrchaHash";
pub(crate) const ZCASH_ORCHARD_ACTION_GROUPS_SIGS_HASH_PERSONALIZATION: &[u8; 16] =
    b"ZTxAuthOrcAGHash";
pub(crate) const ZCASH_ORCHARD_SPEND_AUTH_SIGS_HASH_PERSONALIZATION: &[u8; 16] =
    b"ZTxAuthOrSASHash";

pub(crate) fn hasher(personal: &[u8; 16]) -> State {
    Params::new().hash_length(32).personal(personal).to_state()
}

/// Evaluate `orchard_digest` for the bundle as defined in
/// [ZIP-244: Transaction Identifier Non-Malleability][zip244]
/// for NormalFlavor and as defined in
/// [ZIP-246: Digests for the Version 6 Transaction Format][zip246]
/// for ZsaFlavor
///
/// [zip244]: https://zips.z.cash/zip-0244
/// [zip246]: https://zips.z.cash/zip-0246
pub(crate) fn hash_bundle_txid_data<
    A: Authorization,
    V: Copy + Into<i64>,
    Pr: NoteFlavor,
>(
    bundle: &Bundle<A, V, Pr>,
) -> Blake2bHash {
    if Pr::COMPACT_NOTE_SIZE > COMPACT_NOTE_SIZE_VANILLA {
        hash_bundle_txid_data_v6(bundle)
    } else {
        hash_bundle_txid_data_v5(bundle)
    }
}

fn hash_bundle_txid_data_v5<A: Authorization, V: Copy + Into<i64>, Pr: NoteFlavor>(
    bundle: &Bundle<A, V, Pr>,
) -> Blake2bHash {
    let mut h = hasher(ZCASH_ORCHARD_HASH_PERSONALIZATION);

    let mut ch = hasher(ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION);
    let mut mh = hasher(ZCASH_ORCHARD_ACTIONS_MEMOS_HASH_PERSONALIZATION);
    let mut nh = hasher(ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION);

    for action in bundle.actions().iter() {
        ch.update(&action.nullifier().to_bytes());
        ch.update(&action.cmx().to_bytes());
        ch.update(&action.encrypted_note().epk_bytes);
        ch.update(&action.encrypted_note().enc_ciphertext.as_ref()[..Pr::COMPACT_NOTE_SIZE]);

        mh.update(
            &action.encrypted_note().enc_ciphertext.as_ref()
                [Pr::COMPACT_NOTE_SIZE..Pr::COMPACT_NOTE_SIZE + MEMO_SIZE],
        );

        nh.update(&action.cv_net().to_bytes());
        nh.update(&<[u8; 32]>::from(action.rk()));
        nh.update(
            &action.encrypted_note().enc_ciphertext.as_ref()
                [Pr::COMPACT_NOTE_SIZE + MEMO_SIZE..],
        );
        nh.update(&action.encrypted_note().out_ciphertext);
    }

    h.update(ch.finalize().as_bytes());
    h.update(mh.finalize().as_bytes());
    h.update(nh.finalize().as_bytes());

    h.update(&[bundle.flags().to_byte()]);
    h.update(&(*bundle.value_balance()).into().to_le_bytes());
    h.update(&bundle.anchor().to_bytes());
    h.finalize()
}

fn hash_bundle_txid_data_v6<A: Authorization, V: Copy + Into<i64>, Pr: NoteFlavor>(
    bundle: &Bundle<A, V, Pr>,
) -> Blake2bHash {
    let mut h = hasher(ZCASH_ORCHARD_HASH_PERSONALIZATION);
    let mut agh = hasher(ZCASH_ORCHARD_ACTION_GROUPS_HASH_PERSONALIZATION);

    let mut ch = hasher(ZCASH_ORCHARD_ACTIONS_COMPACT_HASH_PERSONALIZATION_V6);
    let mut mh = hasher(ZCASH_ORCHARD_ACTIONS_MEMOS_HASH_PERSONALIZATION);
    let mut nh = hasher(ZCASH_ORCHARD_ACTIONS_NONCOMPACT_HASH_PERSONALIZATION_V6);

    for action in bundle.actions().iter() {
        ch.update(&action.nullifier().to_bytes());
        ch.update(&action.cmx().to_bytes());
        ch.update(&action.encrypted_note().epk_bytes);
        ch.update(&action.encrypted_note().enc_ciphertext.as_ref()[..Pr::COMPACT_NOTE_SIZE]);

        mh.update(
            &action.encrypted_note().enc_ciphertext.as_ref()
                [Pr::COMPACT_NOTE_SIZE..Pr::COMPACT_NOTE_SIZE + MEMO_SIZE],
        );

        nh.update(&action.cv_net().to_bytes());
        nh.update(&<[u8; 32]>::from(action.rk()));
        nh.update(
            &action.encrypted_note().enc_ciphertext.as_ref()
                [Pr::COMPACT_NOTE_SIZE + MEMO_SIZE..],
        );
        nh.update(&action.encrypted_note().out_ciphertext);
    }

    agh.update(ch.finalize().as_bytes());
    agh.update(mh.finalize().as_bytes());
    agh.update(nh.finalize().as_bytes());

    agh.update(&[bundle.flags().to_byte()]);
    agh.update(&bundle.anchor().to_bytes());
    agh.update(&0u32.to_le_bytes());  // expiry_height for ZSA is 0

    let mut burn_hasher = hasher(ZCASH_ORCHARD_ZSA_BURN_HASH_PERSONALIZATION);
    for burn_item in bundle.burn() {
        burn_hasher.update(&burn_item.0.to_bytes());
        burn_hasher.update(&burn_item.1.to_bytes());
    }
    agh.update(burn_hasher.finalize().as_bytes());
    h.update(agh.finalize().as_bytes());

    h.update(&(*bundle.value_balance()).into().to_le_bytes());
    h.finalize()
}

/// Construct the commitment for the absent bundle as defined in
/// [ZIP-244: Transaction Identifier Non-Malleability][zip244]
///
/// [zip244]: https://zips.z.cash/zip-0244
pub fn hash_bundle_txid_empty() -> Blake2bHash {
    hasher(ZCASH_ORCHARD_HASH_PERSONALIZATION).finalize()
}

/// Construct the `orchard_auth_digest` commitment to the authorizing data of an
/// authorized bundle as defined in
/// [ZIP-244: Transaction Identifier Non-Malleability][zip244]
/// for NormalFlavor and as defined in
/// [ZIP-246: Digests for the Version 6 Transaction Format][zip246]
/// for ZsaFlavor
///
/// The `sighash_info_for_kind` closure returns the `SighashInfo` encoding
/// for a given [`OrchardSighashKind`].
///
/// [zip244]: https://zips.z.cash/zip-0244
/// [zip246]: https://zips.z.cash/zip-0246
pub(crate) fn hash_bundle_auth_data<V, Pr: NoteFlavor>(
    bundle: &Bundle<Authorized, V, Pr>,
    sighash_info_for_kind: impl Fn(&OrchardSighashKind) -> Vec<u8>,
) -> Blake2bHash {
    if Pr::COMPACT_NOTE_SIZE > COMPACT_NOTE_SIZE_VANILLA {
        hash_bundle_auth_data_v6(bundle, sighash_info_for_kind)
    } else {
        hash_bundle_auth_data_v5(bundle, sighash_info_for_kind)
    }
}

fn hash_bundle_auth_data_v5<V, Pr: NoteFlavor>(
    bundle: &Bundle<Authorized, V, Pr>,
    _sighash_info_for_kind: impl Fn(&OrchardSighashKind) -> Vec<u8>,
) -> Blake2bHash {
    let mut h = hasher(ZCASH_ORCHARD_SIGS_HASH_PERSONALIZATION);
    h.update(bundle.authorization().proof().as_ref());
    for action in bundle.actions().iter() {
        assert_eq!(
            *action.authorization().sighash_kind(),
            OrchardSighashKind::AllEffecting
        );
        h.update(&<[u8; 64]>::from(action.authorization().sig()));
    }
    assert_eq!(
        *bundle.authorization().binding_signature().sighash_kind(),
        OrchardSighashKind::AllEffecting
    );
    h.update(&<[u8; 64]>::from(
        bundle.authorization().binding_signature().sig(),
    ));
    h.finalize()
}

fn hash_bundle_auth_data_v6<V, Pr: NoteFlavor>(
    bundle: &Bundle<Authorized, V, Pr>,
    sighash_info_for_kind: impl Fn(&OrchardSighashKind) -> Vec<u8>,
) -> Blake2bHash {
    let mut h = hasher(ZCASH_ORCHARD_SIGS_HASH_PERSONALIZATION);
    let mut agh = hasher(ZCASH_ORCHARD_ACTION_GROUPS_SIGS_HASH_PERSONALIZATION);
    agh.update(bundle.authorization().proof().as_ref());
    let mut sash = hasher(ZCASH_ORCHARD_SPEND_AUTH_SIGS_HASH_PERSONALIZATION);
    for action in bundle.actions().iter() {
        let sighash_info = sighash_info_for_kind(action.authorization().sighash_kind());
        sash.update(&get_compact_size(sighash_info.len()));
        sash.update(sighash_info.as_slice());
        sash.update(&<[u8; 64]>::from(action.authorization().sig()));
    }
    agh.update(sash.finalize().as_bytes());
    h.update(agh.finalize().as_bytes());

    let sighash_info =
        sighash_info_for_kind(bundle.authorization().binding_signature().sighash_kind());
    h.update(&get_compact_size(sighash_info.len()));
    h.update(sighash_info.as_slice());
    h.update(&<[u8; 64]>::from(
        bundle.authorization().binding_signature().sig(),
    ));
    h.finalize()
}

/// Construct the `orchard_auth_digest` commitment for an absent bundle as defined in
/// [ZIP-244: Transaction Identifier Non-Malleability][zip244]
///
/// [zip244]: https://zips.z.cash/zip-0244
pub fn hash_bundle_auth_empty() -> Blake2bHash {
    hasher(ZCASH_ORCHARD_SIGS_HASH_PERSONALIZATION).finalize()
}

/// Encodes a size in the CompactSize format.
///
/// This implementation is inspired from `zcash_encoding::CompactSize::write` [code]
/// We cannot use zcash_encoding crate to avoid circular dependency.
///
/// [code]: https://github.com/zcash/librustzcash/blob/8be259c579762f1b0f569453a20c0d0dbeae6c07/components/zcash_encoding/src/lib.rs#L93
pub fn get_compact_size(size: usize) -> Vec<u8> {
    match size {
        s if s < 253 => vec![s as u8],
        s if s <= 0xFFFF => [&[253_u8], &(s as u16).to_le_bytes()[..]].concat(),
        s if s <= 0xFFFFFFFF => [&[254_u8], &(s as u32).to_le_bytes()[..]].concat(),
        s => [&[255_u8], &(s as u64).to_le_bytes()[..]].concat(),
    }
}

#[cfg(all(test, feature = "circuit"))]
mod tests {
    use crate::{
        builder::{Builder, BundleType, UnauthorizedBundle},
        bundle::{
            commitments::{get_compact_size, hash_bundle_auth_data, hash_bundle_txid_data},
            Authorized, Bundle,
        },
        circuit::ProvingKey,
        flavor::{OrchardFlavor, NormalFlavor, ZsaFlavor},
        keys::{FullViewingKey, Scope, SpendingKey},
        note::AssetBase,
        sighash_kind::test_sighash_info_for_kind,
        value::NoteValue,
        Anchor,
    };
    use rand::{rngs::StdRng, SeedableRng};

    fn generate_bundle<FL: OrchardFlavor>(bundle_type: BundleType) -> UnauthorizedBundle<i64, FL> {
        let rng = StdRng::seed_from_u64(5);

        let sk = SpendingKey::from_bytes([7; 32]).unwrap();
        let recipient = FullViewingKey::from(&sk).address_at(0u32, Scope::External);

        let mut builder = Builder::new(bundle_type, Anchor::from_bytes([0; 32]).unwrap());
        builder
            .add_output(
                None,
                recipient,
                NoteValue::from_raw(10),
                AssetBase::zatoshi(),
                [0u8; 512],
            )
            .unwrap();

        builder
            .add_output(
                None,
                recipient,
                NoteValue::from_raw(20),
                AssetBase::zatoshi(),
                [0u8; 512],
            )
            .unwrap();

        builder.build::<i64, FL>(rng).unwrap().unwrap().0
    }

    /// Verifies that the hash for an Orchard Vanilla bundle matches a fixed reference value.
    ///
    /// This is a regression test: inputs are fully deterministic (seeded RNG and fixed
    /// bundle contents), so the resulting digest must remain stable. The reference value
    /// was (re)generated after intentional changes that affect the digest, and
    /// is now treated as the expected output for this implementation.
    #[test]
    fn test_hash_bundle_txid_data_for_orchard_vanilla() {
        let bundle = generate_bundle::<NormalFlavor>(BundleType::DEFAULT);
        let sighash = hash_bundle_txid_data(&bundle);
        assert_eq!(
            sighash.to_hex().as_str(),
            // Bundle hash for Orchard (vanilla) generated using
            // Zcash/Orchard commit: 9d89b504
            "0ac1e319f6761a8561b7bd3fc0907a5c73ed5590a6c210c4d39ffae1d5741875"
        );
    }

    /// Verifies that the hash for an ZsaFlavor bundle matches a fixed reference value.
    ///
    /// This is a regression test: inputs are fully deterministic (seeded RNG and fixed
    /// bundle contents), so the resulting digest must remain stable. The reference value
    /// was (re)generated after intentional changes that affect the digest, and
    /// is now treated as the expected output for this implementation.
    #[test]
    fn test_hash_bundle_txid_data_for_orchard_zsa() {
        let bundle = generate_bundle::<ZsaFlavor>(BundleType::DEFAULT_ZSA);
        let sighash = hash_bundle_txid_data(&bundle);
        assert_eq!(
            sighash.to_hex().as_str(),
            "f84871d872081fa7744cbaf575e342cf81951a9b17818264170243d1551a99ea"
        );
    }

    fn generate_auth_bundle<FL: OrchardFlavor>(
        bundle_type: BundleType,
    ) -> Bundle<Authorized, i64, FL> {
        let mut rng = StdRng::seed_from_u64(6);
        let pk = ProvingKey::build::<FL>();
        let bundle = generate_bundle(bundle_type)
            .create_proof(&pk, &mut rng)
            .unwrap();
        let sighash = bundle.commitment().into();
        bundle.prepare(rng, sighash).finalize().unwrap()
    }

    /// Verifies that the authorizing data commitment for an Orchard Vanilla bundle matches a fixed
    /// reference value.
    ///
    /// This is a regression test: inputs are fully deterministic (seeded RNG and fixed
    /// bundle contents), so the resulting digest must remain stable. The reference value
    /// was (re)generated after intentional changes that affect the digest, and
    /// is now treated as the expected output for this implementation.
    #[test]
    fn test_hash_bundle_auth_data_for_orchard_vanilla() {
        let bundle = generate_auth_bundle::<NormalFlavor>(BundleType::DEFAULT);
        let orchard_auth_digest = hash_bundle_auth_data(&bundle, test_sighash_info_for_kind);
        assert_eq!(
            orchard_auth_digest.to_hex().as_str(),
            // Bundle hash for Orchard (vanilla) generated using
            // Zcash/Orchard commit: 9d89b504
            "37d6c29faa98c2cb54420f3f7cac0477fdb105df1cdfde7adb7fbf68a24e3085"
        );
    }

    /// Verifies that the authorizing data commitment for an ZsaFlavor bundle matches a fixed
    /// reference value.
    ///
    /// This is a regression test: inputs are fully deterministic (seeded RNG and fixed
    /// bundle contents), so the resulting digest must remain stable. The reference value
    /// was (re)generated after intentional changes that affect the digest, and
    /// is now treated as the expected output for this implementation.
    #[test]
    fn test_hash_bundle_auth_data_for_orchard_zsa() {
        let bundle = generate_auth_bundle::<ZsaFlavor>(BundleType::DEFAULT_ZSA);
        let orchard_auth_digest = hash_bundle_auth_data(&bundle, test_sighash_info_for_kind);
        assert_eq!(
            orchard_auth_digest.to_hex().as_str(),
            "15b22e13014abb14bd7902d64957bacf7fe6828c6488e03347d6c2e0d0219275"
        );
    }

    #[test]
    fn test_compact_size() {
        assert_eq!(get_compact_size(0), vec![0]);
        assert_eq!(get_compact_size(1), vec![1]);
        assert_eq!(get_compact_size(252), vec![252]);
        assert_eq!(get_compact_size(253), vec![253, 253, 0]);
        assert_eq!(get_compact_size(254), vec![253, 254, 0]);
        assert_eq!(get_compact_size(255), vec![253, 255, 0]);
        assert_eq!(get_compact_size(65535), vec![253, 255, 255]);
        assert_eq!(get_compact_size(65536), vec![254, 0, 0, 1, 0]);
        assert_eq!(get_compact_size(65537), vec![254, 1, 0, 1, 0]);
        assert_eq!(get_compact_size(33554432), vec![254, 0, 0, 0, 2]);
    }
}
