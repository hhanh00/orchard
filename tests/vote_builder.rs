#![cfg(feature = "vote")]

use incrementalmerkletree::{bridgetree::BridgeTree, Hashable, Tree};
use orchard::{
    builder::Builder,
    bundle::Flags,
    keys::{FullViewingKey, PreparedIncomingViewingKey, Scope, SpendingKey},
    note::ExtractedNoteCommitment,
    note_encryption::OrchardDomain,
    tree::MerkleHashOrchard,
    value::NoteValue,
    vote::{self, Circuit},
    Bundle,
};
use pasta_curves::{
    group::ff::{Field, PrimeField},
    Fp,
};
use rand::rngs::OsRng;
use zcash_note_encryption::try_note_decryption;

/// Build a sorted `nfs` array of `[low, high]` pairs covering the entire Pallas
/// field with gap widths < 2^250 (required by the IntervalGate range check).
///
/// Uses 65 evenly-spaced sentinels at multiples of 2^248, plus p−1.
fn build_covering_nfs() -> Vec<Fp> {
    let pow248 = Fp::from(2u64).pow([248, 0, 0, 0]);
    let mut sentinels: Vec<Fp> = (0u64..=64).map(|k| Fp::from(k) * pow248).collect();
    sentinels.push(-Fp::one()); // p - 1
    sentinels.sort();
    sentinels.dedup();

    let mut nfs = Vec::new();
    for i in 0..sentinels.len() - 1 {
        nfs.push(sentinels[i]);
        nfs.push(sentinels[i + 1]);
    }
    nfs
}

/// Full end-to-end test: create a note via the standard Orchard builder,
/// build a ballot with real IPA proofs (K=15), and verify each proof.
///
/// Exercises the entire vote circuit including the new Poseidon-based NF
/// Merkle tree and IMT non-membership gadget.
///
/// Runtime: ~10 minutes (IPA keygen at K=15 + 2 proof generations).
#[test]
#[ignore] // Run with: cargo test --release -p orchard --test vote_builder -- --ignored --nocapture
fn vote_e2e_prove_verify() {
    let mut rng = OsRng;

    // ---- Keys ----
    let sk = SpendingKey::from_bytes([7u8; 32]).unwrap();
    let fvk = FullViewingKey::from(&sk);
    let recipient = fvk.address_at(0u64, Scope::External);

    // ---- Create a note by building a shielding bundle, then decrypting ----
    eprintln!("Building standard Orchard proving/verifying keys…");
    let orchard_pk = orchard::circuit::ProvingKey::build();
    let _orchard_vk = orchard::circuit::VerifyingKey::build();

    let shielding_bundle: Bundle<_, i64> = {
        let anchor = MerkleHashOrchard::empty_root(32.into()).into();
        let mut builder = Builder::new(Flags::from_parts(false, true), anchor);
        builder
            .add_recipient(None, recipient, NoteValue::from_raw(1000), None)
            .unwrap();
        let unauthorized = builder.build(&mut rng).unwrap();
        let sighash = unauthorized.commitment().into();
        let proven = unauthorized.create_proof(&orchard_pk, &mut rng).unwrap();
        proven.apply_signatures(rng, sighash, &[]).unwrap()
    };

    // Decrypt the output to recover the Note
    let ivk = PreparedIncomingViewingKey::new(&fvk.to_ivk(Scope::External));
    let (note, _, _) = shielding_bundle
        .actions()
        .iter()
        .find_map(|action| {
            let domain = OrchardDomain::for_action(action);
            try_note_decryption(&domain, &ivk, action)
        })
        .expect("should decrypt one output");

    // ---- Build CMX tree with the note ----
    let cmx = ExtractedNoteCommitment::from(note.commitment());
    let leaf = MerkleHashOrchard::from_cmx(&cmx);
    let mut tree = BridgeTree::<MerkleHashOrchard, 32>::new(0);
    tree.append(&leaf);
    let _position = tree.witness().unwrap();

    // CMX as field element for the vote builder
    let cmxs = vec![Fp::from_repr(cmx.to_bytes()).unwrap()];
    let cmx_position = 0u32;

    // ---- NF ranges covering the entire field ----
    let nfs = build_covering_nfs();

    // ---- Build vote proving & verifying keys (K=15, expensive) ----
    eprintln!("Building vote circuit proving key (K=15)…");
    let pk = vote::ProvingKey::<Circuit>::build();
    eprintln!("Building vote circuit verifying key…");
    let vk = vote::VerifyingKey::<Circuit>::build();

    // ---- Vote! ----
    let domain = vote::calculate_domain(b"test-election-e2e");
    eprintln!("Creating ballot (2 actions, 2 proofs)…");
    let ballot = vote::vote(
        domain,
        true, // signature_required
        Some(sk),
        &fvk,
        recipient,
        500, // vote amount (< 1000, so 500 change)
        &[(note, cmx_position)],
        &nfs,
        &cmxs,
        &mut rng,
        &pk,
        &vk,
    )
    .expect("vote() should succeed");

    // ---- Verify ballot structure ----
    // 1 real input → max(1, 2) = 2 actions
    assert_eq!(ballot.data.actions.len(), 2);
    assert_eq!(ballot.witnesses.proofs.len(), 2);
    assert!(ballot.witnesses.sp_signatures.is_some());
    assert_eq!(ballot.witnesses.sp_signatures.as_ref().unwrap().len(), 2);
    assert!(!ballot.witnesses.binding_signature.is_empty());

    eprintln!("Vote e2e prove/verify test passed!");
}
