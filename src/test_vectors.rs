pub(crate) mod asset_base;
pub(crate) mod commitment_tree;
pub(crate) mod issuance_auth_sig;
pub(crate) mod keys;
pub(crate) mod merkle_path;
pub(crate) mod note_encryption_vanilla;
pub(crate) mod note_encryption_zsa;

/// Re-export the vanilla note encryption test vectors under a generic name.
pub(crate) mod note_encryption {
    /// Returns the test vectors for vanilla Orchard note encryption.
    pub(crate) fn test_vectors() -> &'static [super::note_encryption_vanilla::TestVector] {
        super::note_encryption_vanilla::TEST_VECTORS
    }
}
pub(crate) mod zip32;
