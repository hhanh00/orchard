use rand::{CryptoRng, RngCore};

use crate::{
    keys::SpendAuthorizingKey,
    primitives::redpallas::{self, SpendAuth},
};

impl super::Action {
    /// Signs the Orchard spend with the given spend authorizing key.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn sign<R: RngCore + CryptoRng>(
        &mut self,
        sighash: [u8; 32],
        ask: &SpendAuthorizingKey,
        rng: R,
    ) -> Result<(), SignerError> {
        let alpha = self
            .spend
            .alpha
            .ok_or(SignerError::MissingSpendAuthRandomizer)?;

        let rsk = ask.randomize(&alpha);
        let rk = redpallas::VerificationKey::from(&rsk);

        if self.spend.rk == rk {
            self.spend.spend_auth_sig = Some(rsk.sign(rng, &sighash));
            Ok(())
        } else {
            Err(SignerError::WrongSpendAuthorizingKey)
        }
    }

    /// Applies the given signature to the Orchard spend, if valid.
    ///
    /// It is the caller's responsibility to perform any semantic validity checks on the
    /// PCZT (for example, comfirming that the change amounts are correct) before calling
    /// this method.
    pub fn apply_signature(
        &mut self,
        sighash: [u8; 32],
        signature: redpallas::Signature<SpendAuth>,
    ) -> Result<(), SignerError> {
        if self.spend.rk.verify(&sighash, &signature).is_ok() {
            self.spend.spend_auth_sig = Some(signature);
            Ok(())
        } else {
            Err(SignerError::InvalidExternalSignature)
        }
    }
}

/// Errors that can occur while signing an Orchard action in a PCZT.
#[derive(Debug)]
#[non_exhaustive]
pub enum SignerError {
    /// A provided external signature was not valid for the action's spend.
    InvalidExternalSignature,
    /// The Signer role requires `alpha` to be set.
    MissingSpendAuthRandomizer,
    /// The provided `ask` does not own the action's spent note.
    WrongSpendAuthorizingKey,
}
