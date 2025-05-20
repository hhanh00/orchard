use rand::{CryptoRng, RngCore};
use reddsa::orchard::SpendAuth;

use crate::{keys::SpendAuthorizingKey, primitives::redpallas::{self, Signature}};

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

    /// Set the spend authorizing key to the given value.
    ///
    /// It is the caller's responsibility to ensure that the signature is
    /// valid and corresponds to the spend authorizing key used to create the Orchard
    /// spend.
    pub fn spend_auth_sig(&mut self, signature: Signature<SpendAuth>) {
        self.spend.spend_auth_sig = Some(signature);
    }
}

/// Errors that can occur while signing an Orchard action in a PCZT.
#[derive(Debug)]
pub enum SignerError {
    /// The Signer role requires `alpha` to be set.
    MissingSpendAuthRandomizer,
    /// The provided `ask` does not own the action's spent note.
    WrongSpendAuthorizingKey,
}
