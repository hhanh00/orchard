use halo2_proofs::plonk::Error as PlonkError;
use thiserror::Error;

///
#[derive(Error, Debug)]
pub enum VoteError {
    ///
    #[error(transparent)]
    PlonkError(#[from] PlonkError),

    ///
    #[error("Invalid Binding Signature")]
    InvalidBindingSignature,

    ///
    #[error("Decryption Error")]
    DecryptionError,
}
