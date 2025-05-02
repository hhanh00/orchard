use halo2_proofs::plonk::Error as PlonkError;
use thiserror::Error;
use alloc::string::String;

///
#[derive(Error, Debug)]
pub enum VoteError {
    ///
    #[error(transparent)]
    PlonkError(#[from] PlonkError),

    ///
    #[error("Invalid Ballot: {0}")]
    InvalidBallot(String),

    ///
    #[error("Invalid Signature: {0}")]
    InvalidSignature(String),

    ///
    #[error("Invalid Key: {0}")]
    InvalidKey(String),

    ///
    #[error("Decryption Error")]
    DecryptionError,

    ///
    #[error("Input Error")]
    InputError,

    ///
    #[error("Not Enough Funds")]
    NotEnoughFunds,
}
