//! Proof of Balance
mod ballot;
mod builder;
mod circuit;
mod errors;
mod frontier;
mod interval;
mod logical;
mod path;
mod proof;
mod util;
mod validate;

pub use ballot::{Ballot, BallotData, BallotAnchors, BallotWitnesses};
pub use circuit::Circuit;
pub use errors::VoteError;
pub use frontier::{Frontier, OrchardHash};
pub use path::calculate_merkle_paths;
pub use proof::{ProvingKey, VerifyingKey};
pub use util::calculate_domain;
pub use validate::{try_decrypt_ballot, validate_ballot};
pub use builder::{encrypt_ballot_action, vote};

type Hash = [u8; 32];
const DEPTH: usize = 32;
