//! Proof of Balance
mod ballot;
mod builder;
mod circuit;
mod errors;
mod frontier;
mod interval;
mod logical;
mod path;
pub(crate) mod poseidon_merkle;
mod proof;
mod util;
mod validate;

pub use ballot::{Ballot, BallotData};
pub use circuit::Circuit;
pub use errors::VoteError;
pub use frontier::{Frontier, OrchardHash};
pub use path::calculate_merkle_paths;
pub use proof::{ProvingKey, VerifyingKey};
pub use util::{calculate_domain, poseidon_hash};
pub use validate::{try_decrypt_ballot, validate_ballot};
pub use builder::vote;

type Hash = [u8; 32];
const DEPTH: usize = 32;
