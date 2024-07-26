//! Proof of Balance
mod path;
mod circuit;
mod count;
mod proof;
mod ballot;
mod encryption;

pub use path::build_nf_ranges;
pub use circuit::{Circuit as BallotCircuit, ElectionDomain};
pub use count::{Circuit as CountCircuit, CountBuilder, CandidateCountEnvelope};
pub use ballot::{BallotBuilder, BallotEnvelope};
pub use encryption::{EncryptedVote, DecryptedVote};
pub use proof::{ProvingKey, VerifyingKey};

type Hash = [u8; 32];
const DEPTH: usize = 32;


///
#[derive(Clone, Debug)]
pub struct Election {
    name: String,
}
