//! Proof of Balance
mod path;
mod circuit;
mod count;
mod proof;
mod builder;
mod encryption;

pub use path::build_nf_ranges;
pub use builder::{BallotBuilder, BallotAction};
pub use encryption::{EncryptedVote, DecryptedVote};

type Hash = [u8; 32];
const DEPTH: usize = 32;


///
#[derive(Clone, Debug)]
pub struct Election {
    name: String,
}
