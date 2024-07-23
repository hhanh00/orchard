//! Proof of Balance
mod path;
mod circuit;
mod count;
mod proof;
mod builder;

type Hash = [u8; 32];
const DEPTH: usize = 32;


///
#[derive(Clone, Debug)]
pub struct Election {
    name: String,
}
