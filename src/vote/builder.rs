use std::error::Error;

use super::{Hash, Election};

pub struct BallotBuilder {
    election: Election,
    cmxs: Vec<Hash>,
    nfs: Vec<Hash>,
}

impl BallotBuilder {
    pub fn add_note(&self) -> Result<(), Box<dyn Error>> { todo!() }
    pub fn add_candidate(&self) -> Result<(), Box<dyn Error>> { todo!() }
    pub fn build(self) -> Result<(), Box<dyn Error>> { todo!() }
}
