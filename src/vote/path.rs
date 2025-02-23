use ff::PrimeField as _;
use incrementalmerkletree::Hashable as _;
use pasta_curves::Fp;

use crate::{note::ExtractedNoteCommitment, tree::MerkleHashOrchard};

use super::DEPTH;

///
pub fn calculate_merkle_paths(
    position_offset: usize,
    positions: &[u32],
    hashes: &[Fp],
) -> (Fp, Vec<MerklePath>) {
    let mut paths = positions
        .iter()
        .map(|p| {
            let rel_p = *p as usize - position_offset;
            MerklePath {
                value: hashes[rel_p],
                position: rel_p as u32,
                path: [Fp::default(); DEPTH],
                p: rel_p,
            }
        })
        .collect::<Vec<_>>();
    let mut er = Fp::from(2);
    let mut layer = Vec::with_capacity(positions.len() + 2);
    for i in 0..32 {
        if i == 0 {
            layer.extend(hashes);
            if layer.is_empty() {
                layer.push(er);
            }
            if layer.len() & 1 == 1 {
                layer.push(er);
            }
        }

        for path in paths.iter_mut() {
            let idx = path.p;
            if idx & 1 == 1 {
                path.path[i] = layer[idx as usize - 1];
            } else {
                path.path[i] = layer[idx as usize + 1];
            }
            path.p /= 2;
        }

        let pairs = layer.len() / 2;
        let mut next_layer = Vec::with_capacity(pairs + 2);

        for j in 0..pairs {
            let h = cmx_hash(i as u8, layer[j * 2], layer[j * 2 + 1]);
            next_layer.push(h);
        }

        er = cmx_hash(i as u8, er, er);
        if next_layer.len() & 1 == 1 {
            next_layer.push(er);
        }

        std::mem::swap(&mut layer, &mut next_layer);
    }

    let root = layer[0];

    // Check the consistency between the merkle paths
    // and the root
    if cfg!(test)
    {
        for p in paths.iter() {
            let mp = p.to_orchard_merkle_tree();
            let mp_root = mp.root(ExtractedNoteCommitment::from_bytes(&p.value.to_repr()).unwrap());
            assert_eq!(root, mp_root.inner());
        }
    }
    (root, paths)
}

///
#[derive(Clone, Default, Debug)]
pub struct MerklePath {
    pub value: Fp,
    pub position: u32,
    pub path: [Fp; DEPTH],
    p: usize,
}

impl MerklePath {
    pub fn to_orchard_merkle_tree(&self) -> crate::tree::MerklePath {
        let auth_path = self
            .path
            .map(|h| MerkleHashOrchard::from_bytes(&h.to_repr()).unwrap());
        let omp = crate::tree::MerklePath::from_parts(self.position, auth_path);
        omp
    }
}

pub fn cmx_hash(level: u8, left: Fp, right: Fp) -> Fp {
    let left = MerkleHashOrchard::from_base(left);
    let right = MerkleHashOrchard::from_base(right);
    let h = MerkleHashOrchard::combine(incrementalmerkletree::Altitude::from(level), &left, &right);
    h.inner()
}
