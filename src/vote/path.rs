use std::mem::swap;

use ff::PrimeField;
use pasta_curves::Fp;

use crate::note::Nullifier;

use super::{Hash, DEPTH};

#[derive(Clone, Default)]
pub struct MerklePath {
    pub value: Hash,
    pub position: u32,
    pub path: [Hash; DEPTH],
    p: usize,
}

pub fn calculate_merkle_paths(
    position_offset: usize,
    positions: &[u32],
    hashes: &[Hash],
) -> (Hash, Vec<MerklePath>) {
    let mut paths = positions
        .iter()
        .map(|p| {
            let rel_p = *p as usize - position_offset;
            MerklePath {
                value: hashes[rel_p],
                position: rel_p as u32,
                path: [Hash::default(); DEPTH],
                p: rel_p,
            }
        })
        .collect::<Vec<_>>();
    let mut er = crate::pob::empty_hash();
    let mut layer = Vec::with_capacity(positions.len() + 2);
    for i in 0..32 {
        if i == 0 {
            layer.extend(hashes);
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
            let h = crate::pob::cmx_hash(i as u8, &layer[j * 2], &layer[j * 2 + 1]);
            next_layer.push(h);
        }

        er = crate::pob::cmx_hash(i as u8, &er, &er);
        if next_layer.len() & 1 == 1 {
            next_layer.push(er);
        }

        swap(&mut layer, &mut next_layer);
    }

    let root = layer[0];
    (root, paths)
}

pub fn make_nf_leaves(nfs: &[Nullifier], my_nfs: &[Nullifier]) -> (Vec<Hash>, Vec<u32>) {
    let mut prev = Fp::zero();
    let mut leaves = vec![];
    let mut nfs_pos = vec![0u32; my_nfs.len()];
    for (pos, r) in nfs.iter().enumerate() {
        let r = r.0;
        // Skip empty ranges when nullifiers are consecutive
        // (with statistically negligible odds)
        if prev < r {
            // Ranges are inclusive of both ends
            let a = prev;
            let b = r - Fp::one();

            for (idx, n) in my_nfs.iter().enumerate() {
                if n.0 >= a && n.0 <= b {
                    nfs_pos[idx] = (2 * pos) as u32;
                }
            }

            leaves.push(a);
            leaves.push(b);
        }
        prev = r + Fp::one();
    }
    if prev != Fp::zero() {
        // overflow when a nullifier == max
        let a = prev;
        let b = Fp::one().neg();
        for (idx, n) in my_nfs.iter().enumerate() {
            if n.0 >= a && n.0 <= b {
                nfs_pos[idx] = (2 * nfs.len()) as u32;
            }
        }

        leaves.push(a);
        leaves.push(b);
    }
    (leaves.iter().map(|v| v.to_repr()).collect(), nfs_pos)
}
