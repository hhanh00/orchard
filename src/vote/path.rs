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

// ================================================================
// Poseidon-based NF tree (for PIR nullifier integration)
// ================================================================

use super::imt_circuit::NF_MERKLE_DEPTH;
use super::util::poseidon_hash;

/// Pre-computed NF proof data for a single note's nullifier.
///
/// Contains the IMT range `[low, low + width]` that the nullifier falls in,
/// and a depth-29 Poseidon Merkle authentication path from the leaf
/// `Poseidon(low, width)` to the tree root.
#[derive(Clone, Debug)]
pub struct NfProofData {
    /// Lower bound of the IMT gap range.
    pub low: Fp,
    /// Width of the IMT gap range (`high - low`).
    pub width: Fp,
    /// Leaf position in the Poseidon NF Merkle tree.
    pub leaf_pos: u32,
    /// Poseidon Merkle authentication path siblings.
    pub path: [Fp; NF_MERKLE_DEPTH],
}

/// Compute empty subtree roots for the Poseidon-based NF tree.
///
/// `roots[0]` is the empty leaf hash `Poseidon(0, 0)` — i.e. the commitment
/// of a zero-range leaf.  Each subsequent level is built from two copies of
/// the level below: `roots[i] = Poseidon(roots[i-1], roots[i-1])`.
fn poseidon_empty_roots() -> [Fp; NF_MERKLE_DEPTH + 1] {
    let mut roots = [Fp::zero(); NF_MERKLE_DEPTH + 1];
    roots[0] = poseidon_hash(Fp::zero(), Fp::zero());
    for i in 1..=NF_MERKLE_DEPTH {
        roots[i] = poseidon_hash(roots[i - 1], roots[i - 1]);
    }
    roots
}

/// Build a Poseidon Merkle tree from NF ranges and extract authentication paths.
///
/// `nfs` is a sorted flat array where pairs `(nfs[2i], nfs[2i+1])` are
/// `(range_start, range_end)`. Each pair becomes a leaf `Poseidon(low, width)`
/// where `low = nfs[2i]` and `width = nfs[2i+1] - nfs[2i]`.
///
/// `nf_leaf_positions` are the leaf indices (range indices) for which to
/// extract authentication paths.
///
/// Returns `(root, proofs)`.
pub fn calculate_poseidon_nf_paths(
    nf_leaf_positions: &[u32],
    nfs: &[Fp],
) -> (Fp, Vec<NfProofData>) {
    let empty_roots = poseidon_empty_roots();
    let n_ranges = nfs.len() / 2;

    // Convert to (low, width) pairs
    let ranges: Vec<(Fp, Fp)> = (0..n_ranges)
        .map(|i| {
            let low = nfs[2 * i];
            let width = nfs[2 * i + 1] - low;
            (low, width)
        })
        .collect();

    // Initialize proof data
    let mut proofs: Vec<NfProofData> = nf_leaf_positions
        .iter()
        .map(|&pos| NfProofData {
            low: ranges[pos as usize].0,
            width: ranges[pos as usize].1,
            leaf_pos: pos,
            path: [Fp::zero(); NF_MERKLE_DEPTH],
        })
        .collect();

    // Track positions through the tree
    let mut positions: Vec<usize> = nf_leaf_positions.iter().map(|&p| p as usize).collect();

    // Compute leaf hashes
    let mut layer: Vec<Fp> = ranges
        .iter()
        .map(|(low, width)| poseidon_hash(*low, *width))
        .collect();

    // Build tree level by level
    for level in 0..NF_MERKLE_DEPTH {
        // Pad to even length with empty hash for this level
        if layer.len() & 1 == 1 {
            layer.push(empty_roots[level]);
        }

        // Extract siblings for each proof
        for (proof, pos) in proofs.iter_mut().zip(positions.iter()) {
            let sibling_idx = if pos & 1 == 0 { pos + 1 } else { pos - 1 };
            proof.path[level] = if sibling_idx < layer.len() {
                layer[sibling_idx]
            } else {
                empty_roots[level]
            };
        }

        // Compute next level
        let pairs = layer.len() / 2;
        let mut next_layer = Vec::with_capacity(pairs + 1);
        for j in 0..pairs {
            next_layer.push(poseidon_hash(layer[2 * j], layer[2 * j + 1]));
        }
        layer = next_layer;

        // Update positions for next level
        for pos in positions.iter_mut() {
            *pos /= 2;
        }
    }

    let root = if layer.is_empty() {
        empty_roots[NF_MERKLE_DEPTH]
    } else {
        layer[0]
    };

    (root, proofs)
}
