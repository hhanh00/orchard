//!
use std::marker::PhantomData;

use halo2_proofs::{
    plonk::{self, SingleVerifier},
    transcript::{Blake2bRead, Blake2bWrite},
};
use pasta_curves::{pallas, vesta};
use rand::RngCore;

const K: u32 = 15;

///
pub trait Statement {
    ///
    type Circuit: plonk::Circuit<pallas::Base> + Default;
    ///
    type Instance: Halo2Instance;
}

///
pub trait Halo2Instance {
    ///
    fn to_halo2_instance(&self) -> Vec<vesta::Scalar>;
}

/// The proving key for the Orchard Action circuit.
#[derive(Debug)]
pub struct ProvingKey<S> {
    params: halo2_proofs::poly::commitment::Params<vesta::Affine>,
    pk: plonk::ProvingKey<vesta::Affine>,
    phantom: PhantomData<S>,
}

impl<S: Statement> ProvingKey<S> {
    /// Builds the proving key.
    pub fn build() -> Self {
        let params = halo2_proofs::poly::commitment::Params::new(K);
        let circuit: S::Circuit = Default::default();

        let vk = plonk::keygen_vk(&params, &circuit).unwrap();
        let pk = plonk::keygen_pk(&params, vk, &circuit).unwrap();

        ProvingKey {
            params,
            pk,
            phantom: PhantomData::default(),
        }
    }
}

/// The verifying key for the Orchard Action circuit.
#[derive(Debug)]
pub struct VerifyingKey<S> {
    pub(crate) params: halo2_proofs::poly::commitment::Params<vesta::Affine>,
    pub(crate) vk: plonk::VerifyingKey<vesta::Affine>,
    phantom: PhantomData<S>,
}

impl<S: Statement> VerifyingKey<S> {
    /// Builds the verifying key.
    pub fn build() -> Self {
        let params = halo2_proofs::poly::commitment::Params::new(K);
        let circuit: S::Circuit = Default::default();

        let vk = plonk::keygen_vk(&params, &circuit).unwrap();

        VerifyingKey {
            params,
            vk,
            phantom: PhantomData::default(),
        }
    }
}

/// A proof of the validity of an Orchard [`Bundle`].
///
/// [`Bundle`]: crate::bundle::Bundle
#[derive(Clone, Debug)]
pub struct Proof<S>(Vec<u8>, PhantomData<S>);

impl<S> AsRef<[u8]> for Proof<S> {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl<S: Statement> Proof<S> {
    /// Creates a proof for the given circuits and instances.
    pub fn create(
        pk: &ProvingKey<S>,
        circuits: &[S::Circuit],
        instances: &[S::Instance],
        mut rng: impl RngCore,
    ) -> Result<Self, plonk::Error> {
        let instances: Vec<_> = instances
            .iter()
            .map(|i| vec![i.to_halo2_instance()])
            .collect();
        let instances: Vec<_> = instances.iter().map(|i| &i[..]).collect();
        let instances: Vec<Vec<_>> = instances
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instances: Vec<_> = instances.iter().map(|i| &i[..]).collect();

        let mut transcript = Blake2bWrite::<_, vesta::Affine, _>::init(vec![]);
        plonk::create_proof(
            &pk.params,
            &pk.pk,
            circuits,
            &instances,
            &mut rng,
            &mut transcript,
        )?;
        Ok(Proof(transcript.finalize(), PhantomData::<S>::default()))
    }

    /// Verifies this proof with the given instances.
    pub fn verify(
        &self,
        vk: &VerifyingKey<S>,
        instances: &[S::Instance],
    ) -> Result<(), plonk::Error> {
        let instances: Vec<_> = instances
            .iter()
            .map(|i| vec![i.to_halo2_instance()])
            .collect();
        let instances: Vec<_> = instances.iter().map(|i| &i[..]).collect();
        let instances: Vec<Vec<_>> = instances
            .iter()
            .map(|i| i.iter().map(|c| &c[..]).collect())
            .collect();
        let instances: Vec<_> = instances.iter().map(|i| &i[..]).collect();

        let strategy = SingleVerifier::new(&vk.params);
        let mut transcript = Blake2bRead::init(&self.0[..]);
        plonk::verify_proof(&vk.params, &vk.vk, strategy, &instances, &mut transcript)
    }

    /// Constructs a new Proof value.
    pub fn new(bytes: Vec<u8>) -> Self {
        Proof(bytes, PhantomData::default())
    }
}
